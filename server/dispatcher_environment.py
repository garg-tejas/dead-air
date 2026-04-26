"""Dispatcher environment: core reset/step loop for DispatchR."""

import re
from typing import Any, Dict, List, Optional
from uuid import uuid4

import numpy as np

from openenv.core import Environment
from openenv.core.env_server.types import State

from .adversarial_designer import AdversarialCityDesigner
from .call_generator import CallGenerator
from .city_graph import CityGraph
from .constants import COVERAGE_THRESHOLD, DEFAULT_UNITS, MAX_STEPS, MUTUAL_AID_BUDGET, NODE_ZONES
from .curriculum import CurriculumController
from .event_scheduler import EventScheduler
from .hospital_model import HospitalModel
from .log_manager import LogManager
from .traffic_model import TrafficModel
from .unit_model import RadioDelayBuffer, Unit


class DispatcherEnvironment(Environment):
    """Full emergency dispatch environment with hidden state and POMDP mechanics."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.city_graph = CityGraph()
        self.traffic_model = TrafficModel(self.city_graph.edges())
        self.call_generator = CallGenerator(rng=self.rng)
        self.hospital_model = HospitalModel(rng=self.rng)
        self.event_scheduler = EventScheduler(rng=self.rng)
        self.log_manager = LogManager()
        self.radio_buffer = RadioDelayBuffer(delay_prob=0.10, min_delay=2, max_delay=3, rng=self.rng)

        self.units: List[Unit] = []
        self.mutual_aid_used = 0
        self._mutual_aid_arrivals: List[Dict[str, Any]] = []  # countdowns for incoming aid units
        self.step_count = 0
        self.difficulty = "warmup"
        self._next_call_step = 1
        self._episode_ended = False
        self.adversarial_designer = AdversarialCityDesigner(rng=self.rng)
        self.curriculum = CurriculumController()
        self._use_internal_curriculum = True  # Set False when CLI manages curriculum

        # Radio delay: track last known status per unit (unit_id -> status dict)
        self._last_known_statuses: Dict[int, Dict[str, Any]] = {}

        # Snapshot unit start locations for oracle time computation
        self._unit_start_locations: Dict[int, int] = {}

        # Per-step coverage accumulator (time-averaged coverage score)
        self._covered_steps = 0

        # Step-level validity tracking (anti-hacking / action quality)
        self._valid_action_count = 0
        self._invalid_action_count = 0
        self._hold_count = 0
        self._dispatch_count = 0

        self._state = State(episode_id=str(uuid4()), step_count=0)

    def reseed(self, seed: int) -> None:
        """Propagate a new RNG seed to all sub-components for deterministic replay."""
        self.rng = np.random.default_rng(seed)
        self.call_generator.rng = self.rng
        self.hospital_model.rng = self.rng
        self.event_scheduler.rng = self.rng
        self.radio_buffer.rng = self.rng
        self.adversarial_designer.rng = self.rng

    def reset(self, difficulty: str = "warmup") -> Dict[str, Any]:
        """Reset environment for a new episode."""
        # Use curriculum-controlled difficulty if not explicitly overridden
        if difficulty == "curriculum":
            difficulty = self.curriculum.phase
        self.difficulty = difficulty
        self.step_count = 0
        self.mutual_aid_used = 0
        self._mutual_aid_arrivals.clear()
        self._episode_ended = False
        self._last_known_statuses.clear()
        self._state = State(episode_id=str(uuid4()), step_count=0)

        # Reset subsystems
        self.units = [Unit(**u) for u in DEFAULT_UNITS]
        self.call_generator.reset(difficulty)
        self.hospital_model.reset()
        self.log_manager.reset()
        self.radio_buffer.reset()
        self.traffic_model.clear_accidents()
        self.traffic_model.set_time_of_day("midday")

        # Configure call generator from curriculum
        from .constants import CURRICULUM_PHASES
        phase = CURRICULUM_PHASES.get(difficulty, CURRICULUM_PHASES["warmup"])
        self.call_generator.configure(
            false_alarm_rate=phase["false_alarm_rate"],
            panic_range=phase["panic_range"],
            ghost_rate=phase["ghost_rate"],
            calls_per_shift=phase.get("calls_per_shift", 999),
        )

        # Randomize caller bias per zone per episode
        zones = set(NODE_ZONES.values())
        self.call_generator.caller_bias_by_zone = {
            z: float(self.rng.uniform(0.7, 1.3)) for z in zones
        }

        # Apply adversarial bias from weakness tracker
        adversarial_bias = self.adversarial_designer.get_bias()

        # Configure events from curriculum
        self.event_scheduler.reset(event_prob=phase["event_prob"])

        # Schedule first call with adversarial bias
        self._next_call_step = self.call_generator.next_call_time(0)

        # Snapshot start locations for oracle time computation
        self._unit_start_locations = {u.unit_id: u.location for u in self.units}

        # Reset per-step coverage accumulator
        self._covered_steps = 0

        # Reset validity tracking
        self._valid_action_count = 0
        self._invalid_action_count = 0
        self._hold_count = 0
        self._dispatch_count = 0
        # Seed last known statuses with initial unit states
        for u in self.units:
            self._last_known_statuses[u.unit_id] = u.get_observable_status()

        # Initial observation
        return self._build_observation(reward=0.0)

    def set_full_visibility(self, enabled: bool = True) -> None:
        """Enable/disable full observability for demo/eval/diagnose.

        When enabled:
        - Radio delay is disabled (status updates are immediate)
        - Last known statuses are seeded with true current statuses
        """
        if enabled:
            self.radio_buffer.delay_prob = 0.0
            for u in self.units:
                self._last_known_statuses[u.unit_id] = u.get_observable_status()
        else:
            self.radio_buffer.delay_prob = 0.10

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one dispatch step."""
        if self._episode_ended:
            return self._build_observation(reward=0.0, done=True)

        events: List[str] = []

        self.step_count += 1
        self._state.step_count = self.step_count

        # Process agent action
        action_events = self._process_action(action)
        events.extend(action_events)

        # Track action validity for anti-hacking metrics
        action_type = action.get("action_type", "hold")
        has_invalid = any(str(e).startswith("Invalid") for e in action_events)
        if has_invalid:
            self._invalid_action_count += 1
        elif action_type == "hold":
            self._hold_count += 1
        elif action_type == "dispatch" and not has_invalid:
            self._valid_action_count += 1
            self._dispatch_count += 1
        elif action_type in ("reroute", "stage", "divert", "request_mutual_aid", "verify", "log"):
            self._valid_action_count += 1

        # Advance units
        for unit in self.units:
            unit_events = unit.tick(self.city_graph, self.rng)
            events.extend(unit_events)
            # Track call arrivals and resolve calls when units clear them
            _ARRIVED_RE = re.compile(r"arrived at call (\d+)")
            _CLEARED_RE = re.compile(r"cleared call (\d+)")
            for evt in unit_events:
                m = _ARRIVED_RE.search(evt)
                if m:
                    call_id = int(m.group(1))
                    for c in self.call_generator.active_calls:
                        if c["call_id"] == call_id:
                            c["time_arrived"] = self.step_count
                            break
                m = _CLEARED_RE.search(evt)
                if m:
                    call_id = int(m.group(1))
                    self.call_generator.resolve_call(call_id, self.step_count)
            # Submit status to radio buffer
            self.radio_buffer.submit(self.step_count, unit.get_observable_status())

        # Release delayed radio updates -> update last known statuses
        released = self.radio_buffer.release(self.step_count)
        for status in released:
            self._last_known_statuses[status["unit_id"]] = status

        # Advance calls
        call_events = self.call_generator.tick(self.step_count)
        events.extend(call_events)

        # Generate new calls if due
        if self.step_count >= self._next_call_step:
            bias = self.adversarial_designer.get_bias()
            call = self.call_generator.generate_call(self.step_count, self.city_graph.nodes(), adversarial_bias=bias)
            events.append(f"New call: {call['reported_type']} at Node {call['location']}. Caller: {call['caller_tone']}.")
            self._next_call_step = self.call_generator.next_call_time(self.step_count)

        # Check city events
        event_alerts = self.event_scheduler.check(self.step_count)
        if event_alerts:
            events.extend(event_alerts)
            active_event = self.event_scheduler.get_active_event(self.step_count)
            if active_event:
                event_effects = self.event_scheduler.apply_event(
                    active_event["name"],
                    self.units,
                    self.traffic_model,
                    self.hospital_model,
                    self.call_generator,
                    self.city_graph,
                )
                events.extend(event_effects)

        # Process mutual aid arrivals
        new_arrivals = []
        for aid in self._mutual_aid_arrivals:
            aid["arrival_step"] -= 1
            if aid["arrival_step"] <= 0:
                new_unit = Unit(**aid["unit_config"])
                self.units.append(new_unit)
                self._last_known_statuses[new_unit.unit_id] = new_unit.get_observable_status()
                events.append(f"Mutual aid Unit {new_unit.unit_id} arrived and is idle")
            else:
                new_arrivals.append(aid)
        self._mutual_aid_arrivals = new_arrivals

        # Mark fatalities for calls that exceeded deadline.
        # Must check BOTH active and resolved calls because a call can be
        # resolved (cleared) after its deadline has already passed.
        for call in self.call_generator.active_calls + self.call_generator.resolved_calls:
            if call.get("fatality"):
                continue
            if call["time_elapsed"] > call.get("effective_deadline", float("inf")):
                call["fatality"] = True

        # Accumulate coverage score per-step (time-averaged)
        if self._compute_coverage(self.units) == 1.0:
            self._covered_steps += 1

        # Check episode end
        done = False
        reward = None
        if self.step_count >= MAX_STEPS:
            done = True
            self._episode_ended = True
            # Compute episode reward
            from .reward import RewardComputer
            rc = RewardComputer(self.city_graph)
            gt = self.get_ground_truth()
            coverage_score = self._covered_steps / self.step_count if self.step_count > 0 else 1.0
            result = rc.compute_episode_reward(
                calls=gt["calls"],
                units=self.units,
                oracle_assignments=gt["optimal_assignments"],
                unit_start_locations=self._unit_start_locations,
                coverage_score=coverage_score,
            )
            # Start from base reward (includes perfect-run bonus if earned)
            reward = result["total_reward"]

            # Apply action-validity shaping (anti-hacking layer)
            # Valid actions are rewarded; invalid actions and excessive holding are penalized
            validity_bonus = 0.0
            invalidity_penalty = 0.0
            idle_penalty = 0.0
            total_actions = self._valid_action_count + self._invalid_action_count + self._hold_count
            if total_actions > 0:
                validity_bonus = (self._valid_action_count / total_actions) * 0.05
                invalidity_penalty = (self._invalid_action_count / total_actions) * 0.05
                idle_penalty = (self._hold_count / total_actions) * 0.02
                reward = reward + validity_bonus - invalidity_penalty - idle_penalty
                reward = max(0.0, reward)  # Floor at 0; total_reward from RewardComputer may already exceed 1.0

            # Attach step-level metrics to result for inspection
            result["validity_bonus"] = validity_bonus if total_actions > 0 else 0.0
            result["invalidity_penalty"] = invalidity_penalty if total_actions > 0 else 0.0
            result["idle_penalty"] = idle_penalty if total_actions > 0 else 0.0
            result["valid_action_rate"] = self._valid_action_count / total_actions if total_actions > 0 else 0.0
            result["dispatch_rate"] = self._dispatch_count / total_actions if total_actions > 0 else 0.0

            # Update curriculum and adversarial designer
            # Skip internal curriculum when training script manages difficulty externally
            if self._use_internal_curriculum:
                self.curriculum.record_reward(reward)
                self.curriculum.update_phase()
            active_event = self.event_scheduler.triggered_event
            event_name = active_event["name"] if active_event else None
            # Tag calls with zones for adversarial tracking
            for call in gt["calls"]:
                call["zone"] = NODE_ZONES.get(call["location"], "unknown")
            self.adversarial_designer.record_episode(
                calls=gt["calls"],
                fatalities=result["fatalities"],
                event_name=event_name,
            )

        obs = self._build_observation(reward=reward, done=done, events=events)
        return obs

    def _process_action(self, action: Dict[str, Any]) -> List[str]:
        """Process a dispatch action. Returns events."""
        events = []
        action_type = action.get("action_type", "hold")

        if action_type == "dispatch":
            uid = action.get("unit_id")
            cid = action.get("call_id")
            unit = self._get_unit(uid)
            call = self._get_call(cid)
            if unit and call and unit.is_available():
                path = self.city_graph.path(unit.location, call["location"])
                unit.dispatch(cid, call["location"], path, call_type=call.get("call_type", "trauma"))
                call["assigned_unit"] = uid
                events.append(f"Dispatched Unit {uid} to Call {cid}")
            else:
                events.append(f"Invalid dispatch: unit {uid} or call {cid} unavailable")

        elif action_type == "reroute":
            uid = action.get("unit_id")
            cid = action.get("call_id")
            unit = self._get_unit(uid)
            call = self._get_call(cid)
            if unit and call and unit.status == "en_route":
                path = self.city_graph.path(unit.location, call["location"])
                unit.reroute(cid, call["location"], path, call_type=call.get("call_type", "trauma"))
                call["assigned_unit"] = uid
                events.append(f"Rerouted Unit {uid} to Call {cid}")
            else:
                events.append(f"Invalid reroute: unit {uid} not en_route")

        elif action_type == "verify":
            cid = action.get("call_id")
            call = self._get_call(cid)
            if call:
                confidence = self.call_generator.verify_call(cid)
                events.append(f"Verified Call {cid}: {confidence} confidence")
            else:
                events.append(f"Invalid verify: call {cid} not found")

        elif action_type == "stage":
            uid = action.get("unit_id")
            loc = action.get("location_node")
            unit = self._get_unit(uid)
            valid_node = loc is not None and loc in self.city_graph.nodes()
            if unit and unit.is_available() and valid_node:
                path = self.city_graph.path(unit.location, loc)
                unit.stage(loc, path)
                events.append(f"Staged Unit {uid} to Node {loc}")
            else:
                events.append(f"Invalid stage: unit {uid} unavailable or invalid location {loc}")

        elif action_type == "request_mutual_aid":
            if self.mutual_aid_used < MUTUAL_AID_BUDGET:
                self.mutual_aid_used += 1
                # Create external unit config, arriving in 6 steps
                aid_unit_id = max(u.unit_id for u in self.units) + 1 if self.units else 100
                self._mutual_aid_arrivals.append({
                    "arrival_step": 6,
                    "unit_config": {
                        "unit_id": aid_unit_id,
                        "location": 0,  # starts at downtown
                        "speed": 1.0,
                        "reliability": 0.90,
                    },
                })
                events.append(f"Mutual aid requested. External Unit {aid_unit_id} arriving in 6 steps.")
            else:
                events.append("Mutual aid budget exhausted")

        elif action_type == "divert":
            uid = action.get("unit_id")
            hid = action.get("hospital_id")
            unit = self._get_unit(uid)
            hospital = self.hospital_model.get_hospital(hid)
            if unit and hospital:
                path = self.city_graph.path(unit.location, hospital.location)
                unit.target_node = hospital.location
                unit.path_remaining = path[1:] if len(path) > 1 else []
                unit.status = "en_route"
                events.append(f"Diverted Unit {uid} to Hospital {hid} ({hospital.name})")
            else:
                events.append(f"Invalid divert: unit {uid} or hospital {hid}")

        elif action_type == "hold":
            events.append("Hold. No action taken.")

        elif action_type == "log":
            note = action.get("note", "")
            if note:
                self.log_manager.append(note)
                events.append(f"Log: {note[:50]}")

        return events

    def _get_unit(self, unit_id: Optional[int]) -> Optional[Unit]:
        if unit_id is None:
            return None
        for u in self.units:
            if u.unit_id == unit_id:
                return u
        return None

    def _get_call(self, call_id: Optional[int]) -> Optional[Dict[str, Any]]:
        if call_id is None:
            return None
        for c in self.call_generator.get_active():
            if c["call_id"] == call_id:
                return c
        return None

    def _build_observation(self, reward: Optional[float] = None, done: bool = False, events: Optional[List[str]] = None) -> Dict[str, Any]:
        """Build agent observation using last known statuses (radio delay applied)."""
        events = events or []

        unit_statuses = []
        for u in self.units:
            # Use last known status from radio buffer if available, else current
            known = self._last_known_statuses.get(u.unit_id, u.get_observable_status())
            unit_statuses.append({
                "unit_id": known["unit_id"],
                "last_known_location": known["last_known_location"],
                "last_known_status": known["last_known_status"],
                "current_call": known["current_call"],
                "last_update_step": known.get("confirmed_at_step", self.step_count),
            })

        obs = {
            "unit_statuses": unit_statuses,
            "active_calls": self.call_generator.get_summary(),
            "traffic_alerts": self.traffic_model.get_alerts(),
            "hospital_statuses": self.hospital_model.get_statuses(),
            "recent_events": events,
            "mutual_aid_remaining": MUTUAL_AID_BUDGET - self.mutual_aid_used,
            "step_number": self.step_count,
            "max_steps": MAX_STEPS,
            "dispatch_log": self.log_manager.get_log(),
            "done": done,
            "reward": reward,
        }
        return obs

    @property
    def state(self) -> State:
        return self._state

    def _compute_coverage(self, units) -> float:
        """Check if every zone has a unit within COVERAGE_THRESHOLD minutes."""
        zones = set(NODE_ZONES.values())
        if not zones:
            return 1.0
        covered_zones = set()
        for zone in zones:
            zone_nodes = [n for n, z in NODE_ZONES.items() if z == zone]
            for unit in units:
                if unit.status == "out_of_service":
                    continue
                for zn in zone_nodes:
                    if self.city_graph.travel_time(unit.location, zn) <= COVERAGE_THRESHOLD:
                        covered_zones.add(zone)
                        break
        return len(covered_zones) / len(zones)

    def get_ground_truth(self) -> Dict[str, Any]:
        """Reveal ground truth at episode end."""
        # Deduplicate: a call could theoretically appear in both lists
        # during a race window between resolve_call() and tick().
        seen_ids = set()
        calls = []
        for c in self.call_generator.active_calls + self.call_generator.resolved_calls:
            cid = c["call_id"]
            if cid in seen_ids:
                continue
            seen_ids.add(cid)
            calls.append({
                "call_id": cid,
                "location": c["location"],
                "call_type": c["call_type"],
                "effective_deadline": c["effective_deadline"],
                "severity_modifier": c["severity_modifier"],
                "panic_modifier": c["panic_modifier"],
                "is_false_alarm": c["is_false_alarm"],
                "is_ghost": c["is_ghost"],
                "time_elapsed": c["time_elapsed"],
                "time_received": c.get("time_received", 0),
                "resolved": c["resolved"],
                "fatality": c.get("fatality", False),
            })

        oracle_assignments = self.city_graph.oracle_assignment(
            calls=[c for c in self.call_generator.active_calls + self.call_generator.resolved_calls if not c.get("is_false_alarm", False) and not c.get("is_ghost", False)],
            idle_units=[
                {"unit_id": u.unit_id, "location": self._unit_start_locations.get(u.unit_id, u.location)}
                for u in self.units
            ],
        )

        # Compute average response time in minutes (graph travel time from
        # assigned unit's start location to call location)
        response_times = []
        for c in calls:
            if not c.get("is_false_alarm", False) and not c.get("is_ghost", False):
                assigned_unit = c.get("assigned_unit")
                if assigned_unit is not None:
                    unit_loc = self._unit_start_locations.get(assigned_unit, 0)
                    rt = self.city_graph.travel_time(unit_loc, c["location"])
                    response_times.append(rt)

        total_actions = self._valid_action_count + self._invalid_action_count + self._hold_count

        return {
            "calls": calls,
            "unit_reliability": {u.unit_id: u.reliability for u in self.units},
            "caller_bias_by_zone": self.call_generator.caller_bias_by_zone,
            "hospital_capacity": {hid: h.capacity for hid, h in self.hospital_model.hospitals.items()},
            "city_event": self.event_scheduler.triggered_event,
            "optimal_assignments": oracle_assignments,
            "fatality_count": sum(1 for c in calls if c.get("fatality", False)),
            # Step-level action quality metrics (for monitoring / judging)
            "valid_action_rate": self._valid_action_count / total_actions if total_actions > 0 else 0.0,
            "invalid_action_rate": self._invalid_action_count / total_actions if total_actions > 0 else 0.0,
            "hold_rate": self._hold_count / total_actions if total_actions > 0 else 0.0,
            "dispatch_rate": self._dispatch_count / total_actions if total_actions > 0 else 0.0,
            "avg_response_time": sum(response_times) / len(response_times) if response_times else 0.0,
            "calls_missed": sum(1 for c in calls if not c.get("resolved", False) and c.get("fatality", False)),
        }
