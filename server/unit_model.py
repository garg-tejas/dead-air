"""Unit state machine and radio delay buffer for Dead Air."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class RadioDelayBuffer:
    """Buffer that delays status updates with configurable probability."""

    def __init__(self, delay_prob: float = 0.10, min_delay: int = 2, max_delay: int = 3, rng: Optional[np.random.Generator] = None):
        self.delay_prob = delay_prob
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.rng = rng or np.random.default_rng()
        # Queue of (release_step, status_update)
        self._queue: List[Tuple[int, Dict[str, Any]]] = []

    def submit(self, current_step: int, status: Dict[str, Any]) -> None:
        """Submit a status update; it may be delayed."""
        if self.rng.random() < self.delay_prob:
            delay = int(self.rng.integers(self.min_delay, self.max_delay + 1))
            release_step = current_step + delay
            self._queue.append((release_step, status))
        else:
            # Immediate: release now
            self._queue.append((current_step, status))

    def release(self, current_step: int) -> List[Dict[str, Any]]:
        """Return all status updates whose delay has elapsed."""
        released = []
        remaining = []
        for release_step, status in self._queue:
            if release_step <= current_step:
                released.append(status)
            else:
                remaining.append((release_step, status))
        self._queue = remaining
        return released

    def reset(self) -> None:
        self._queue.clear()


class Unit:
    """Emergency response unit with full state machine."""

    def __init__(self, unit_id: int, location: int, speed: float = 1.0, reliability: float = 0.95):
        self.unit_id = unit_id
        self.location = location
        self.speed = speed
        self.reliability = reliability
        self.status = "idle"  # idle, en_route, on_scene, returning, out_of_service
        self.current_call: Optional[int] = None
        self.target_node: Optional[int] = None
        self.path_remaining: List[int] = []
        self.time_on_scene: int = 0
        self.total_travel_time: int = 0
        self.total_distance: int = 0
        self.breakdown_timer: int = 0
        self._delay_timer: int = 0  # reliability delay (flat tire, wrong turn)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "location": self.location,
            "speed": self.speed,
            "reliability": self.reliability,
            "status": self.status,
            "current_call": self.current_call,
            "target_node": self.target_node,
            "path_remaining": self.path_remaining.copy(),
            "time_on_scene": self.time_on_scene,
            "total_travel_time": self.total_travel_time,
            "total_distance": self.total_distance,
            "breakdown_timer": self.breakdown_timer,
        }

    def is_available(self) -> bool:
        return self.status == "idle"

    def is_active(self) -> bool:
        return self.status != "out_of_service"

    def dispatch(self, call_id: int, target_node: int, path: List[int]) -> None:
        """Dispatch unit to a call."""
        self.status = "en_route"
        self.current_call = call_id
        self.target_node = target_node
        self.path_remaining = path[1:] if len(path) > 1 else []
        self.time_on_scene = 0

    def reroute(self, new_call_id: int, target_node: int, path: List[int]) -> None:
        """Reroute unit to a new call."""
        self.status = "en_route"
        self.current_call = new_call_id
        self.target_node = target_node
        self.path_remaining = path[1:] if len(path) > 1 else []

    def stage(self, target_node: int, path: List[int]) -> None:
        """Stage unit to a pre-position location."""
        self.status = "en_route"
        self.current_call = None
        self.target_node = target_node
        self.path_remaining = path[1:] if len(path) > 1 else []

    def set_out_of_service(self, steps: int) -> None:
        """Take unit out of service (e.g., breakdown)."""
        self.status = "out_of_service"
        self.breakdown_timer = steps
        self.current_call = None
        self.target_node = None
        self.path_remaining = []

    def tick(self, city_graph, rng: Optional[np.random.Generator] = None) -> List[str]:
        """Advance unit by one step. Returns list of events."""
        events: List[str] = []
        rng = rng or np.random.default_rng()

        if self.status == "out_of_service":
            self.breakdown_timer -= 1
            if self.breakdown_timer <= 0:
                self.status = "idle"
                self.breakdown_timer = 0
                events.append(f"Unit {self.unit_id} back in service at node {self.location}")
            return events

        if self.status == "en_route" and self.path_remaining:
            # Reliability check: chance of delay event
            if self._delay_timer > 0:
                self._delay_timer -= 1
                if self._delay_timer == 0:
                    events.append(f"Unit {self.unit_id} delay cleared, resuming")
                else:
                    events.append(f"Unit {self.unit_id} delayed ({self._delay_timer} steps remaining)")
            elif rng.random() > self.reliability:
                # Delay event: flat tire, wrong turn, traffic jam (2-4 steps)
                self._delay_timer = int(rng.integers(2, 5))
                events.append(f"Unit {self.unit_id} delayed: breakdown/traffic ({self._delay_timer} steps)")
            else:
                next_node = self.path_remaining.pop(0)
                edge_data = city_graph.graph.get_edge_data(self.location, next_node)
                travel = edge_data.get("weight", 1) if edge_data else 1
                self.total_travel_time += travel
                self.total_distance += travel
                self.location = next_node
                if not self.path_remaining:
                    # Arrived
                    if self.current_call is not None:
                        self.status = "on_scene"
                        self.time_on_scene = 0
                        events.append(f"Unit {self.unit_id} arrived at call {self.current_call}")
                    else:
                        self.status = "idle"
                        self.target_node = None
                        events.append(f"Unit {self.unit_id} arrived at staging location {self.location}")

        elif self.status == "on_scene":
            self.time_on_scene += 1
            # Scene time: 3 steps for cardiac/trauma, 5 for fire
            scene_time = 3 if self.current_call is not None else 0
            if self.time_on_scene >= scene_time:
                self.status = "returning"
                events.append(f"Unit {self.unit_id} cleared call {self.current_call}")
                # For simplicity, returning units go idle immediately (transport omitted in basic)
                self.status = "idle"
                self.current_call = None
                self.target_node = None

        return events

    def get_observable_status(self) -> Dict[str, Any]:
        """Return status for agent observation."""
        return {
            "unit_id": self.unit_id,
            "last_known_location": self.location,
            "last_known_status": self.status,
            "current_call": self.current_call,
        }
