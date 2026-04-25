"""Reward computation for DispatchR environment."""

from typing import Any, Dict, List, Optional

from .constants import COVERAGE_THRESHOLD


class RewardComputer:
    """Compute episode reward from 3 components plus perfect-run bonus."""

    def __init__(self, city_graph):
        self.city_graph = city_graph

    def compute_episode_reward(
        self,
        calls: List[Dict[str, Any]],
        units: List[Any],
        oracle_assignments: Dict[int, int],
        unit_start_locations: Optional[Dict[int, int]] = None,
        coverage_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Compute full episode reward and metrics.

        Returns dict with:
        - episode_reward: float (0-1)
        - response_score: float
        - fatality_penalty: float
        - coverage_score: float
        - fatalities: int
        """
        response_scores = []
        fatality_penalties = []
        fatalities = 0

        for call in calls:
            if call.get("is_false_alarm", False) or call.get("is_ghost", False):
                continue

            actual_time = call.get("time_elapsed", 0)
            if call.get("resolved", False):
                # Use time_arrived (exclude on-scene time) for response score
                if call.get("time_arrived") is not None:
                    actual_time = call["time_arrived"] - call["time_received"]
                elif call.get("time_resolved"):
                    actual_time = call["time_resolved"] - call["time_received"]

            effective_deadline = call.get("effective_deadline", float("inf"))
            oracle_time = self._get_oracle_time(call, oracle_assignments, unit_start_locations)

            # Response time score
            if oracle_time > 0 and actual_time > 0:
                ratio = oracle_time / actual_time
            else:
                ratio = 1.0
            response_score = min(1.0, ratio)

            # If exceeded deadline, response score is 0 and triggers fatality
            if actual_time > effective_deadline:
                response_score = 0.0
                fatality_penalties.append(-0.5)
                fatalities += 1
            else:
                fatality_penalties.append(0.0)

            response_scores.append(response_score)

        mean_response = sum(response_scores) / max(1, len(response_scores))
        total_fatality_penalty = sum(fatality_penalties)
        fatality_component = 1.0 + total_fatality_penalty  # 1.0 if zero deaths

        # Coverage score (use provided time-averaged score if available, else snapshot)
        if coverage_score is None:
            coverage_score = self._compute_coverage(units)

        # Base episode reward
        episode_reward = (
            0.50 * mean_response
            + 0.30 * fatality_component
            + 0.20 * coverage_score
        )
        episode_reward = max(0.0, min(1.0, episode_reward))

        # Perfect-run bonus: reward excellence beyond the baseline
        perfect_run_bonus = 0.0
        if fatalities == 0 and mean_response > 0.5:
            perfect_run_bonus = 0.5

        return {
            "episode_reward": episode_reward,
            "perfect_run_bonus": perfect_run_bonus,
            "total_reward": episode_reward + perfect_run_bonus,
            "response_score": mean_response,
            "fatality_penalty": total_fatality_penalty,
            "coverage_score": coverage_score,
            "fatalities": fatalities,
        }

    def _get_oracle_time(
        self,
        call: Dict[str, Any],
        oracle_assignments: Dict[int, int],
        unit_start_locations: Optional[Dict[int, int]] = None,
    ) -> float:
        """Get oracle response time for a call from the assigned unit's start location.

        NOTE: Uses episode-start locations as an approximation. See
        ``oracle_assignment()`` docstring for details.
        """
        call_id = call["call_id"]
        if call_id in oracle_assignments:
            unit_id = oracle_assignments[call_id]
            unit_loc = unit_start_locations.get(unit_id, 0) if unit_start_locations else 0
            return self.city_graph.travel_time(unit_loc, call["location"])
        return 1.0

    def _compute_coverage(self, units: List[Any]) -> float:
        """Check if every zone has a unit within COVERAGE_THRESHOLD minutes."""
        from .constants import NODE_ZONES

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

    def compute_evaluation_metrics(
        self,
        calls: List[Dict[str, Any]],
        event_triggered: bool,
        pre_event_fatalities: int,
        post_event_fatalities: int,
        total_distance: float,
        max_distance: float,
    ) -> Dict[str, float]:
        """Compute evaluation metrics (not used for training)."""
        # Caller inference score
        caller_inference = 0.0
        for call in calls:
            panic = call.get("panic_modifier", 1.0)
            dispatched_quickly = call.get("time_elapsed", 999) <= 2
            held_or_downgraded = not call.get("assigned_unit") and call.get("time_elapsed", 0) > 5

            if panic < 0.8 and dispatched_quickly:
                caller_inference += 0.05
            if panic > 1.3 and held_or_downgraded:
                caller_inference += 0.03

        # Event adaptation score
        event_adaptation = 1.0
        if event_triggered and post_event_fatalities > 0:
            event_adaptation = pre_event_fatalities / max(1, post_event_fatalities)

        # Efficiency score
        efficiency = 1.0
        if max_distance > 0:
            efficiency = 1.0 - (total_distance / max_distance)

        return {
            "caller_inference_score": min(1.0, caller_inference),
            "event_adaptation_score": event_adaptation,
            "efficiency_score": efficiency,
        }
