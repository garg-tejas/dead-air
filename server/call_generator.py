"""Call generator for emergency incidents."""

import random
from typing import Any, Dict, List, Optional

import numpy as np

from .constants import DEADLINES, DEFAULT_UNITS, NODE_ZONES


class CallGenerator:
    """Generate emergency calls with Poisson arrivals."""

    def __init__(self, rng: Optional[np.random.Generator] = None):
        self.rng = rng or np.random.default_rng()
        self.call_counter = 0
        self.active_calls: List[Dict[str, Any]] = []
        self.resolved_calls: List[Dict[str, Any]] = []

    def reset(self, difficulty: str = "warmup") -> None:
        self.call_counter = 0
        self.active_calls.clear()
        self.resolved_calls.clear()
        self.difficulty = difficulty

    def _next_call_time(self, current_step: int) -> int:
        """Sample next call arrival via Poisson process."""
        # Mean inter-arrival: 8 steps in warmup, 5 in learning, etc.
        lambdas = {"warmup": 8, "learning": 5, "advanced": 4, "expert": 3}
        lam = lambdas.get(self.difficulty, 5)
        delta = self.rng.poisson(lam=lam) + 1
        return current_step + int(delta)

    def generate_call(self, step: int, city_nodes: List[int]) -> Dict[str, Any]:
        """Generate a new emergency call."""
        self.call_counter += 1
        call_types = ["cardiac", "trauma", "fire"]
        # In MVP, no false alarms yet
        weights = [0.4, 0.35, 0.25]
        call_type = self.rng.choice(call_types, p=weights).item()
        location = int(self.rng.choice(city_nodes))
        deadline = DEADLINES.get(call_type, 12)

        call = {
            "call_id": self.call_counter,
            "location": location,
            "call_type": call_type,
            "reported_type": call_type,
            "deadline": deadline,
            "time_received": step,
            "time_elapsed": 0,
            "assigned_unit": None,
            "resolved": False,
            "fatality": False,
            # Hidden fields (will be used in later phases)
            "severity_modifier": 1.0,
            "panic_modifier": 1.0,
            "is_false_alarm": False,
            "is_ghost": False,
        }
        self.active_calls.append(call)
        return call

    def tick(self, step: int) -> List[str]:
        """Advance active calls by one step. Returns events."""
        events = []
        for call in self.active_calls:
            if not call["resolved"]:
                call["time_elapsed"] = step - call["time_received"]
                if call["time_elapsed"] > call["deadline"] * 2 and call["assigned_unit"] is None:
                    # Missed entirely: mark as fatality for scoring
                    pass  # handled at episode end
        return events

    def resolve_call(self, call_id: int, step: int) -> None:
        """Mark a call as resolved."""
        for call in self.active_calls:
            if call["call_id"] == call_id:
                call["resolved"] = True
                call["time_resolved"] = step
                self.resolved_calls.append(call)
                break

    def get_active(self) -> List[Dict[str, Any]]:
        return [c for c in self.active_calls if not c["resolved"]]

    def get_summary(self) -> List[Dict[str, Any]]:
        """Return observable call summaries."""
        summaries = []
        for call in self.get_active():
            summaries.append({
                "call_id": call["call_id"],
                "location": call["location"],
                "reported_type": call["reported_type"],
                "caller_tone": "calm",  # MVP: neutral callers
                "time_elapsed": call["time_elapsed"],
                "assigned_unit": call["assigned_unit"],
            })
        return summaries
