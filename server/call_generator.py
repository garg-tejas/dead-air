"""Call generator for emergency incidents with panic bias, false alarms, and ghost calls."""

from typing import Any, Dict, List, Optional

import numpy as np

from .constants import DEADLINES, NODE_ZONES


class CallGenerator:
    """Generate emergency calls with Poisson arrivals and hidden state."""

    def __init__(self, rng: Optional[np.random.Generator] = None):
        self.rng = rng or np.random.default_rng()
        self.call_counter = 0
        self.active_calls: List[Dict[str, Any]] = []
        self.resolved_calls: List[Dict[str, Any]] = []
        self.difficulty = "warmup"
        self.ghost_rate = 0.0
        self.false_alarm_rate = 0.0
        self.panic_range = (1.0, 1.0)
        self.heatwave_active = 0
        self.caller_bias_by_zone: Dict[str, float] = {}

    def reset(self, difficulty: str = "warmup") -> None:
        self.call_counter = 0
        self.active_calls.clear()
        self.resolved_calls.clear()
        self.difficulty = difficulty
        self.heatwave_active = 0
        self.caller_bias_by_zone = {}

    def configure(self, false_alarm_rate: float = 0.0, panic_range: tuple = (1.0, 1.0), ghost_rate: float = 0.0) -> None:
        self.false_alarm_rate = false_alarm_rate
        self.panic_range = panic_range
        self.ghost_rate = ghost_rate

    def next_call_time(self, current_step: int) -> int:
        """Sample next call arrival via Poisson process."""
        lambdas = {"warmup": 8, "learning": 5, "advanced": 4, "expert": 3}
        lam = lambdas.get(self.difficulty, 5)
        delta = self.rng.poisson(lam=lam) + 1
        return current_step + int(delta)

    def _sample_panic_modifier(self) -> float:
        """Sample panic modifier from configured range."""
        low, high = self.panic_range
        return float(self.rng.uniform(low, high))

    def _panic_to_tone(self, panic: float) -> str:
        """Convert panic modifier to observable caller tone."""
        if panic < 0.9:
            return "calm"
        elif panic < 1.2:
            return "agitated"
        else:
            return "screaming"

    def generate_call(self, step: int, city_nodes: List[int], adversarial_bias: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Generate a new emergency call with full hidden state."""
        self.call_counter += 1
        adversarial_bias = adversarial_bias or {}

        # Determine if ghost call
        is_ghost = self.rng.random() < self.ghost_rate

        # Determine call type
        call_types = ["cardiac", "trauma", "fire"]
        weights = [0.4, 0.35, 0.25]

        # Heatwave: double cardiac probability
        if self.heatwave_active > 0:
            weights[0] *= 2.0
            weights = [w / sum(weights) for w in weights]

        # Apply adversarial bias to call type
        if "cardiac" in adversarial_bias:
            weights[0] *= (1.0 + adversarial_bias["cardiac"])
            weights = [w / sum(weights) for w in weights]

        call_type = self.rng.choice(call_types, p=weights).item()
        location = int(self.rng.choice(city_nodes))

        # Apply adversarial bias to location (zone)
        zone = NODE_ZONES.get(location, "unknown")
        if zone in adversarial_bias:
            # Reroll location within biased zone with some probability
            if self.rng.random() < min(0.5, adversarial_bias[zone]):
                zone_nodes = [n for n, z in NODE_ZONES.items() if z == zone]
                location = int(self.rng.choice(zone_nodes))

        deadline = DEADLINES.get(call_type, 12)

        # Hidden severity modifier
        severity_modifier = float(self.rng.uniform(0.8, 1.2))
        effective_deadline = deadline * severity_modifier

        # False alarm
        is_false_alarm = (not is_ghost) and (self.rng.random() < self.false_alarm_rate)
        if is_false_alarm:
            effective_deadline = float("inf")
            severity_modifier = 0.5

        # Panic modifier (hidden) - apply per-zone bias
        base_panic = self._sample_panic_modifier()
        zone_bias = self.caller_bias_by_zone.get(zone, 1.0)
        panic_modifier = base_panic * zone_bias
        caller_tone = self._panic_to_tone(panic_modifier)

        # Ghost calls have perfect mimicry
        if is_ghost:
            caller_tone = self._panic_to_tone(self.rng.uniform(0.8, 1.3))
            is_false_alarm = False

        call = {
            "call_id": self.call_counter,
            "location": location,
            "call_type": call_type,
            "reported_type": call_type,
            "deadline": deadline,
            "effective_deadline": effective_deadline,
            "time_received": step,
            "time_elapsed": 0,
            "assigned_unit": None,
            "resolved": False,
            "fatality": False,
            "time_resolved": None,
            # Hidden fields
            "severity_modifier": severity_modifier,
            "panic_modifier": panic_modifier,
            "caller_tone": caller_tone,
            "is_false_alarm": is_false_alarm,
            "is_ghost": is_ghost,
        }
        self.active_calls.append(call)
        return call

    def tick(self, step: int) -> List[str]:
        """Advance active calls by one step. Resolved calls are already removed by resolve_call()."""
        for call in self.active_calls:
            call["time_elapsed"] = step - call["time_received"]
        # Decrement heatwave countdown per-step (not per-call)
        if self.heatwave_active > 0:
            self.heatwave_active -= 1
        return []

    def resolve_call(self, call_id: int, step: int) -> None:
        """Mark a call as resolved and move it from active to resolved list."""
        for i, call in enumerate(self.active_calls):
            if call["call_id"] == call_id:
                call["resolved"] = True
                call["time_resolved"] = step
                self.resolved_calls.append(call)
                self.active_calls.pop(i)
                break

    def get_active(self) -> List[Dict[str, Any]]:
        return [c for c in self.active_calls if not c["resolved"]]

    def get_summary(self) -> List[Dict[str, Any]]:
        """Return observable call summaries (agent-facing)."""
        summaries = []
        for call in self.get_active():
            summaries.append({
                "call_id": call["call_id"],
                "location": call["location"],
                "reported_type": call["reported_type"],
                "caller_tone": call["caller_tone"],
                "time_elapsed": call["time_elapsed"],
                "assigned_unit": call["assigned_unit"],
            })
        return summaries

    def verify_call(self, call_id: int) -> str:
        """Run background check on a call. Returns confidence level."""
        for call in self.active_calls:
            if call["call_id"] == call_id:
                if call["is_ghost"]:
                    # Ghost: biased toward medium/low confidence
                    p = self.rng.random()
                    if p < 0.40:
                        return "high"
                    elif p < 0.75:
                        return "medium"
                    else:
                        return "low"
                else:
                    # Real call: biased toward high confidence
                    p = self.rng.random()
                    if p < 0.80:
                        return "high"
                    elif p < 0.95:
                        return "medium"
                    else:
                        return "low"
        return "high"  # default if call not found
