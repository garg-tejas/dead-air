"""Adversarial City Designer: tracks agent weaknesses and biases future scenarios."""

from typing import Any, Dict, Optional

import numpy as np


class AdversarialCityDesigner:
    """Tracks agent failures and escalates difficulty dynamically."""

    def __init__(self, rng: Optional[np.random.Generator] = None, reset_interval: int = 20):
        self.rng = rng or np.random.default_rng()
        self.weaknesses: Dict[str, float] = {}
        self.episode_count = 0
        self.reset_interval = reset_interval

    def reset(self) -> None:
        """Clear weakness history and restart the episode counter.

        Called automatically every ``reset_interval`` episodes to prevent
        stale biases from dominating call generation.
        """
        self.weaknesses.clear()
        self.episode_count = 0

    def record_episode(self, calls: list, fatalities: int, event_name: Optional[str] = None) -> None:
        """Record failures from an episode to update weakness tracker.

        Weaknesses are accumulated over ``reset_interval`` consecutive
        episodes, then cleared. This gives a rolling window of recent
        failures rather than an ever-growing history.
        """
        self.episode_count += 1
        if self.episode_count > self.reset_interval:
            self.reset()
            self.episode_count = 1

        for call in calls:
            if call.get("fatality", False):
                # Track by zone
                zone = call.get("zone", "unknown")
                self.weaknesses[f"zone:{zone}"] = self.weaknesses.get(f"zone:{zone}", 0) + 1

                # Track by call type
                call_type = call.get("call_type", "unknown")
                self.weaknesses[f"type:{call_type}"] = self.weaknesses.get(f"type:{call_type}", 0) + 1

        if fatalities > 0 and event_name:
            self.weaknesses[f"event:{event_name}"] = self.weaknesses.get(f"event:{event_name}", 0) + 1

    def get_bias(self) -> Dict[str, float]:
        """Return adversarial bias dict for call generator."""
        bias = {}
        for key, count in self.weaknesses.items():
            if count >= 2:
                # Normalize to reasonable bias values
                bias[key.split(":")[1]] = min(0.5, count * 0.1)
        return bias

    def get_event_boost(self, event_name: str) -> float:
        """Return probability boost for a specific event type."""
        key = f"event:{event_name}"
        count = self.weaknesses.get(key, 0)
        return min(0.3, count * 0.1)
