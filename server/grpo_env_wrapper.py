"""GRPO-compatible environment wrapper for DispatchR.

Exposes DispatchR dispatch actions as tools for TRL's GRPOTrainer
with environment_factory support.
"""

import json
import re
from typing import Any, Dict, Optional

from .constants import MAX_STEPS
from .dispatcher_environment import DispatcherEnvironment
from .prompt_utils import format_observation


class DispatchRGRPOEnv:
    """Wrapper that exposes DispatchR as a tool-using environment for TRL GRPO.

    Each public method becomes a tool the LLM can call. The episode advances
    one step per tool call.

    Also supports ``step(text)`` for raw-completion parsing (used when the
    model outputs free-form text rather than structured tool calls).
    """

    def __init__(self, seed: Optional[int] = None, difficulty: str = "learning"):
        self._env = DispatcherEnvironment(seed=seed)
        self._difficulty = difficulty
        self._obs: Optional[Dict[str, Any]] = None
        self._episode_reward: Optional[float] = None
        self._events_log: list = []

    def reset(self, **kwargs: Any) -> str:
        """Reset the environment and return the initial prompt string.

        Args:
            **kwargs: Passed from dataset row (e.g., prompt, seed).

        Returns:
            Initial observation formatted as a text prompt.
        """
        seed = kwargs.get("seed")
        diff = kwargs.get("difficulty", self._difficulty)
        # Apply seed BEFORE reset so the episode is generated with the correct RNG
        if seed is not None:
            import numpy as np
            self._env.rng = np.random.default_rng(seed)
        self._obs = self._env.reset(difficulty=diff)
        self._episode_reward = None
        self._events_log.clear()
        return self._format_prompt(self._obs)

    def _step(self, action: Dict[str, Any]) -> str:
        """Internal step wrapper. Returns event summary."""
        if self._obs and self._obs.get("done"):
            return "Episode has already ended."
        self._obs = self._env.step(action)
        events = self._obs.get("recent_events", [])
        self._events_log.extend(events)
        if self._obs.get("reward") is not None:
            self._episode_reward = self._obs["reward"]
        return "\n".join(events) if events else "Action executed."

    def step(self, action_text: str) -> str:
        """Execute one environment step from a raw text completion.

        TRL's ``environment_factory`` may pass the model's completion text
        directly to ``step()`` when tool-call parsing fails.  We parse the
        text into an action dict and forward it to the underlying env.

        Args:
            action_text: Raw completion string from the LLM.

        Returns:
            Event summary string.
        """
        action = self._parse_action(action_text)
        return self._step(action)

    def _parse_action(self, text: str) -> Dict[str, Any]:
        """Parse a raw completion into an action dict.

        Supports formats like::

            dispatch(unit_id=1, call_id=2)
            hold()
            stage(unit_id=0, location_node=5)
            verify(call_id=3)

        Unknown or malformed text falls back to ``hold()``.
        """
        text = text.strip()
        if not text:
            return {"action_type": "hold"}

        # Prefer a final JSON action if one is present anywhere in the output.
        json_matches = re.findall(r"\{[^{}]*\}", text, flags=re.DOTALL)
        for candidate in reversed(json_matches):
            try:
                action = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(action, dict) and action.get("action_type"):
                return action

        # Qwen3/3.5 puts reasoning first and the action at the END.
        # Look at the last non-empty line to find the actual action.
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        action_line = lines[-1] if lines else text.strip()
        lower = action_line.lower()

        # dispatch(unit_id=1, call_id=2)
        m = re.match(
            r"dispatch\s*\(\s*unit_id\s*=\s*(\d+)\s*,\s*call_id\s*=\s*(\d+)\s*\)",
            lower,
        )
        if m:
            return {"action_type": "dispatch", "unit_id": int(m.group(1)), "call_id": int(m.group(2))}

        # reroute(unit_id=1, call_id=2)
        m = re.match(
            r"reroute\s*\(\s*unit_id\s*=\s*(\d+)\s*,\s*call_id\s*=\s*(\d+)\s*\)",
            lower,
        )
        if m:
            return {"action_type": "reroute", "unit_id": int(m.group(1)), "call_id": int(m.group(2))}

        # stage(unit_id=0, location_node=5)
        m = re.match(
            r"stage\s*\(\s*unit_id\s*=\s*(\d+)\s*,\s*location_node\s*=\s*(\d+)\s*\)",
            lower,
        )
        if m:
            return {"action_type": "stage", "unit_id": int(m.group(1)), "location_node": int(m.group(2))}

        # divert(unit_id=1, hospital_id=0)
        m = re.match(
            r"divert\s*\(\s*unit_id\s*=\s*(\d+)\s*,\s*hospital_id\s*=\s*(\d+)\s*\)",
            lower,
        )
        if m:
            return {"action_type": "divert", "unit_id": int(m.group(1)), "hospital_id": int(m.group(2))}

        # verify(call_id=3)
        m = re.match(r"verify\s*\(\s*call_id\s*=\s*(\d+)\s*\)", lower)
        if m:
            return {"action_type": "verify", "call_id": int(m.group(1))}

        # log(note="...")
        m = re.match(r'log\s*\(\s*note\s*=\s*"([^"]*)"\s*\)', lower)
        if m:
            return {"action_type": "log", "note": m.group(1)}

        # hold() or just "hold"
        if lower.startswith("hold") or lower.startswith("wait"):
            return {"action_type": "hold"}

        # request_mutual_aid()
        if lower.startswith("request_mutual_aid") or lower.startswith("mutual_aid"):
            return {"action_type": "request_mutual_aid"}

        # Fallback: hold
        return {"action_type": "hold"}

    def _format_prompt(self, obs: Dict[str, Any]) -> str:
        """Format observation into a prompt for the LLM.

        Delegates to the shared ``format_observation`` so prompt formatting
        stays consistent across training, inference, and wrapper usage.
        """
        return format_observation(obs)

    # ------------------------------------------------------------------
    # Individual tool methods (used by TRL when tool-call parsing works)
    # ------------------------------------------------------------------

    def dispatch(self, unit_id: int, call_id: int) -> str:
        """Dispatch an idle unit to an active call.

        Args:
            unit_id: The unit to dispatch.
            call_id: The call to respond to.

        Returns:
            Event summary string.
        """
        return self._step({"action_type": "dispatch", "unit_id": unit_id, "call_id": call_id})

    def reroute(self, unit_id: int, call_id: int) -> str:
        """Reroute a unit currently en route to a different call.

        Args:
            unit_id: The unit to reroute.
            call_id: The new call to respond to.

        Returns:
            Event summary string.
        """
        return self._step({"action_type": "reroute", "unit_id": unit_id, "call_id": call_id})

    def stage(self, unit_id: int, location_node: int) -> str:
        """Pre-position an idle unit at a specific node.

        Args:
            unit_id: The unit to move.
            location_node: The node to stage at.

        Returns:
            Event summary string.
        """
        return self._step({"action_type": "stage", "unit_id": unit_id, "location_node": location_node})

    def hold(self) -> str:
        """Wait one step without taking action.

        Returns:
            Event summary string.
        """
        return self._step({"action_type": "hold"})

    def request_mutual_aid(self) -> str:
        """Request external backup unit.

        Returns:
            Event summary string.
        """
        return self._step({"action_type": "request_mutual_aid"})

    def divert(self, unit_id: int, hospital_id: int) -> str:
        """Divert a unit to a specific hospital.

        Args:
            unit_id: The unit to divert.
            hospital_id: The hospital to route to.

        Returns:
            Event summary string.
        """
        return self._step({"action_type": "divert", "unit_id": unit_id, "hospital_id": hospital_id})

    def log(self, note: str) -> str:
        """Append a note to the dispatch log.

        Args:
            note: The note to record.

        Returns:
            Event summary string.
        """
        return self._step({"action_type": "log", "note": note})

    def verify(self, call_id: int) -> str:
        """Run a background check on a call.

        NOTE: This consumes one environment step (advances step_count).

        Args:
            call_id: The call to verify.

        Returns:
            Confidence level string.
        """
        return self._step({"action_type": "verify", "call_id": call_id})

    @property
    def reward(self) -> Optional[float]:
        """Episode reward (available after episode ends)."""
        return self._episode_reward

    @property
    def metrics(self) -> Optional[Dict[str, Any]]:
        """Episode metrics dict (available after episode ends).

        Includes keys:
        - valid_action_rate, invalid_action_rate, hold_rate, dispatch_rate
        - avg_response_time, calls_missed, fatality_count
        """
        if self._episode_reward is None:
            return None
        return self._env.get_ground_truth()

    @property
    def env(self) -> DispatcherEnvironment:
        """Access the underlying environment."""
        return self._env
