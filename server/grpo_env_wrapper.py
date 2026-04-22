"""GRPO-compatible environment wrapper for Dead Air.

Exposes Dead Air dispatch actions as tools for TRL's GRPOTrainer
with environment_factory support.
"""

from typing import Any, Dict, Optional

from .constants import MAX_STEPS
from .dispatcher_environment import DispatcherEnvironment


class DeadAirGRPOEnv:
    """Wrapper that exposes Dead Air as a tool-using environment for TRL GRPO.

    Each public method becomes a tool the LLM can call. The episode advances
    one step per tool call (except verify, which is free).
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
        self._obs = self._env.reset(difficulty=diff)
        if seed is not None:
            self._env.rng = __import__("numpy").random.default_rng(seed)
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

    def _format_prompt(self, obs: Dict[str, Any]) -> str:
        """Format observation into a prompt for the LLM."""
        lines = [
            "# Emergency Dispatch Commander",
            f"Step {obs['step_number']}/{obs['max_steps']}",
            "",
            "## Units",
        ]
        for u in obs.get("unit_statuses", []):
            call_info = f" -> Call {u['current_call']}" if u.get("current_call") else ""
            lines.append(
                f"- Unit {u['unit_id']}: {u['last_known_status']} at Node {u['last_known_location']}{call_info}"
            )
        lines.append("")
        lines.append("## Active Calls")
        for c in obs.get("active_calls", []):
            assigned = f" (Unit {c['assigned_unit']})" if c.get("assigned_unit") else ""
            lines.append(
                f"- Call {c['call_id']}: {c['reported_type']} at Node {c['location']} ({c['caller_tone']}) elapsed={c['time_elapsed']}min{assigned}"
            )
        lines.append("")
        lines.append("## Traffic & Hospitals")
        for alert in obs.get("traffic_alerts", []):
            lines.append(f"- {alert}")
        for h in obs.get("hospital_statuses", []):
            lines.append(f"- Hospital {h['hospital_id']}: {h['reported_status']}")
        lines.append("")
        lines.append(f"Mutual aid remaining: {obs['mutual_aid_remaining']}")
        lines.append("")
        lines.append("Choose your next action using the available tools.")
        return "\n".join(lines)

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
        """Run a background check on a call (free action, does not advance time).

        Args:
            call_id: The call to verify.

        Returns:
            Confidence level string.
        """
        if self._obs and self._obs.get("done"):
            return "Episode has already ended."
        self._obs = self._env.step({"action_type": "verify", "call_id": call_id})
        events = self._obs.get("recent_events", [])
        return "\n".join(events) if events else "Verified."

    @property
    def reward(self) -> Optional[float]:
        """Episode reward (available after episode ends)."""
        return self._episode_reward

    @property
    def env(self) -> DispatcherEnvironment:
        """Access the underlying environment."""
        return self._env
