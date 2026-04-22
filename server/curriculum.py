"""Adaptive curriculum controller for progressive difficulty."""

from typing import Any, Dict, List, Optional

from .constants import CURRICULUM_PHASES


class CurriculumController:
    """Manages difficulty phases with auto-escalation and de-escalation."""

    def __init__(self):
        self.phase = "warmup"
        self.phases = ["warmup", "learning", "advanced", "expert"]
        self.phase_index = 0
        self.reward_history: List[float] = []
        self.history_window = 10
        self.escalate_threshold = 0.6
        self.deescalate_threshold = 0.3

    def reset(self) -> None:
        self.phase = "warmup"
        self.phase_index = 0
        self.reward_history.clear()

    def record_reward(self, reward: float) -> None:
        """Record episode reward for phase tracking."""
        self.reward_history.append(reward)
        if len(self.reward_history) > self.history_window:
            self.reward_history.pop(0)

    def should_escalate(self) -> bool:
        """Check if agent is ready for next phase."""
        if len(self.reward_history) < self.history_window // 2:
            return False
        mean_reward = sum(self.reward_history) / len(self.reward_history)
        return mean_reward > self.escalate_threshold

    def should_deescalate(self) -> bool:
        """Check if agent is struggling and needs easier phase."""
        if len(self.reward_history) < self.history_window // 2:
            return False
        mean_reward = sum(self.reward_history) / len(self.reward_history)
        return mean_reward < self.deescalate_threshold

    def update_phase(self) -> str:
        """Update phase based on recent performance."""
        if self.should_escalate() and self.phase_index < len(self.phases) - 1:
            self.phase_index += 1
            self.phase = self.phases[self.phase_index]
            self.reward_history.clear()
        elif self.should_deescalate() and self.phase_index > 0:
            self.phase_index -= 1
            self.phase = self.phases[self.phase_index]
            self.reward_history.clear()
        return self.phase

    def get_config(self) -> Dict[str, Any]:
        """Return current phase configuration."""
        return CURRICULUM_PHASES.get(self.phase, CURRICULUM_PHASES["warmup"])
