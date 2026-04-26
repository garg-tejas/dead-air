"""DispatchR Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import DispatchAction, DispatchObservation


class EmergencyDispatcherClient(EnvClient[DispatchAction, DispatchObservation, State]):
    """Client for the DispatchR Dispatcher Environment.

    Inherits ``from_hub()``, ``reset()``, ``step()``, and ``close()``
    from :class:`EnvClient`.  Only the payload parsers need to be
    overridden for DispatchR-specific types.
    """

    def _step_payload(self, action: DispatchAction) -> Dict:
        return {
            "action_type": action.action_type,
            "unit_id": action.unit_id,
            "call_id": action.call_id,
            "location_node": action.location_node,
            "hospital_id": action.hospital_id,
            "note": action.note,
        }

    def _parse_result(self, payload: Dict) -> StepResult[DispatchObservation]:
        obs_data = payload.get("observation", {})
        observation = DispatchObservation(
            unit_statuses=obs_data.get("unit_statuses", []),
            active_calls=obs_data.get("active_calls", []),
            traffic_alerts=obs_data.get("traffic_alerts", []),
            hospital_statuses=obs_data.get("hospital_statuses", []),
            recent_events=obs_data.get("recent_events", []),
            mutual_aid_remaining=obs_data.get("mutual_aid_remaining", 0),
            step_number=obs_data.get("step_number", 0),
            max_steps=obs_data.get("max_steps", 80),
            dispatch_log=obs_data.get("dispatch_log", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
