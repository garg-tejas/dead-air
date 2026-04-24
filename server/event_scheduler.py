"""City event scheduler with dynamic disruptions."""

from typing import Any, Dict, List, Optional

import numpy as np


class EventScheduler:
    """Schedules mid-episode city events that change environment dynamics."""

    def __init__(self, rng: Optional[np.random.Generator] = None):
        self.rng = rng or np.random.default_rng()
        self.event_templates = [
            {
                "name": "bridge_collapse",
                "description": "Bridge collapse on Route 7. All crossings delayed.",
                "min_step": 20,
                "max_step": 60,
            },
            {
                "name": "hospital_divert",
                "description": "County General mass casualty incident. On divert.",
                "min_step": 20,
                "max_step": 60,
            },
            {
                "name": "heatwave",
                "description": "Heatwave alert. Cardiac call probability 2x for next 20 steps.",
                "min_step": 15,
                "max_step": 50,
            },
            {
                "name": "unit_breakdown",
                "description": "Unit breakdown. Out of service for 10 steps.",
                "min_step": 10,
                "max_step": 60,
            },
        ]
        self.triggered_event: Optional[Dict[str, Any]] = None
        self.event_trigger_step: Optional[int] = None

    def reset(self, event_prob: float = 0.0) -> None:
        self.triggered_event = None
        self.event_trigger_step = None
        if self.rng.random() < event_prob:
            template = self.rng.choice(self.event_templates)
            step = int(self.rng.integers(template["min_step"], template["max_step"] + 1))
            self.event_trigger_step = step
            self.triggered_event = {
                "name": template["name"],
                "description": template["description"],
                "trigger_step": step,
            }

    def check(self, step: int) -> List[str]:
        """Check if an event should trigger at this step."""
        events = []
        if self.event_trigger_step == step and self.triggered_event:
            events.append(f"ALERT: {self.triggered_event['description']}")
        return events

    def is_active(self, step: int) -> bool:
        """Returns True if the city event has triggered by this step."""
        if self.event_trigger_step is None:
            return False
        return step >= self.event_trigger_step

    def get_active_event(self, step: int) -> Optional[Dict[str, Any]]:
        if self.is_active(step):
            return self.triggered_event
        return None

    def apply_event(
        self,
        event_name: str,
        units: List[Any],
        traffic_model: Any,
        hospital_model: Any,
        call_generator: Any,
        city_graph: Any = None,
    ) -> List[str]:
        """Apply the effects of a triggered event."""
        events = []
        if event_name == "bridge_collapse":
            # Delay edges crossing between downtown/highway and hills/industrial
            # Simplified: add accident on key bridge edges
            traffic_model.add_accident(5, 10, 3.0)
            traffic_model.add_accident(2, 13, 3.0)
            # Actually update city graph so units route around the collapsed bridge
            if city_graph is not None:
                city_graph.update_edge_weight(5, 10, 21.0)  # was 7, now 3x
                city_graph.update_edge_weight(2, 13, 18.0)  # was 6, now 3x
            events.append("Bridge collapse: Route 7 and Central Ave crossings x3")
        elif event_name == "hospital_divert":
            # Force County General on divert
            if hospital_model.hospitals:
                hg = hospital_model.hospitals.get(0)
                if hg:
                    hg.capacity = 0.1
                    events.append("County General forced on divert")
        elif event_name == "heatwave":
            # Increase cardiac probability for next 20 steps
            call_generator.heatwave_active = 20
            events.append("Heatwave active: cardiac probability doubled for 20 steps")
        elif event_name == "unit_breakdown":
            # Random unit goes out of service
            if units:
                unit = self.rng.choice(units)
                unit.set_out_of_service(10)
                events.append(f"Unit {unit.unit_id} breakdown: out of service 10 steps")
        return events
