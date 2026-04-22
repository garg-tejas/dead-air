"""Traffic model for time-varying edge weights and dynamic disruptions."""

from typing import Dict, List, Tuple


class TrafficModel:
    """Dynamic traffic with time-of-day multipliers and accident injection."""

    def __init__(self, base_edges: List[Tuple[int, int, Dict]]):
        self.base_weights: Dict[Tuple[int, int], float] = {}
        for u, v, data in base_edges:
            w = data.get("weight", 1)
            self.base_weights[(u, v)] = w
            self.base_weights[(v, u)] = w
        self.active_accidents: List[Tuple[int, int, float]] = []  # (u, v, multiplier)
        self.time_of_day = "midday"
        self.global_multiplier = 1.0

    def set_time_of_day(self, time: str) -> None:
        """Set time period: morning, midday, evening, night."""
        from .constants import TRAFFIC_MULTIPLIERS
        self.time_of_day = time
        self.global_multiplier = TRAFFIC_MULTIPLIERS.get(time, 1.0)

    def current_weight(self, u: int, v: int) -> float:
        """Return current travel time for edge (u, v)."""
        base = self.base_weights.get((u, v), 1.0)
        mult = self.global_multiplier
        for au, av, m in self.active_accidents:
            if (u == au and v == av) or (u == av and v == au):
                mult = max(mult, m)
        return base * mult

    def add_accident(self, u: int, v: int, multiplier: float = 2.0) -> None:
        self.active_accidents.append((u, v, multiplier))

    def clear_accidents(self) -> None:
        self.active_accidents.clear()

    def get_alerts(self) -> List[str]:
        alerts = []
        for u, v, m in self.active_accidents:
            edge_name = f"edge {u}-{v}"
            alerts.append(f"Accident on {edge_name}: travel time x{m}")
        return alerts

    def apply_bridge_collapse(self, route_nodes: List[int], multiplier: float = 3.0) -> None:
        """Apply massive delay to all edges crossing a route."""
        for i in range(len(route_nodes) - 1):
            self.add_accident(route_nodes[i], route_nodes[i + 1], multiplier)

    def to_graph_weights(self) -> Dict[Tuple[int, int], float]:
        """Return current weights for all edges."""
        weights = {}
        for (u, v), base in self.base_weights.items():
            weights[(u, v)] = self.current_weight(u, v)
        return weights
