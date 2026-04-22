"""City graph builder and Dijkstra oracle for Dead Air."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from .constants import CANONICAL_CITY_EDGES, NODE_ZONES


class CityGraph:
    """Canonical 20-node city road network with pre-computed shortest paths."""

    def __init__(self):
        self.graph = nx.Graph()
        for u, v, data in CANONICAL_CITY_EDGES:
            self.graph.add_edge(u, v, **data)
        self._precompute_paths()

    def _precompute_paths(self) -> None:
        """Pre-compute all-pairs shortest path lengths for O(1) oracle lookups."""
        self._path_lengths: Dict[int, Dict[int, float]] = {}
        self._paths: Dict[int, Dict[int, List[int]]] = {}
        for source in self.graph.nodes():
            lengths, paths = nx.single_source_dijkstra(self.graph, source, weight="weight")
            self._path_lengths[int(source)] = {int(k): float(v) for k, v in lengths.items()}
            self._paths[int(source)] = {int(k): [int(n) for n in v] for k, v in paths.items()}

    def save_paths(self, path: Path) -> None:
        """Save pre-computed paths to JSON for fast reload."""
        data = {
            "path_lengths": self._path_lengths,
            "paths": {str(k): {str(kk): vv for kk, vv in v.items()} for k, v in self._paths.items()},
        }
        path.write_text(json.dumps(data))

    def load_paths(self, path: Path) -> None:
        """Load pre-computed paths from JSON."""
        data = json.loads(path.read_text())
        self._path_lengths = {int(k): {int(kk): vv for kk, vv in v.items()} for k, v in data["path_lengths"].items()}
        self._paths = {int(k): {int(kk): vv for kk, vv in v.items()} for k, v in data["paths"].items()}

    def travel_time(self, origin: int, destination: int) -> float:
        """Return shortest travel time in minutes between two nodes."""
        return self._path_lengths.get(origin, {}).get(destination, float("inf"))

    def path(self, origin: int, destination: int) -> List[int]:
        """Return shortest path (list of nodes) between two nodes."""
        return self._paths.get(origin, {}).get(destination, [])

    def nodes(self) -> List[int]:
        return list(self.graph.nodes())

    def edges(self) -> List[Tuple[int, int, Dict]]:
        return list(self.graph.edges(data=True))

    def neighbors(self, node: int) -> List[int]:
        return list(self.graph.neighbors(node))

    def zone(self, node: int) -> str:
        return NODE_ZONES.get(node, "unknown")

    def nodes_in_zone(self, zone: str) -> List[int]:
        return [n for n, z in NODE_ZONES.items() if z == zone]

    def all_nodes_within(self, origin: int, threshold: float) -> List[int]:
        """Return all nodes reachable within threshold minutes."""
        return [n for n, d in self._path_lengths.get(origin, {}).items() if d <= threshold]

    def oracle_assignment(
        self,
        calls: List[Dict[str, Any]],
        idle_units: List[Dict[str, Any]],
    ) -> Dict[int, int]:
        """Compute optimal unit->call assignment with perfect information.

        Greedy assignment by severity / remaining_time.
        Returns: {call_id: unit_id}
        """
        from .constants import DEADLINES, SEVERITY_WEIGHTS

        if not calls or not idle_units:
            return {}

        assignments: Dict[int, int] = {}
        used_units: set = set()

        scored_calls = []
        for call in calls:
            call_type = call.get("call_type", "trauma")
            deadline = DEADLINES.get(call_type, 12)
            severity = SEVERITY_WEIGHTS.get(call_type, 0.5)
            time_elapsed = call.get("time_elapsed", 0)
            remaining = max(1, deadline - time_elapsed)
            urgency = severity / remaining
            scored_calls.append((urgency, call))

        scored_calls.sort(reverse=True, key=lambda x: x[0])

        for _, call in scored_calls:
            call_id = call["call_id"]
            location = call["location"]
            best_unit = None
            best_time = float("inf")
            for unit in idle_units:
                uid = unit["unit_id"]
                if uid in used_units:
                    continue
                t = self.travel_time(unit["location"], location)
                if t < best_time:
                    best_time = t
                    best_unit = uid
            if best_unit is not None:
                assignments[call_id] = best_unit
                used_units.add(best_unit)

        return assignments
