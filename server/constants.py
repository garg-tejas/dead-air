"""Constants and configuration for the DispatchR environment."""

from typing import Any, Dict, List, Tuple

# Medical response deadlines (minutes)
DEADLINES = {
    "cardiac": 8,
    "trauma": 12,
    "fire": 15,
    "false_alarm": float("inf"),
}

# Call severity base weights for oracle prioritization
SEVERITY_WEIGHTS = {
    "cardiac": 1.0,
    "trauma": 0.7,
    "fire": 0.5,
    "false_alarm": 0.0,
}

# Caller tone mapping from panic modifier
PANIC_TONES = {
    "calm": (0.6, 0.9),
    "agitated": (0.9, 1.2),
    "screaming": (1.2, 1.5),
}

# 20-node canonical city topology
# 5 zones x 4 nodes each
CANONICAL_CITY_EDGES: List[Tuple[int, int, Dict[str, Any]]] = [
    # Downtown cluster (0,1,2,3)
    (0, 1, {"weight": 2, "name": "Main St"}),
    (1, 2, {"weight": 3, "name": "2nd Ave"}),
    (2, 3, {"weight": 2, "name": "3rd Ave"}),
    (0, 3, {"weight": 4, "name": "Broadway"}),
    (0, 2, {"weight": 3, "name": "Market St"}),
    # Highway cluster (4,5,6,7)
    (4, 5, {"weight": 3, "name": "Highway 101 N"}),
    (5, 6, {"weight": 2, "name": "Highway 101 C"}),
    (6, 7, {"weight": 3, "name": "Highway 101 S"}),
    (4, 7, {"weight": 5, "name": "Bypass Rd"}),
    (5, 7, {"weight": 4, "name": "Connector"}),
    # Hills cluster (8,9,10,11)
    (8, 9, {"weight": 2, "name": "Hills Dr"}),
    (9, 10, {"weight": 3, "name": "Ridge Rd"}),
    (10, 11, {"weight": 2, "name": "Summit Ln"}),
    (8, 11, {"weight": 4, "name": "Valley Rd"}),
    (9, 11, {"weight": 3, "name": "Scenic Hwy"}),
    # Industrial cluster (12,13,14,15)
    (12, 13, {"weight": 2, "name": "Factory Ln"}),
    (13, 14, {"weight": 3, "name": "Warehouse Blvd"}),
    (14, 15, {"weight": 2, "name": "Dock Rd"}),
    (12, 15, {"weight": 4, "name": "Commerce St"}),
    (13, 15, {"weight": 3, "name": "Industry Pkwy"}),
    # Suburbs cluster (16,17,18,19)
    (16, 17, {"weight": 2, "name": "Oak St"}),
    (17, 18, {"weight": 3, "name": "Maple Ave"}),
    (18, 19, {"weight": 2, "name": "Pine Ln"}),
    (16, 19, {"weight": 4, "name": "Suburb Cir"}),
    (17, 19, {"weight": 3, "name": "Cedar Rd"}),
    # Cross-cluster links
    (3, 4, {"weight": 5, "name": "Downtown-Hwy Onramp"}),
    (3, 8, {"weight": 6, "name": "Hills Rd"}),
    (7, 12, {"weight": 5, "name": "Industrial Access"}),
    (7, 16, {"weight": 6, "name": "South Hwy"}),
    (11, 16, {"weight": 5, "name": "East Connector"}),
    (11, 12, {"weight": 7, "name": "Mountain Pass"}),
    (15, 19, {"weight": 5, "name": "East Blvd"}),
    (0, 16, {"weight": 8, "name": "Long Route N"}),
    (5, 10, {"weight": 7, "name": "Highway Overpass"}),
    (2, 13, {"weight": 6, "name": "Central Ave"}),
]

# Zone mapping: node_id -> zone name
NODE_ZONES: Dict[int, str] = {
    0: "downtown", 1: "downtown", 2: "downtown", 3: "downtown",
    4: "highway", 5: "highway", 6: "highway", 7: "highway",
    8: "hills", 9: "hills", 10: "hills", 11: "hills",
    12: "industrial", 13: "industrial", 14: "industrial", 15: "industrial",
    16: "suburbs", 17: "suburbs", 18: "suburbs", 19: "suburbs",
}

# Hospital locations (node_id)
HOSPITALS = {
    0: {"name": "County General", "node": 1},
    1: {"name": "St. Mary's", "node": 14},
    2: {"name": "City Medical", "node": 17},
}

# Unit configuration (6 units)
DEFAULT_UNITS = [
    {"unit_id": 0, "location": 0, "speed": 1.0, "reliability": 0.95},
    {"unit_id": 1, "location": 5, "speed": 1.0, "reliability": 0.90},
    {"unit_id": 2, "location": 9, "speed": 1.0, "reliability": 0.92},
    {"unit_id": 3, "location": 13, "speed": 1.0, "reliability": 0.88},
    {"unit_id": 4, "location": 17, "speed": 1.0, "reliability": 0.93},
    {"unit_id": 5, "location": 3, "speed": 1.0, "reliability": 0.91},
]

# Episode parameters
MAX_STEPS = 80
MUTUAL_AID_BUDGET = 2

# Time-of-day traffic multipliers
TRAFFIC_MULTIPLIERS = {
    "morning": 1.2,
    "midday": 1.0,
    "evening": 1.5,
    "night": 0.8,
}

# Coverage threshold (minutes)
COVERAGE_THRESHOLD = 10

# Curriculum phases
CURRICULUM_PHASES = {
    "warmup": {"calls_per_shift": 3, "false_alarm_rate": 0.0, "panic_range": (1.0, 1.0), "event_prob": 0.0, "ghost_rate": 0.0},
    "learning": {"calls_per_shift": 5, "false_alarm_rate": 0.10, "panic_range": (0.8, 1.2), "event_prob": 0.10, "ghost_rate": 0.0},
    "advanced": {"calls_per_shift": 8, "false_alarm_rate": 0.15, "panic_range": (0.7, 1.4), "event_prob": 0.25, "ghost_rate": 0.05},
    "expert": {"calls_per_shift": 12, "false_alarm_rate": 0.20, "panic_range": (0.6, 1.5), "event_prob": 0.40, "ghost_rate": 0.10},
}
