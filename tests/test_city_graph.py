"""Tests for city graph and oracle."""

import pytest

from server.city_graph import CityGraph


def test_city_graph_has_20_nodes():
    g = CityGraph()
    assert len(g.nodes()) == 20


def test_travel_time_symmetric():
    g = CityGraph()
    assert g.travel_time(0, 5) == g.travel_time(5, 0)


def test_travel_time_nonzero():
    g = CityGraph()
    assert g.travel_time(0, 19) > 0
    assert g.travel_time(0, 19) != float("inf")


def test_path_exists():
    g = CityGraph()
    path = g.path(0, 19)
    assert len(path) > 0
    assert path[0] == 0
    assert path[-1] == 19


def test_zone_lookup():
    g = CityGraph()
    assert g.zone(0) == "downtown"
    assert g.zone(8) == "hills"
    assert g.zone(16) == "suburbs"


def test_nodes_in_zone():
    g = CityGraph()
    assert len(g.nodes_in_zone("downtown")) == 4
    assert len(g.nodes_in_zone("highway")) == 4


def test_oracle_assignment_basic():
    g = CityGraph()
    calls = [{"call_id": 1, "call_type": "cardiac", "location": 5, "time_elapsed": 0}]
    units = [{"unit_id": 0, "location": 0}]
    result = g.oracle_assignment(calls, units)
    assert result.get(1) == 0


def test_oracle_prefers_severe():
    g = CityGraph()
    calls = [
        {"call_id": 1, "call_type": "fire", "location": 5, "time_elapsed": 0},
        {"call_id": 2, "call_type": "cardiac", "location": 5, "time_elapsed": 0},
    ]
    units = [{"unit_id": 0, "location": 0}, {"unit_id": 1, "location": 1}]
    result = g.oracle_assignment(calls, units)
    # Cardiac should get the closer unit (0 from node 0)
    assert result.get(2) == 0
