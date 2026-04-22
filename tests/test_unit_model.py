"""Tests for unit state machine and radio delay buffer."""

import pytest

from dead_air.server.city_graph import CityGraph
from dead_air.server.unit_model import RadioDelayBuffer, Unit


def test_unit_initially_idle():
    u = Unit(0, 5)
    assert u.status == "idle"
    assert u.is_available()


def test_unit_dispatch_changes_status():
    u = Unit(0, 0)
    u.dispatch(1, 5, [0, 2, 5])
    assert u.status == "en_route"
    assert u.current_call == 1


def test_unit_tick_moves_along_path():
    g = CityGraph()
    u = Unit(0, 0)
    u.dispatch(1, 5, g.path(0, 5))
    start_loc = u.location
    events = u.tick(g)
    assert u.location != start_loc or len(g.path(0, 5)) <= 2


def test_unit_arrives_at_destination():
    g = CityGraph()
    u = Unit(0, 0)
    path = g.path(0, 1)
    u.dispatch(1, 1, path)
    for _ in range(10):
        events = u.tick(g)
        if u.status == "on_scene":
            break
    assert u.status in ["on_scene", "idle"]


def test_reroute_changes_call():
    g = CityGraph()
    u = Unit(0, 0)
    u.dispatch(1, 5, g.path(0, 5))
    u.reroute(2, 8, g.path(u.location, 8))
    assert u.current_call == 2


def test_stage_has_no_call():
    g = CityGraph()
    u = Unit(0, 0)
    u.stage(5, g.path(0, 5))
    assert u.current_call is None


def test_radio_delay_buffer_delays():
    import numpy as np
    rng = np.random.default_rng(42)
    buf = RadioDelayBuffer(delay_prob=1.0, min_delay=2, max_delay=2, rng=rng)
    buf.submit(1, {"status": "en_route"})
    assert len(buf.release(1)) == 0
    assert len(buf.release(3)) == 1


def test_radio_delay_buffer_immediate():
    import numpy as np
    rng = np.random.default_rng(42)
    buf = RadioDelayBuffer(delay_prob=0.0, rng=rng)
    buf.submit(1, {"status": "idle"})
    assert len(buf.release(1)) == 1


def test_unit_out_of_service():
    g = CityGraph()
    u = Unit(0, 0)
    u.set_out_of_service(3)
    assert u.status == "out_of_service"
    u.tick(g)
    u.tick(g)
    u.tick(g)
    assert u.status == "idle"
