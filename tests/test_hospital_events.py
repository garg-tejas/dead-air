"""Tests for hospital model and event scheduler."""

import pytest

from dead_air.server.event_scheduler import EventScheduler
from dead_air.server.hospital_model import Hospital, HospitalModel


def test_hospital_capacity_hidden():
    h = Hospital(0, "Test", 1)
    assert 0.6 <= h.capacity <= 1.0


def test_hospital_reported_status():
    h = Hospital(0, "Test", 1)
    h.capacity = 0.9
    assert h.reported_status() == "accepting"
    h.capacity = 0.2
    assert h.reported_status() == "on_divert"


def test_hospital_admission_reduces_capacity():
    h = Hospital(0, "Test", 1)
    old_cap = h.capacity
    h.admit()
    assert h.capacity < old_cap


def test_hospital_model_has_three_hospitals():
    hm = HospitalModel()
    assert len(hm.hospitals) == 3


def test_event_scheduler_may_trigger():
    import numpy as np
    rng = np.random.default_rng(42)
    es = EventScheduler(rng=rng)
    es.reset(event_prob=1.0)
    assert es.event_trigger_step is not None
    assert es.triggered_event is not None


def test_event_scheduler_no_trigger():
    import numpy as np
    rng = np.random.default_rng(42)
    es = EventScheduler(rng=rng)
    es.reset(event_prob=0.0)
    assert es.event_trigger_step is None


def test_event_check_at_trigger_step():
    import numpy as np
    rng = np.random.default_rng(42)
    es = EventScheduler(rng=rng)
    es.reset(event_prob=1.0)
    step = es.event_trigger_step
    alerts = es.check(step)
    assert len(alerts) > 0


def test_event_not_triggered_before():
    import numpy as np
    rng = np.random.default_rng(42)
    es = EventScheduler(rng=rng)
    es.reset(event_prob=1.0)
    alerts = es.check(es.event_trigger_step - 1)
    assert len(alerts) == 0
