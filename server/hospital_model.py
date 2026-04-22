"""Hospital capacity model with hidden state and noisy signals."""

from typing import Any, Dict, List, Optional

import numpy as np

from .constants import HOSPITALS


class Hospital:
    """Hospital with hidden capacity and noisy divert signals."""

    def __init__(self, hospital_id: int, name: str, location: int, rng: Optional[np.random.Generator] = None):
        self.hospital_id = hospital_id
        self.name = name
        self.location = location
        self.rng = rng or np.random.default_rng()
        self.capacity = float(self.rng.uniform(0.6, 1.0))  # fraction of beds available
        self._wait_time_added = 0  # hidden wait time if almost full

    def reset(self) -> None:
        self.capacity = float(self.rng.uniform(0.6, 1.0))
        self._wait_time_added = 0

    def reported_status(self) -> str:
        """Return observable status: accepting or on_divert."""
        if self.capacity > 0.8:
            return "accepting"
        elif self.capacity < 0.4:
            return "on_divert"
        else:
            # Noisy signal: might say accepting but mean almost full
            if self.rng.random() < 0.7:
                return "accepting"
            else:
                return "on_divert"

    def admit(self) -> Dict[str, Any]:
        """Process patient admission. Returns hidden wait time."""
        if self.capacity < 0.5:
            self._wait_time_added = int(self.rng.integers(10, 25))  # 10-25 min wait
        else:
            self._wait_time_added = 0
        # Capacity decreases slightly with each admission
        self.capacity = max(0.2, self.capacity - self.rng.uniform(0.02, 0.08))
        return {"wait_time": self._wait_time_added}

    def to_observable(self) -> Dict[str, Any]:
        return {
            "hospital_id": self.hospital_id,
            "location": self.location,
            "reported_status": self.reported_status(),
        }


class HospitalModel:
    """Manages all hospitals in the city."""

    def __init__(self, rng: Optional[np.random.Generator] = None):
        self.rng = rng or np.random.default_rng()
        self.hospitals: Dict[int, Hospital] = {}
        for hid, info in HOSPITALS.items():
            self.hospitals[hid] = Hospital(hid, info["name"], info["node"], rng=self.rng)

    def reset(self) -> None:
        for h in self.hospitals.values():
            h.reset()

    def get_statuses(self) -> List[Dict[str, Any]]:
        return [h.to_observable() for h in self.hospitals.values()]

    def get_hospital(self, hospital_id: int) -> Optional[Hospital]:
        return self.hospitals.get(hospital_id)
