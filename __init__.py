"""Dead Air Environment."""

try:
    from .client import EmergencyDispatcherClient
    from .models import (
        CallSummary,
        DispatchAction,
        DispatchObservation,
        EpisodeGroundTruth,
        HospitalStatus,
        UnitStatus,
    )
except ImportError:
    # Allow direct import during testing without package context
    EmergencyDispatcherClient = None
    from models import (
        CallSummary,
        DispatchAction,
        DispatchObservation,
        EpisodeGroundTruth,
        HospitalStatus,
        UnitStatus,
    )

__all__ = [
    "DispatchAction",
    "DispatchObservation",
    "CallSummary",
    "UnitStatus",
    "HospitalStatus",
    "EpisodeGroundTruth",
    "EmergencyDispatcherClient",
]
