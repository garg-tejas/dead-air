"""Dead Air Environment."""

try:
    from client import EmergencyDispatcherClient
    from models import (
        CallSummary,
        DispatchAction,
        DispatchObservation,
        EpisodeGroundTruth,
        HospitalStatus,
        UnitStatus,
    )
except ImportError:
    # Fallback when package is not installed (running from source)
    from client import EmergencyDispatcherClient
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
