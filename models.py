"""Data models for the Dead Air emergency dispatch environment."""

from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field


class UnitStatus(BaseModel):
    """Observable status of an emergency unit."""

    unit_id: int = Field(..., description="Unique unit identifier")
    last_known_location: int = Field(..., description="Last known node position")
    last_known_status: str = Field(..., description="One of: idle, en_route, on_scene, returning, out_of_service")
    current_call: Optional[int] = Field(default=None, description="Call ID this unit is assigned to, if any")
    last_update_step: int = Field(default=0, description="Step when this status was last confirmed")


class CallSummary(BaseModel):
    """Observable summary of an emergency call."""

    call_id: int = Field(..., description="Unique call identifier")
    location: int = Field(..., description="Node ID where the call originated")
    reported_type: str = Field(..., description="Reported emergency type: cardiac, trauma, fire, false_alarm")
    caller_tone: str = Field(default="calm", description="Caller tone: calm, agitated, screaming")
    time_elapsed: int = Field(default=0, description="Minutes since call was received")
    assigned_unit: Optional[int] = Field(default=None, description="Unit ID assigned to this call, if any")


class HospitalStatus(BaseModel):
    """Observable status of a hospital."""

    hospital_id: int = Field(..., description="Unique hospital identifier")
    location: int = Field(..., description="Node ID where hospital is located")
    reported_status: str = Field(..., description="accepting or on_divert")


class DispatchAction(Action):
    """Structured dispatch command issued by the agent."""

    action_type: str = Field(..., description="One of: dispatch, reroute, stage, request_mutual_aid, divert, hold, log, verify")
    unit_id: Optional[int] = Field(default=None, description="Unit to act upon")
    call_id: Optional[int] = Field(default=None, description="Call to act upon")
    location_node: Optional[int] = Field(default=None, description="Node to stage at")
    hospital_id: Optional[int] = Field(default=None, description="Hospital to divert to")
    note: Optional[str] = Field(default=None, description="Note to append to dispatch log")


class DispatchObservation(Observation):
    """Observation returned to the dispatch agent after each step."""

    unit_statuses: List[UnitStatus] = Field(default_factory=list, description="Status of all units")
    active_calls: List[CallSummary] = Field(default_factory=list, description="Currently active emergency calls")
    traffic_alerts: List[str] = Field(default_factory=list, description="Active traffic alerts")
    hospital_statuses: List[HospitalStatus] = Field(default_factory=list, description="Status of all hospitals")
    recent_events: List[str] = Field(default_factory=list, description="Event feed from this step")
    mutual_aid_remaining: int = Field(default=0, description="External backup requests remaining")
    step_number: int = Field(default=0, description="Current step in the episode")
    max_steps: int = Field(default=80, description="Maximum steps per episode (8-hour shift)")
    dispatch_log: str = Field(default="", description="Agent's external memory log")
    done: bool = Field(default=False, description="Whether the episode has ended")
    reward: Optional[float] = Field(default=None, description="Episode reward (only set at episode end)")


class EpisodeGroundTruth(BaseModel):
    """Ground truth revealed at episode end for scoring and analysis."""

    calls: List[Dict[str, Any]] = Field(default_factory=list, description="Full call records with hidden fields")
    unit_reliability: Dict[int, float] = Field(default_factory=dict, description="Secret reliability per unit")
    radio_delays: List[Dict[str, Any]] = Field(default_factory=list, description="Delayed status updates")
    caller_bias_by_zone: Dict[int, float] = Field(default_factory=dict, description="True panic modifier distribution per zone")
    hospital_capacity: Dict[int, float] = Field(default_factory=dict, description="True capacity at diversion time")
    city_event: Optional[Dict[str, Any]] = Field(default=None, description="Event that triggered and when")
    optimal_assignments: Dict[int, int] = Field(default_factory=dict, description="Oracle unit assignment per call")
    fatality_count: int = Field(default=0, description="Number of deaths under this dispatch plan")
    coverage_gaps: List[Dict[str, Any]] = Field(default_factory=list, description="Periods with no unit within 10 min of a zone")


class DispatcherState(BaseModel):
    """Full environment state exposed via /state endpoint."""

    episode_id: str = Field(..., description="Unique episode identifier")
    step_count: int = Field(default=0, description="Current step")
    city_id: str = Field(default="canonical", description="City topology identifier")
    shift_start_time: str = Field(default="", description="Simulated shift start")
    total_calls_received: int = Field(default=0, description="Total calls this episode")
    total_calls_resolved: int = Field(default=0, description="Total calls resolved this episode")
    fatalities: int = Field(default=0, description="Deaths so far")
    city_event_triggered: bool = Field(default=False, description="Whether a city event fired")
    event_type: Optional[str] = Field(default=None, description="Type of city event")
