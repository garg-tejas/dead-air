"""Quick test to verify the improved system prompt and observation formatting.
Does NOT import train_unsloth_grpo.py (unsloth not installed locally)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from server.prompt_utils import SYSTEM_PROMPT, format_observation

# Test 1: Empty calls
obs_empty = {
    'step_number': 0,
    'max_steps': 80,
    'unit_statuses': [
        {'unit_id': 0, 'last_known_status': 'idle', 'last_known_location': 0},
        {'unit_id': 1, 'last_known_status': 'idle', 'last_known_location': 5},
    ],
    'active_calls': [],
    'traffic_alerts': [],
    'hospital_statuses': [{'hospital_id': 0, 'reported_status': 'accepting'}],
    'mutual_aid_remaining': 2,
}

print("=== SYSTEM PROMPT ===")
print(SYSTEM_PROMPT)
print("\n" + "="*60)
print("\n=== OBSERVATION (no calls) ===")
obs_text = format_observation(obs_empty)
print(obs_text)
print("\n=== CHECK: '(none)' present?", "(none)" in obs_text)

# Test 2: With calls
obs_with_calls = {
    'step_number': 5,
    'max_steps': 80,
    'unit_statuses': [
        {'unit_id': 0, 'last_known_status': 'idle', 'last_known_location': 0},
        {'unit_id': 1, 'last_known_status': 'en_route', 'last_known_location': 3, 'current_call': 1},
    ],
    'active_calls': [
        {'call_id': 2, 'reported_type': 'cardiac', 'location': 7, 'caller_tone': 'panic', 'time_elapsed': 3, 'assigned_unit': None},
        {'call_id': 3, 'reported_type': 'trauma', 'location': 12, 'caller_tone': 'calm', 'time_elapsed': 1, 'assigned_unit': None},
    ],
    'traffic_alerts': ['Bridge collapse between Node 5 and Node 6'],
    'hospital_statuses': [{'hospital_id': 0, 'reported_status': 'accepting'}],
    'mutual_aid_remaining': 2,
}

print("\n=== OBSERVATION (with calls) ===")
obs_text2 = format_observation(obs_with_calls)
print(obs_text2)
print("\n=== CHECK: '(none)' NOT present?", "(none)" not in obs_text2)
print("=== CHECK: Call 2 present?", "Call 2" in obs_text2)
