"""Quick test to verify the improved system prompt and observation formatting.
Does NOT import train_unsloth_grpo.py (unsloth not installed locally)."""

SYSTEM_PROMPT = (
    "You are an emergency dispatch AI managing 6 ambulance units in a 20-node city. "
    "Every step, output exactly one JSON object on the VERY LAST LINE.\n\n"
    "RULES:\n"
    "- If Active Calls is empty or says '(none)', output: {\"action_type\":\"hold\"}\n"
    "- If there are active calls, dispatch the closest idle unit to the most urgent call.\n"
    "- Keep reasoning to 1-2 sentences. Do not overthink.\n"
    "- The JSON must be the very last thing you output. No markdown, no extra text after it.\n\n"
    "ACTIONS:\n"
    '{"action_type":"dispatch","unit_id":0,"call_id":1}\n'
    '{"action_type":"hold"}\n'
    '{"action_type":"verify","call_id":1}\n\n'
    "Example (no calls): All units idle, no active calls. {\"action_type\":\"hold\"}\n"
    "Example (with calls): Call 2 is cardiac (most urgent). Unit 1 is idle and closest. {\"action_type\":\"dispatch\",\"unit_id\":1,\"call_id\":2}"
)


def format_observation(obs):
    lines = [
        "# Emergency Dispatch Commander",
        f"Step {obs['step_number']}/{obs['max_steps']}",
        "",
        "## Units",
    ]
    for u in obs.get("unit_statuses", []):
        call_info = f" -> Call {u['current_call']}" if u.get("current_call") else ""
        lines.append(
            f"- Unit {u['unit_id']}: {u['last_known_status']} at Node {u['last_known_location']}{call_info}"
        )
    lines.append("")
    lines.append("## Active Calls")
    active_calls = obs.get("active_calls", [])
    if active_calls:
        for c in active_calls:
            assigned = f" (Unit {c['assigned_unit']})" if c.get("assigned_unit") else ""
            lines.append(
                f"- Call {c['call_id']}: {c['reported_type']} at Node {c['location']} ({c['caller_tone']}) elapsed={c['time_elapsed']}min{assigned}"
            )
    else:
        lines.append("(none)")
    lines.append("")
    lines.append("## Traffic & Hospitals")
    for alert in obs.get("traffic_alerts", []):
        lines.append(f"- {alert}")
    for h in obs.get("hospital_statuses", []):
        lines.append(f"- Hospital {h['hospital_id']}: {h['reported_status']}")
    lines.append("")
    lines.append(f"Mutual aid remaining: {obs['mutual_aid_remaining']}")
    lines.append("")
    lines.append("Choose your next action.")
    return "\n".join(lines)


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
