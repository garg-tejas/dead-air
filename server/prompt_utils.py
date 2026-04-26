"""Shared prompt formatting utilities for DispatchR.

Used by training scripts, inference, and the GRPO wrapper to ensure
consistent prompt formatting everywhere.
"""

from typing import Dict

from .constants import DEADLINES

SYSTEM_PROMPT = (
    "You are an emergency dispatch commander managing 6 ambulance units across a 20-node city.\n\n"
    "OBJECTIVE: Minimize fatalities and response times. You are scored on:\n"
    "- Response time vs optimal (50%): arrive before medical deadlines\n"
    "- Fatality prevention (30%): missing a cardiac deadline causes a fatality\n"
    "- Zone coverage (20%): keep units spread so every zone has nearby coverage\n\n"
    "MEDICAL DEADLINES:\n"
    "- cardiac: 8 min (life-threatening — highest priority)\n"
    "- trauma: 12 min\n"
    "- fire: 15 min\n"
    "- false_alarm: no deadline (consider verifying before dispatching)\n\n"
    "CALLER TONE hints at true severity: screaming > agitated > calm.\n"
    "Ghost calls (no real emergency) and false alarms exist — use verify wisely.\n\n"
    "AVAILABLE ACTIONS (output exactly one JSON as the final line):\n"
    '{"action_type":"dispatch","unit_id":0,"call_id":1}  — send idle unit to call\n'
    '{"action_type":"reroute","unit_id":0,"call_id":2}  — redirect en-route unit to higher priority call\n'
    '{"action_type":"stage","unit_id":0,"location_node":5}  — pre-position idle unit for coverage\n'
    '{"action_type":"verify","call_id":1}  — investigate suspicious call before committing a unit\n'
    '{"action_type":"divert","unit_id":0,"hospital_id":1}  — send unit to specific hospital\n'
    '{"action_type":"request_mutual_aid"}  — call external backup (limited uses)\n'
    '{"action_type":"hold"}  — wait (use when no action improves the situation)\n\n'
    "OUTPUT FORMAT: Reason briefly, then end with the JSON on its own line. No text after the JSON."
)


def format_observation(obs: Dict) -> str:
    """Format env observation into prompt text."""
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
            deadline = DEADLINES.get(c['reported_type'], '?')
            deadline_str = "none" if deadline == float("inf") else f"{deadline}min"
            lines.append(
                f"- Call {c['call_id']}: {c['reported_type']} at Node {c['location']} "
                f"({c['caller_tone']}) elapsed={c['time_elapsed']}min deadline={deadline_str}{assigned}"
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


def build_chat_prompt(tokenizer, system: str, user: str) -> str:
    """Build a chat-formatted prompt using the tokenizer's chat template.

    Enables thinking mode for Qwen 3/3.5 models.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if tokenizer.chat_template is not None:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                chat_template_kwargs={"enable_thinking": True},
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    return f"{system}\n\n{user}\n\nAssistant:"
