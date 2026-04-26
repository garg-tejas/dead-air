"""Shared prompt formatting utilities for DispatchR.

Used by training scripts, inference, and the GRPO wrapper to ensure
consistent prompt formatting everywhere.
"""

from typing import Dict

from .constants import DEADLINES

SYSTEM_PROMPT = (
    "You are an emergency dispatch commander managing 6 emergency units across a 20-node city.\n\n"
    "OBJECTIVE: Save lives. Your reward depends on three things:\n"
    "- Response time (50%): arrive before medical deadlines — faster is better\n"
    "- Fatalities prevented (30%): each missed cardiac deadline costs -0.5 reward\n"
    "- Zone coverage (20%): keep units spread so every zone has nearby coverage\n\n"
    "REWARD STRUCTURE (this determines your learning signal):\n"
    "- DISPATCHING a unit to a pending call: +0.02 immediate reward\n"
    "- Successfully resolving a call before deadline: large positive reward\n"
    "- HOLDING when pending calls exist and idle units are available: -0.01 per step\n"
    "- INVALID action (e.g., dispatching busy unit): -0.02 penalty\n"
    "- Each fatality (missed deadline): -0.5 penalty\n"
    "- All-hold episode with fatalities: approximately -1.0 total reward\n"
    "- Good dispatch episode with zero fatalities: approximately +0.5 to +1.0 reward\n\n"
    "MEDICAL DEADLINES (minutes from call arrival):\n"
    "- cardiac: 8 min (life-threatening — highest priority)\n"
    "- trauma: 12 min\n"
    "- fire: 15 min\n"
    "- false_alarm: no deadline (verify before dispatching)\n\n"
    "CALLER TONE hints at true severity: screaming > agitated > calm.\n"
    "Ghost calls (no real emergency) and false alarms exist — use verify wisely.\n\n"
    "DISPATCH RULES (violate these and you lose reward):\n"
    "1. If pending calls exist AND idle units are available → YOU MUST DISPATCH. Holding is wrong.\n"
    "2. Only use hold() when there are NO pending calls or NO idle units.\n"
    "3. Cardiac calls have highest priority. Dispatch to cardiac before trauma before fire.\n"
    "4. Use stage() only when there are NO pending calls to pre-position for future coverage.\n"
    "5. Every second you hold with a pending cardiac call, someone might die.\n\n"
    "AVAILABLE ACTIONS (output exactly one JSON as the final line):\n"
    '{"action_type":"dispatch","unit_id":0,"call_id":1}\n'
    '{"action_type":"reroute","unit_id":0,"call_id":2}\n'
    '{"action_type":"stage","unit_id":0,"location_node":5}\n'
    '{"action_type":"verify","call_id":1}\n'
    '{"action_type":"divert","unit_id":0,"hospital_id":1}\n'
    '{"action_type":"request_mutual_aid"}\n'
    '{"action_type":"hold"}\n\n'
    "OUTPUT FORMAT: Think briefly (1-2 sentences max), then output exactly one JSON action.\n"
    'Your response must end with this JSON on the final line: {"action_type": "'
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
    # Include dispatch log if the agent has used the log() action
    dispatch_log = obs.get("dispatch_log", "")
    if dispatch_log:
        lines.append("## Dispatch Log")
        # Show last 5 entries to keep prompt size bounded
        log_entries = [ln for ln in dispatch_log.strip().split("\n") if ln.strip()]
        for entry in log_entries[-5:]:
            lines.append(f"- {entry}")
        lines.append("")
    lines.append("Choose your next action.")
    return "\n".join(lines)


def build_chat_prompt(tokenizer, system: str, user: str, enable_thinking: bool = False) -> str:
    """Build a chat-formatted prompt using the tokenizer's chat template."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    return f"{system}\n\n{user}\n\nAssistant:"
