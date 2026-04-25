"""Shared prompt formatting utilities for DispatchR.

Used by training scripts, inference, and the GRPO wrapper to ensure
consistent prompt formatting everywhere.
"""

from typing import Dict

SYSTEM_PROMPT = (
    "You are an emergency dispatch AI managing 6 ambulance units in a 20-node city. "
    "Every step, output exactly one JSON object on the VERY LAST LINE.\n\n"
    "RULES:\n"
    '- If Active Calls is empty or says \'(none)\', output: {"action_type":"hold"}\n'
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
