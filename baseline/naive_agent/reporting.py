from datetime import datetime
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from config import LOG_DIR


def _truncate(text: str, limit: int = 500) -> str:
    text = str(text).strip()
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [truncated, {len(text)} chars total]"


def build_run_report(messages: list) -> str:
    lines = ["=== Naive Agent Run Report ===", ""]

    step = 0
    total_input = 0
    total_output = 0
    total_tokens = 0
    llm_calls = 0

    for message in messages:
        if isinstance(message, HumanMessage):
            step += 1
            lines.append(f"[Step {step}] USER")
            lines.append(f"  {_truncate(message.content)}")
            lines.append("")

        elif isinstance(message, AIMessage):
            step += 1
            llm_calls += 1
            usage = getattr(message, "usage_metadata", None) or {}
            in_tok = usage.get("input_tokens", 0)
            out_tok = usage.get("output_tokens", 0)
            tot_tok = usage.get("total_tokens", in_tok + out_tok)
            total_input += in_tok
            total_output += out_tok
            total_tokens += tot_tok

            lines.append(f"[Step {step}] AGENT (tokens: in={in_tok}, out={out_tok}, total={tot_tok})")
            if message.content:
                lines.append(f"  thought: {_truncate(message.content)}")
            for call in message.tool_calls or []:
                lines.append(f"  -> tool call: {call['name']}({call.get('args', {})})")
            if not message.content and not message.tool_calls:
                lines.append("  (empty response)")
            lines.append("")

        elif isinstance(message, ToolMessage):
            step += 1
            lines.append(f"[Step {step}] TOOL RESULT ({message.name})")
            lines.append(f"  {_truncate(message.content)}")
            lines.append("")

    lines.append("=== Summary ===")
    lines.append(f"Total messages: {len(messages)}")
    lines.append(f"LLM calls: {llm_calls}")
    lines.append(f"Input tokens:  {total_input}")
    lines.append(f"Output tokens: {total_output}")
    lines.append(f"Total tokens:  {total_tokens}")

    return "\n".join(lines)


def write_run_report(messages: list) -> Path:
    """Write the run report to a timestamped file in LOG_DIR and return its path."""
    report = build_run_report(messages)

    log_dir = Path(LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"run_{timestamp}.log"
    log_path.write_text(report, encoding="utf-8")

    return log_path
