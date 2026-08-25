"""Token and $ cost tracing for LLM-calling agents.

Works against any iterable of LangChain messages (e.g. a LangGraph run's
`result["messages"]`) — it does not depend on any particular agent
implementation. Costs are computed from `config.yaml`'s `cost.pricing`
table; a model with no pricing entry contributes $0 rather than raising,
since that table may not be filled in yet.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

from langchain_core.messages import BaseMessage

from agentprep.config import PriceCfg


@dataclass
class ModelUsage:
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0

    def to_dict(self) -> dict:
        return {
            "llm_calls": self.llm_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
        }


@dataclass
class TokenReport:
    by_model: dict[str, ModelUsage] = field(default_factory=dict)

    @property
    def total_llm_calls(self) -> int:
        return sum(usage.llm_calls for usage in self.by_model.values())

    @property
    def total_input_tokens(self) -> int:
        return sum(usage.input_tokens for usage in self.by_model.values())

    @property
    def total_output_tokens(self) -> int:
        return sum(usage.output_tokens for usage in self.by_model.values())

    @property
    def total_tokens(self) -> int:
        return sum(usage.total_tokens for usage in self.by_model.values())

    @property
    def total_cost_usd(self) -> float:
        return sum(usage.cost_usd for usage in self.by_model.values())

    def to_dict(self) -> dict:
        return {
            "by_model": {name: usage.to_dict() for name, usage in self.by_model.items()},
            "totals": {
                "llm_calls": self.total_llm_calls,
                "input_tokens": self.total_input_tokens,
                "output_tokens": self.total_output_tokens,
                "total_tokens": self.total_tokens,
                "cost_usd": self.total_cost_usd,
            },
        }


def extract_token_usage(
    messages: Iterable[BaseMessage],
    pricing: dict[str, PriceCfg] | None = None,
) -> TokenReport:
    """Aggregate per-model token usage (and cost, if priced) from LangChain messages.

    Messages without `usage_metadata` (e.g. HumanMessage, ToolMessage, or an
    AIMessage the provider didn't attach usage to) are skipped.
    """
    report = TokenReport()

    for message in messages:
        usage = getattr(message, "usage_metadata", None)
        if not usage:
            continue

        response_metadata = getattr(message, "response_metadata", None) or {}
        model_name = response_metadata.get("model_name") or "unknown"

        bucket = report.by_model.setdefault(model_name, ModelUsage())
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        total_tokens = usage.get("total_tokens", input_tokens + output_tokens)

        bucket.llm_calls += 1
        bucket.input_tokens += input_tokens
        bucket.output_tokens += output_tokens
        bucket.total_tokens += total_tokens

        price = (pricing or {}).get(model_name)
        if price is not None:
            bucket.cost_usd += (
                input_tokens / 1_000_000 * price.input_per_mtok
                + output_tokens / 1_000_000 * price.output_per_mtok
            )

    return report


def write_token_report(
    report: TokenReport,
    logs_dir: str | Path,
    run_label: str | None = None,
) -> tuple[Path, Path]:
    """Write a per-run JSON detail file and append one CSV row per model to a
    running ledger, both under `logs_dir`.

    Returns (json_path, csv_path). The CSV ledger accumulates rows across
    every run so multiple agents/runs can be compared later (e.g. with
    pandas).
    """
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_label = run_label or timestamp

    json_path = logs_dir / f"token_report_{timestamp}.json"
    json_path.write_text(
        json.dumps({"run_label": run_label, "timestamp": timestamp, **report.to_dict()}, indent=2),
        encoding="utf-8",
    )

    csv_path = logs_dir / "token_usage.csv"
    is_new = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(
                [
                    "timestamp",
                    "run_label",
                    "model",
                    "llm_calls",
                    "input_tokens",
                    "output_tokens",
                    "total_tokens",
                    "cost_usd",
                ]
            )
        for model_name, usage in report.by_model.items():
            writer.writerow(
                [
                    timestamp,
                    run_label,
                    model_name,
                    usage.llm_calls,
                    usage.input_tokens,
                    usage.output_tokens,
                    usage.total_tokens,
                    f"{usage.cost_usd:.6f}",
                ]
            )

    return json_path, csv_path
