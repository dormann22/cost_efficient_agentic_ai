from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agentprep.config import PriceCfg
from agentprep.token_tracking import extract_token_usage, write_token_report


def _ai_message(input_tokens: int, output_tokens: int, model_name: str) -> AIMessage:
    return AIMessage(
        content="ok",
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
        response_metadata={"model_name": model_name},
    )


def test_sums_tokens_for_single_model():
    messages = [
        _ai_message(100, 20, "gpt-4o-mini"),
        _ai_message(50, 10, "gpt-4o-mini"),
    ]

    report = extract_token_usage(messages)

    assert report.total_llm_calls == 2
    assert report.total_input_tokens == 150
    assert report.total_output_tokens == 30
    assert report.total_tokens == 180
    assert report.total_cost_usd == 0.0


def test_buckets_by_model():
    messages = [
        _ai_message(1000, 100, "gpt-4o"),
        _ai_message(500, 50, "gpt-4o-mini"),
    ]

    report = extract_token_usage(messages)

    assert set(report.by_model) == {"gpt-4o", "gpt-4o-mini"}
    assert report.by_model["gpt-4o"].input_tokens == 1000
    assert report.by_model["gpt-4o-mini"].input_tokens == 500


def test_computes_cost_when_pricing_available():
    pricing = {"gpt-4o": PriceCfg(input_per_mtok=2.5, output_per_mtok=10.0)}
    messages = [_ai_message(1_000_000, 1_000_000, "gpt-4o")]

    report = extract_token_usage(messages, pricing=pricing)

    assert report.by_model["gpt-4o"].cost_usd == 12.5
    assert report.total_cost_usd == 12.5


def test_missing_pricing_defaults_to_zero_cost():
    pricing = {"gpt-4o": PriceCfg(input_per_mtok=2.5, output_per_mtok=10.0)}
    messages = [_ai_message(1000, 500, "gpt-4o-mini")]  # not in pricing table

    report = extract_token_usage(messages, pricing=pricing)

    assert report.by_model["gpt-4o-mini"].cost_usd == 0.0


def test_ignores_non_ai_messages():
    messages = [
        HumanMessage(content="hi"),
        _ai_message(10, 5, "gpt-4o-mini"),
        ToolMessage(content="result", tool_call_id="abc", name="write_code"),
    ]

    report = extract_token_usage(messages)

    assert report.total_llm_calls == 1
    assert report.total_tokens == 15


def test_write_token_report_creates_json_and_appends_csv(tmp_path):
    messages = [_ai_message(10, 5, "gpt-4o-mini")]
    report = extract_token_usage(messages)

    json_path, csv_path = write_token_report(report, tmp_path, run_label="unit-test")

    assert json_path.exists()
    assert csv_path.exists()
    assert "gpt-4o-mini" in csv_path.read_text(encoding="utf-8")
    assert "unit-test" in csv_path.read_text(encoding="utf-8")
