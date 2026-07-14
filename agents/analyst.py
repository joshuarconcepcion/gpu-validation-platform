"""Anthropic Claude integration that turns a validation run summary into a
human-readable diagnostic report for an engineer."""

import anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-sonnet-4-6"


def format_prompt(summary: dict) -> str:
    """Render a validation run summary (as produced by
    validation.benchmarks.generate_validation_summary, optionally with a
    'workload' key attached) into a plain-text prompt for Claude."""
    run_id = summary.get("run_id", "unknown")
    metrics = summary.get("metrics", {})
    validation = summary.get("validation", {})
    regression = summary.get("regression", {})
    workload = summary.get("workload")

    passed = validation.get("overall_passed", False)
    checks = validation.get("checks", {})
    regressions = regression.get("regressions", [])

    lines = [ # list of strings to format into prompt body
        f"GPU Validation Report — Run ID: {run_id}",
        f"Overall status: {'PASSED' if passed else 'FAILED'}",
        "",
        "## Metrics snapshot",
    ]
    for k, v in metrics.items(): # appends metrics and values to lines
        lines.append(f"  {k}: {v}")

    if workload: # appends workload results to lines
        lines += ["", "## Workload results"]
        for k, v in workload.items():
            lines.append(f"  {k}: {v}")

    lines += ["", "## Threshold validation checks"]
    for check, result in checks.items(): # takes values in dict in checks and appends formatted string w/ info to lines
        status = "PASS" if result.get("passed") else "FAIL"
        lines.append(f"  [{status}] {check}: {result.get('value')} (limit: {result.get('rule', '')} {result.get('threshold')})")

    if regressions:
        lines += ["", "## Regression checks"]
        for r in regressions: # each entry is a flagged metric, not a pass/fail dict
            lines.append(
                f"  [REGRESSION] {r['metric']}: {r['current_value']} "
                f"(historical avg: {r['historical_average']}, delta: {r['delta_pct']}%)"
            )

    prompt_body = "\n".join(lines) # concatenates all strings in lines, separated by newlines
    return (
        # instruction for claude:
        f"{prompt_body}\n\n"
        "Analyze this GPU validation run. Summarize what the results indicate about GPU health, "
        "flag any concerns, and suggest follow-up actions if anything failed or regressed. "
        "Be concise and technical as this will be read by an engineer."
    )


def analyze_validation_run(summary: dict) -> str:
    """Send a validation run summary to Claude and return the generated diagnostic report text."""
    client = anthropic.Anthropic()
    prompt = format_prompt(summary)

    try:
        message = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text # returns content block at index 0 and pulls out string with .text
    except anthropic.APIError as e:
        raise RuntimeError(f"Anthropic API error: {e}") from e


# mock main w/ mock data to test api call
if __name__ == "__main__":
    mock_summary = {
        "run_id": "test-run-001",
        "metrics": {
            "gpu_utilization_pct": 92.5,
            "memory_used_mb": 18432.0,
            "memory_total_mb": 24564.0,
            "temperature_c": 85.0,
            "power_draw_w": 355.0,
            "fan_speed_pct": 78.0,
            "clock_graphics_mhz": 1860.0,
            "clock_memory_mhz": 1313.0,
        },
        "workload": {
            "workload_name": "medium",
            "duration_seconds": 30,
            "operations_per_second": 12.4,
            "peak_memory_used_mb": 16384.0,
        },
        "validation": {
            "overall_passed": False,
            "checks": {
                "temperature_c":          {"passed": False, "value": 85.0,   "threshold": 83.0,   "rule": "<="},
                "power_draw_w":           {"passed": False, "value": 355.0,  "threshold": 350.0,  "rule": "<="},
                "gpu_utilization_pct":    {"passed": True,  "value": 92.5,   "threshold": 85.0,   "rule": ">="},
                "memory_utilization_pct": {"passed": True,  "value": 75.0,   "threshold": 95.0,   "rule": "<="},
                "clock_graphics_mhz":     {"passed": True,  "value": 1860.0, "threshold": 1500.0, "rule": ">="},
            },
        },
        "regression": {
            "regression_detected": True,
            "regressions": [
                {"metric": "temperature_c", "current_value": 85.0, "historical_average": 75.9, "delta_pct": 12.0},
            ],
        },
    }

    report = analyze_validation_run(mock_summary)
    print(report)
