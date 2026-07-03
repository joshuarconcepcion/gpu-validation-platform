import anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-sonnet-4-6"


def format_prompt(summary: dict) -> str:
    # .get() with default values to avoid crashing if fields are not populated yet
    run_id = summary.get("run_id", "unknown")
    passed = summary.get("overall_passed", False)
    validations = summary.get("validation_results", {})
    regressions = summary.get("regression_results", {})
    metrics = summary.get("metrics", {})
    workload = summary.get("workload", {})

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
    for check, result in validations.items(): # takes values in dict in validations and appends formatted string w/ info to lines
        status = "PASS" if result.get("passed") else "FAIL"
        lines.append(f"  [{status}] {check}: {result.get('value')} (limit: {result.get('threshold')})")

    if regressions:
        lines += ["", "## Regression checks"]
        for metric, result in regressions.items(): # same as threshold validation check, but for regression
            status = "PASS" if result.get("passed") else "REGRESSION"
            lines.append(f"  [{status}] {metric}: {result.get('value')} (limit: {result.get('threshold')})")

    prompt_body = "\n".join(lines) # concatenates all strings in lines, separated by newlines
    return (
        # instruction for claude:
        f"{prompt_body}\n\n"
        "Analyze this GPU validation run. Summarize what the results indicate about GPU health, "
        "flag any concerns, and suggest follow-up actions if anything failed or regressed. "
        "Be concise and technical as this will be read by an engineer."
    )


def analyze_validation_run(summary: dict) -> str:
    client = anthropic.Anthropic()
    prompt = format_prompt(summary)

    try:
        message = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
    except anthropic.APIError as e:
        raise RuntimeError(f"Anthropic API error: {e}") from e


# mock main w/ mock data to test api call
if __name__ == "__main__":
    mock_summary = {
        "run_id": "test-run-001",
        "overall_passed": False,
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
        "validation_results": {
            "temperature_c":          {"passed": False, "value": 85.0,   "threshold": 83.0},
            "power_draw_w":           {"passed": False, "value": 355.0,  "threshold": 350.0},
            "gpu_utilization_pct":    {"passed": True,  "value": 92.5,   "threshold": 85.0},
            "memory_utilization_pct": {"passed": True,  "value": 75.0,   "threshold": 95.0},
            "clock_graphics_mhz":     {"passed": True,  "value": 1860.0, "threshold": 1500.0},
        },
        "regression_results": {
            "temperature_c": {"passed": False, "value": 85.0, "threshold": 75.9},
        },
    }

    report = analyze_validation_run(mock_summary)
    print(report)
