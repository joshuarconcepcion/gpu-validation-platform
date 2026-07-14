"""FastAPI HTTP layer. This is the composition root: it is the only module
allowed to import from hardware, validation, storage, and agents together,
since its job is to wire those independent modules into request handlers."""

import time
import uuid
from fastapi import FastAPI, HTTPException

from hardware.gpu_monitor import GPUInstrumentation
from validation.benchmarks import (
    light_workload,
    medium_workload,
    stress_workload,
    validate_metrics,
    detect_regression,
    generate_validation_summary,
)
from storage.db import (
    init_db,
    log_metrics,
    log_validation_result,
    log_workload_result,
    log_analyst_report,
    get_run_history,
    get_metrics_for_run,
    get_analyst_report,
    get_historical_averages,
)
from agents.analyst import analyze_validation_run

app = FastAPI(title="GPU Validation Platform")

init_db()

WORKLOADS = {
    "light": light_workload,
    "medium": medium_workload,
    "stress": stress_workload,
}


def _run_validation(run_id: str, workload_result: dict | None = None) -> dict:
    """Collect a metrics snapshot, validate it, check for regression, log everything,
    and generate a Claude diagnostic report for the run."""
    with GPUInstrumentation() as gpu:
        metrics = gpu.collect()

    validation = validate_metrics(metrics)
    historical = get_historical_averages()
    regression = detect_regression(metrics, historical)
    summary = generate_validation_summary(run_id, metrics, validation, regression)

    log_metrics(run_id, metrics)
    log_validation_result(run_id, validation["overall_passed"], validation["checks"])

    if workload_result: # /validate does not run workload, so if statements prevents crashing
        log_workload_result(run_id, workload_result)
        summary["workload"] = workload_result

    report = analyze_validation_run(summary)
    log_analyst_report(run_id, report)

    return {
        "run_id": run_id,
        "metrics": summary["metrics"],
        "validation": summary["validation"],
        "regression": summary["regression"],
        "workload": workload_result,
        "analyst_report": report,
    }


@app.get("/health")
def health():
    """Confirm the API server is running."""
    return {"status": "ok", "timestamp": time.time()}


@app.post("/validate")
def validate():
    """Collect a metrics snapshot and run threshold + regression checks and Claude analysis, without a workload."""
    run_id = str(uuid.uuid4()) # generates random unique ID and converts to string for run_id
    try:
        return _run_validation(run_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/run/{workload_type}")
def run_workload(workload_type: str):
    """Run the named compute workload (light/medium/stress), then validate and analyze the resulting metrics."""
    if workload_type not in WORKLOADS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid workload type '{workload_type}'. Choose from: {list(WORKLOADS)}",
        )

    run_id = str(uuid.uuid4())
    try:
        workload_result = WORKLOADS[workload_type]()
        return _run_validation(run_id, workload_result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/results")
def get_results():
    """Return the 20 most recent validation runs."""
    return {"results": get_run_history()}


@app.get("/results/{run_id}")
def get_result_by_run_id(run_id: str):
    """Return logged metrics and the Claude analyst report for a specific run_id."""
    metrics = get_metrics_for_run(run_id)
    if not metrics:
        raise HTTPException(status_code=404, detail=f"No data found for run_id: {run_id}")

    return {
        "run_id": run_id,
        "metrics": metrics,
        "analyst_report": get_analyst_report(run_id),
    }


@app.get("/regression")
def regression_check():
    """Collect a fresh metrics snapshot and compare it against historical averages for regressions."""
    with GPUInstrumentation() as gpu:
        metrics = gpu.collect()

    historical = get_historical_averages()
    regression = detect_regression(metrics, historical)

    return {
        "timestamp": time.time(),
        "current_metrics": {
            "gpu_utilization_pct": metrics.gpu_utilization_pct,
            "temperature_c": metrics.temperature_c,
            "power_draw_w": metrics.power_draw_w,
            "clock_graphics_mhz": metrics.clock_graphics_mhz,
            "memory_used_mb": metrics.memory_used_mb,
        },
        "historical_averages": historical,
        "regression": regression,
    }
