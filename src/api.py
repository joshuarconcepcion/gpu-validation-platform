import time
import uuid
from fastapi import FastAPI, HTTPException

from src.instrumentation import GPUInstrumentation
from src.workload import light_workload, medium_workload, stress_workload
from src.validator import validate_metrics, detect_regression, generate_validation_summary
from src.database import (
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
from src.analyst import analyze_validation_run

app = FastAPI(title="GPU Validation Harness")

init_db()

WORKLOADS = {
    "light": light_workload,
    "medium": medium_workload,
    "stress": stress_workload,
}


def _run_validation(run_id: str, workload_result: dict | None = None) -> dict:
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

# confirms server is running:
@app.get("/health")
def health():
    return {"status": "ok", "timestamp": time.time()}

# runs w/o workload for quick metrics, validation against thresholds, regression check, and Claude analysis
@app.post("/validate")
def validate():
    run_id = str(uuid.uuid4()) # generates random unique ID and converts to string for run_id
    try:
        return _run_validation(run_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/run/{workload_type}")
def run_workload(workload_type: str):
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

# get's 20 most recent validation runs
@app.get("/results")
def get_results():
    return {"results": get_run_history()}

# returns metrics and Claude analysis based on inputted run_id
@app.get("/results/{run_id}")
def get_result_by_run_id(run_id: str):
    metrics = get_metrics_for_run(run_id)
    if not metrics:
        raise HTTPException(status_code=404, detail=f"No data found for run_id: {run_id}")

    return {
        "run_id": run_id,
        "metrics": metrics,
        "analyst_report": get_analyst_report(run_id),
    }

# regression check:
@app.get("/regression")
def regression_check():
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
