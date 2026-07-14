"""Compute workloads and validation logic: run GPU stress tests, then check
results against static thresholds and historical regression baselines."""

import time

from hardware.gpu_monitor import GPUMetrics

# RTX 3090 Ti baseline thresholds
THRESHOLDS = {  # constant for GPU operating bounds
    "max_temp_c": 83.0,                   # 7 degree buffer before 90C throttle point
    "max_memory_utilization_pct": 95.0,   # headroom before out-of-memory errors
    "min_utilization_pct": 85.0,          # minimum expected during active workload
    "max_power_draw_w": 445.0,            # ceiling below 450W TDP
    "min_graphics_clock_mhz": 1500.0,     # floor (if below, possible throttling)
}

REGRESSION_THRESHOLD = 0.10  # flag if a metric degrades more than 10% from historical average


def _run_workload(workload_name: str, matrix_size: int, duration_seconds: int) -> dict:
    """Run repeated matrix multiplications on the GPU for a fixed duration and report throughput."""
    import cupy as cp  # imported lazily: cupy requires CUDA and is unavailable off the GPU host,
                        # but validate_metrics/detect_regression below must stay importable without it

    start = time.time()
    operations = 0

    # matrix multiplication logic:
    a = cp.random.rand(matrix_size, matrix_size, dtype=cp.float32)  # created outside loop to save time on every operation iteration
    b = cp.random.rand(matrix_size, matrix_size, dtype=cp.float32)

    while time.time() - start < duration_seconds:
        cp.dot(a, b)
        cp.cuda.Stream.null.synchronize()  # wait for GPU to finish before next iteration, otherwise "queue" speed tested rather than compute speed
        operations += 1

    peak_memory_used_mb = cp.get_default_memory_pool().used_bytes() / 1024 / 1024  # returns used bytes in memory pool and converts to mb

    cp.get_default_memory_pool().free_all_blocks()  # release GPU memory after workload

    return {
        "workload_name": workload_name,
        "duration_seconds": duration_seconds,
        "operations_per_second": round(operations / duration_seconds, 2),
        "peak_memory_used_mb": round(peak_memory_used_mb, 2),
    }


def light_workload() -> dict:
    """Run a 10s workload against a 1024x1024 matrix."""
    return _run_workload("light", matrix_size=1024, duration_seconds=10)


def medium_workload() -> dict:
    """Run a 30s workload against a 4096x4096 matrix."""
    return _run_workload("medium", matrix_size=4096, duration_seconds=30)


def stress_workload() -> dict:
    """Run a 60s workload against an 8192x8192 matrix."""
    return _run_workload("stress", matrix_size=8192, duration_seconds=60)


def validate_metrics(metrics: GPUMetrics) -> dict:
    """Check a metrics snapshot against static RTX 3090 Ti thresholds and return pass/fail per check."""
    memory_utilization_pct = (metrics.memory_used_mb / metrics.memory_total_mb) * 100

    # rules are for LLM analyst to create human-readable report
    # <= must stay below, >= must stay above
    checks = {
        "temperature_c": {
            "passed": metrics.temperature_c <= THRESHOLDS["max_temp_c"],
            "value": metrics.temperature_c,
            "threshold": THRESHOLDS["max_temp_c"],
            "rule": "<=",
        },
        "memory_utilization_pct": {
            "passed": memory_utilization_pct <= THRESHOLDS["max_memory_utilization_pct"],
            "value": round(memory_utilization_pct, 2),
            "threshold": THRESHOLDS["max_memory_utilization_pct"],
            "rule": "<=",
        },
        "gpu_utilization_pct": {
            "passed": metrics.gpu_utilization_pct >= THRESHOLDS["min_utilization_pct"],
            "value": metrics.gpu_utilization_pct,
            "threshold": THRESHOLDS["min_utilization_pct"],
            "rule": ">=",
        },
        "power_draw_w": {
            "passed": metrics.power_draw_w <= THRESHOLDS["max_power_draw_w"],
            "value": metrics.power_draw_w,
            "threshold": THRESHOLDS["max_power_draw_w"],
            "rule": "<=",
        },
        "clock_graphics_mhz": {
            "passed": metrics.clock_graphics_mhz >= THRESHOLDS["min_graphics_clock_mhz"],
            "value": metrics.clock_graphics_mhz,
            "threshold": THRESHOLDS["min_graphics_clock_mhz"],
            "rule": ">=",
        },
    }

    overall_passed = all(check["passed"] for check in checks.values())  # all() returns true if every value in iterable returns true

    return {
        "overall_passed": overall_passed,
        "checks": checks,
    }


def detect_regression(current_metrics: GPUMetrics, historical_averages: dict) -> dict:
    """Flag any metric that has degraded more than REGRESSION_THRESHOLD from its historical average."""
    regressions = []

    comparisons = [
        ("gpu_utilization_pct", current_metrics.gpu_utilization_pct, "min"),
        ("temperature_c", current_metrics.temperature_c, "max"),
        ("power_draw_w", current_metrics.power_draw_w, "max"),
        ("clock_graphics_mhz", current_metrics.clock_graphics_mhz, "min"),
        ("memory_used_mb", current_metrics.memory_used_mb, "max"),
    ]

    for metric, current_value, direction in comparisons:
        avg_key = f"avg_{metric}"  # key name for lookup in historical averages dict
        if avg_key not in historical_averages or historical_averages[avg_key] is None:
            continue

        historical = historical_averages[avg_key]
        if historical == 0:
            continue

        delta_pct = (current_value - historical) / historical

        # for "min" metrics (utilization, clocks), a negative delta means degradation
        # for "max" metrics (temp, power), a positive delta means degradation
        if direction == "min":
            degraded = delta_pct < -REGRESSION_THRESHOLD
        else:
            degraded = delta_pct > REGRESSION_THRESHOLD

        if degraded:
            regressions.append({
                "metric": metric,
                "current_value": current_value,
                "historical_average": round(historical, 2),
                "delta_pct": round(delta_pct * 100, 2),
            })

    return {
        "regression_detected": len(regressions) > 0,  # returns true if regression list has 1+ values
        "regressions": regressions,
    }


def generate_validation_summary(
    run_id: str,
    metrics: GPUMetrics,
    validation_results: dict,
    regression_results: dict,
) -> dict:
    """Assemble a run's metrics, threshold results, and regression results into one summary dict."""
    return {
        "run_id": run_id,
        "metrics": {
            "timestamp": metrics.timestamp,
            "gpu_utilization_pct": metrics.gpu_utilization_pct,
            "memory_used_mb": metrics.memory_used_mb,
            "memory_total_mb": metrics.memory_total_mb,
            "temperature_c": metrics.temperature_c,
            "power_draw_w": metrics.power_draw_w,
            "fan_speed_pct": metrics.fan_speed_pct,
            "clock_graphics_mhz": metrics.clock_graphics_mhz,
            "clock_memory_mhz": metrics.clock_memory_mhz,
        },
        "validation": validation_results,
        "regression": regression_results,
    }
