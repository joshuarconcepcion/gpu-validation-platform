from src.instrumentation import GPUMetrics

# RTX 3090 Ti baseline thresholds
THRESHOLDS = { # constant for GPU operating bounds
    "max_temp_c": 83.0,                   # 7 degree buffer before 90C throttle point
    "max_memory_utilization_pct": 95.0,   # headroom before out-of-memory errors
    "min_utilization_pct": 85.0,          # minimum expected during active workload
    "max_power_draw_w": 350.0,            # ceiling below 450W TDP
    "min_graphics_clock_mhz": 1500.0,     # floor (if below, possible throttling)
}

REGRESSION_THRESHOLD = 0.10  # flag if a metric degrades more than 10% from historical average


def validate_metrics(metrics: GPUMetrics) -> dict:
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

    overall_passed = all(check["passed"] for check in checks.values()) # all() returns true if every value in iterable returns true

    return {
        "overall_passed": overall_passed,
        "checks": checks,
    }


def detect_regression(current_metrics: GPUMetrics, historical_averages: dict) -> dict:
    regressions = []

    comparisons = [ 
        ("gpu_utilization_pct", current_metrics.gpu_utilization_pct, "min"),
        ("temperature_c", current_metrics.temperature_c, "max"),
        ("power_draw_w", current_metrics.power_draw_w, "max"),
        ("clock_graphics_mhz", current_metrics.clock_graphics_mhz, "min"),
        ("memory_used_mb", current_metrics.memory_used_mb, "max"),
    ]

    for metric, current_value, direction in comparisons:
        avg_key = f"avg_{metric}" # key name for lookup in historical averages dict
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
        "regression_detected": len(regressions) > 0, # returns true if regression list has 1+ values
        "regressions": regressions,
    }


def generate_validation_summary(
    run_id: str,
    metrics: GPUMetrics,
    validation_results: dict,
    regression_results: dict,
) -> dict:
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
