"""Tests for threshold validation, regression detection, and SQLite persistence.
No real GPU is required — these operate on hand-built GPUMetrics snapshots."""

import pytest

from hardware.gpu_monitor import GPUMetrics
from validation.benchmarks import THRESHOLDS, validate_metrics, detect_regression
from storage import db


def _make_metrics(**overrides) -> GPUMetrics:
    """Build a GPUMetrics snapshot that passes every threshold by default, with fields overridable per test."""
    defaults = dict(
        timestamp=1700000000.0,
        gpu_utilization_pct=90.0,
        memory_used_mb=8192.0,
        memory_total_mb=24564.0,
        temperature_c=70.0,
        power_draw_w=300.0,
        fan_speed_pct=60.0,
        clock_graphics_mhz=1800.0,
        clock_memory_mhz=9500.0,
    )
    defaults.update(overrides)
    return GPUMetrics(**defaults)


class TestThresholdValidation:
    """Pass/fail logic in validate_metrics()."""

    def test_all_metrics_within_thresholds_pass(self):
        result = validate_metrics(_make_metrics())
        assert result["overall_passed"] is True
        assert all(check["passed"] for check in result["checks"].values())

    def test_temperature_above_threshold_fails(self):
        result = validate_metrics(_make_metrics(temperature_c=THRESHOLDS["max_temp_c"] + 1))
        assert result["overall_passed"] is False
        assert result["checks"]["temperature_c"]["passed"] is False

    def test_power_draw_above_threshold_fails(self):
        result = validate_metrics(_make_metrics(power_draw_w=THRESHOLDS["max_power_draw_w"] + 1))
        assert result["checks"]["power_draw_w"]["passed"] is False

    def test_utilization_below_threshold_fails(self):
        result = validate_metrics(_make_metrics(gpu_utilization_pct=THRESHOLDS["min_utilization_pct"] - 1))
        assert result["checks"]["gpu_utilization_pct"]["passed"] is False

    def test_graphics_clock_below_threshold_fails(self):
        result = validate_metrics(_make_metrics(clock_graphics_mhz=THRESHOLDS["min_graphics_clock_mhz"] - 1))
        assert result["checks"]["clock_graphics_mhz"]["passed"] is False

    def test_memory_utilization_above_threshold_fails(self):
        result = validate_metrics(_make_metrics(memory_used_mb=24564.0))  # 100% of total
        assert result["checks"]["memory_utilization_pct"]["passed"] is False

    def test_boundary_value_passes(self):
        # thresholds are inclusive (<=/>=), so sitting exactly on one should still pass
        result = validate_metrics(_make_metrics(temperature_c=THRESHOLDS["max_temp_c"]))
        assert result["checks"]["temperature_c"]["passed"] is True


class TestRegressionDetection:
    """>10% deviation flagging logic in detect_regression()."""

    def test_no_regression_when_within_threshold(self):
        current = _make_metrics(temperature_c=70.0)
        historical = {"avg_temperature_c": 65.0}  # ~7.7% increase, under 10%
        result = detect_regression(current, historical)
        assert result["regression_detected"] is False

    def test_regression_flagged_when_max_metric_exceeds_threshold(self):
        current = _make_metrics(temperature_c=90.0)
        historical = {"avg_temperature_c": 70.0}  # ~28.6% increase, over 10%
        result = detect_regression(current, historical)
        assert result["regression_detected"] is True
        assert "temperature_c" in [r["metric"] for r in result["regressions"]]

    def test_regression_flagged_when_min_metric_drops_below_threshold(self):
        current = _make_metrics(gpu_utilization_pct=70.0)
        historical = {"avg_gpu_utilization_pct": 90.0}  # ~22.2% drop, over 10%
        result = detect_regression(current, historical)
        assert result["regression_detected"] is True
        assert "gpu_utilization_pct" in [r["metric"] for r in result["regressions"]]

    def test_exact_boundary_is_not_flagged(self):
        # exactly 10% degradation should not trigger since the comparison is strict
        current = _make_metrics(power_draw_w=110.0)
        historical = {"avg_power_draw_w": 100.0}
        result = detect_regression(current, historical)
        assert result["regression_detected"] is False

    def test_missing_historical_average_is_skipped(self):
        result = detect_regression(_make_metrics(), {})
        assert result["regression_detected"] is False
        assert result["regressions"] == []

    def test_zero_historical_average_is_skipped(self):
        current = _make_metrics(power_draw_w=300.0)
        historical = {"avg_power_draw_w": 0.0}
        result = detect_regression(current, historical)
        assert result["regression_detected"] is False


class TestDatabaseSchema:
    """SQLite persistence: storing and retrieving a validation run."""

    @pytest.fixture(autouse=True)
    def _isolated_db(self, tmp_path, monkeypatch):
        """Point storage.db at a throwaway SQLite file so tests never touch the real database."""
        monkeypatch.setattr(db, "DB_PATH", tmp_path / "test_gpu_validation.db")
        db.init_db()

    def test_log_and_retrieve_metrics(self):
        metrics = _make_metrics()
        db.log_metrics("run-1", metrics)

        rows = db.get_metrics_for_run("run-1")

        assert len(rows) == 1
        assert rows[0]["run_id"] == "run-1"
        assert rows[0]["temperature_c"] == metrics.temperature_c
        assert rows[0]["gpu_utilization_pct"] == metrics.gpu_utilization_pct

    def test_log_and_retrieve_validation_result(self):
        db.log_validation_result("run-2", True, {"temperature_c": {"passed": True}})

        history = db.get_run_history()

        assert len(history) == 1
        assert history[0]["run_id"] == "run-2"
        assert history[0]["overall_passed"] == 1

    def test_log_and_retrieve_analyst_report(self):
        db.log_analyst_report("run-3", "GPU healthy, no concerns.")

        assert db.get_analyst_report("run-3") == "GPU healthy, no concerns."

    def test_get_analyst_report_returns_none_when_missing(self):
        assert db.get_analyst_report("nonexistent-run") is None

    def test_historical_averages_reflect_logged_metrics(self):
        db.log_metrics("run-4", _make_metrics(temperature_c=60.0))
        db.log_metrics("run-4", _make_metrics(temperature_c=80.0))

        averages = db.get_historical_averages()

        assert averages["avg_temperature_c"] == 70.0
