"""SQLite schema and logging/query helpers for GPU metrics, validation results,
workload results, and analyst reports."""

import sqlite3
import json
import time
from pathlib import Path
from hardware.gpu_monitor import GPUMetrics

DB_PATH = Path(__file__).parent.parent / "gpu_validation.db" # db file created at root


def _connect() -> sqlite3.Connection:
    """Open a SQLite connection to DB_PATH with row access by column name."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # allows column access by name instead of index
    return conn


def init_db():
    """Create all tables if they don't already exist."""
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS gpu_metrics (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id          TEXT NOT NULL,
                timestamp       REAL NOT NULL,
                gpu_utilization_pct  REAL,
                memory_used_mb       REAL,
                memory_total_mb      REAL,
                temperature_c        REAL,
                power_draw_w         REAL,
                fan_speed_pct        REAL,
                clock_graphics_mhz   REAL,
                clock_memory_mhz     REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS validation_results (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id          TEXT NOT NULL,
                timestamp       REAL NOT NULL,
                overall_passed  INTEGER NOT NULL,
                results_json    TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analyst_reports (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id      TEXT NOT NULL,
                timestamp   REAL NOT NULL,
                report_text TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS workload_results (
                id                    INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id                TEXT NOT NULL,
                timestamp             REAL NOT NULL,
                workload_name         TEXT NOT NULL,
                duration_seconds      INTEGER NOT NULL,
                operations_per_second REAL NOT NULL,
                peak_memory_used_mb   REAL NOT NULL
            )
        """)

def log_metrics(run_id: str, metrics: GPUMetrics):
    """Insert a GPU metrics snapshot for the given run."""
    with _connect() as conn:
        conn.execute("""
            INSERT INTO gpu_metrics (
                run_id, timestamp,
                gpu_utilization_pct, memory_used_mb, memory_total_mb,
                temperature_c, power_draw_w, fan_speed_pct,
                clock_graphics_mhz, clock_memory_mhz
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            metrics.timestamp,
            metrics.gpu_utilization_pct,
            metrics.memory_used_mb,
            metrics.memory_total_mb,
            metrics.temperature_c,
            metrics.power_draw_w,
            metrics.fan_speed_pct,
            metrics.clock_graphics_mhz,
            metrics.clock_memory_mhz,
        ))

def log_validation_result(run_id: str, overall_passed: bool, results: dict):
    """Insert a threshold validation result (pass/fail plus per-check detail) for the given run."""
    with _connect() as conn:
        conn.execute("""
            INSERT INTO validation_results (run_id, timestamp, overall_passed, results_json)
            VALUES (?, ?, ?, ?)
        """, (run_id, time.time(), int(overall_passed), json.dumps(results)))

def log_workload_result(run_id: str, result: dict):
    """Insert a compute workload result (throughput, peak memory) for the given run."""
    with _connect() as conn:
        conn.execute("""
            INSERT INTO workload_results (
                run_id, timestamp, workload_name,
                duration_seconds, operations_per_second, peak_memory_used_mb
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            time.time(),
            result["workload_name"],
            result["duration_seconds"],
            result["operations_per_second"],
            result["peak_memory_used_mb"],
        ))


def log_analyst_report(run_id: str, report: str):
    """Insert a Claude-generated diagnostic report for the given run."""
    with _connect() as conn:
        conn.execute("""
            INSERT INTO analyst_reports (run_id, timestamp, report_text)
            VALUES (?, ?, ?)
        """, (run_id, time.time(), report))


def get_recent_metrics(limit: int = 20) -> list[dict]:
    """Return the most recent metric snapshots across all runs, newest first."""
    with _connect() as conn:
        rows = conn.execute("""
            SELECT * FROM gpu_metrics
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,)).fetchall()
    return [dict(row) for row in rows]


def get_run_history() -> list[dict]:
    """Return every validation run's id, timestamp, and overall pass/fail, newest first."""
    with _connect() as conn:
        rows = conn.execute("""
            SELECT run_id, timestamp, overall_passed
            FROM validation_results
            ORDER BY timestamp DESC
        """).fetchall()
    return [dict(row) for row in rows]


def get_metrics_for_run(run_id: str) -> list[dict]:
    """Return all metric snapshots logged for a specific run, oldest first."""
    with _connect() as conn:
        rows = conn.execute("""
            SELECT * FROM gpu_metrics
            WHERE run_id = ?
            ORDER BY timestamp ASC
        """, (run_id,)).fetchall()
    return [dict(row) for row in rows]


def get_analyst_report(run_id: str) -> str | None:
    """Return the most recent analyst report text for a run, or None if none exists."""
    with _connect() as conn:
        row = conn.execute("""
            SELECT report_text FROM analyst_reports
            WHERE run_id = ?
            ORDER BY timestamp DESC
            LIMIT 1
        """, (run_id,)).fetchone()
    return row["report_text"] if row else None


def get_historical_averages() -> dict:
    """Return the all-time average of each metric across logged runs, used for regression detection."""
    with _connect() as conn:
        row = conn.execute("""
            SELECT
                AVG(gpu_utilization_pct) AS avg_gpu_utilization_pct,
                AVG(memory_used_mb)      AS avg_memory_used_mb,
                AVG(temperature_c)       AS avg_temperature_c,
                AVG(power_draw_w)        AS avg_power_draw_w,
                AVG(fan_speed_pct)       AS avg_fan_speed_pct,
                AVG(clock_graphics_mhz)  AS avg_clock_graphics_mhz,
                AVG(clock_memory_mhz)    AS avg_clock_memory_mhz
            FROM gpu_metrics
        """).fetchone()
    return dict(row) if row else {}
