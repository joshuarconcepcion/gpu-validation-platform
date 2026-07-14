# GPU Validation Platform

An autonomous validation platform for NVIDIA GPUs (currently targeting the RTX 3090 Ti). It runs
compute workloads, collects hardware telemetry via NVML, checks results against defined thresholds
and historical baselines, persists everything to SQLite, and generates AI diagnostic reports via the
Anthropic Claude API — all behind a FastAPI HTTP layer.

## Why

Manually watching temperature/power/clock graphs during a stress test doesn't scale, and raw metric
dumps aren't useful to an engineer trying to triage a failure quickly. This project's near-term goal
is a clean, well-tested harness that can run unattended and hand back pass/fail verdicts plus a
readable diagnosis instead of a spreadsheet.

The longer-term goal is a multi-agent validation platform: autonomous agents that run full validation
campaigns, retrieve relevant historical failure context via RAG to triage new failures, and produce
structured diagnostic reports with minimal human involvement — in the spirit of what AMD's GenAI team
is building internally. This repo (`agents/`, `rag/`, `evaluation/`, `frontend/`) is Phase 0: a clean,
modular foundation to build that on top of. Those directories are currently placeholders for later
phases; today, `agents/analyst.py` covers single-shot Claude-generated reports.

## Project Structure

```
gpu-validation-platform/
├── hardware/
│   └── gpu_monitor.py      # NVML metric collection (utilization, memory, temp, power, clocks)
├── validation/
│   └── benchmarks.py       # compute workloads, threshold checks, regression detection
├── storage/
│   └── db.py               # SQLite schema and logging/query helpers
├── api/
│   └── main.py             # FastAPI routes (the composition root wiring the modules together)
├── agents/
│   └── analyst.py          # Claude API integration for diagnostic reports
├── rag/                    # placeholder — failure-triage retrieval (future phase)
├── frontend/                # placeholder — dashboard UI (future phase)
├── evaluation/              # placeholder — agent/report quality evals (future phase)
├── tests/
│   ├── test_validation.py  # threshold + regression + SQLite tests (no GPU required)
│   └── test_hardware.py    # NVML integration tests (requires a real GPU)
├── requirements.txt
├── .env.example
├── .gitignore
├── Dockerfile
└── docker-compose.yml
```

**Import rule:** each domain module (`hardware`, `validation`, `storage`, `agents`) depends on at most
one sibling module — `validation` and `storage` both depend only on `hardware` (for the `GPUMetrics`
type), and `agents` depends on none of them. `api/main.py` is the exception: as the composition root,
it's the only place allowed to import from all of them, since its job is wiring independent modules
into HTTP handlers.

## Tech Stack

| Component | Library |
|---|---|
| GPU metrics | `pynvml` (wraps NVML) |
| Compute workloads | `cupy` (CUDA array ops — Linux only) |
| API layer | `fastapi` + `uvicorn` |
| Persistence | `sqlite3` (stdlib) |
| AI diagnostics | `anthropic` (Claude API) |
| Tests | `pytest` |

## Running Locally

Requires Python 3.11+. `cupy` and `pynvml` require an actual NVIDIA GPU with CUDA drivers, so full
functionality (`/validate`, `/run/{workload_type}`, `/regression`) only works on the GPU host. The
non-GPU logic (threshold checks, regression detection, SQLite) can be developed and tested anywhere.

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env             # then fill in ANTHROPIC_API_KEY

uvicorn api.main:app --reload --port 8000
```

Open `http://localhost:8000/docs` for an interactive UI to try every endpoint.

### Running with Docker

```bash
docker compose up --build
```

This builds the API image and runs it on port 8000, requesting GPU access via the NVIDIA Container
Toolkit (see the `deploy.resources.reservations` block in `docker-compose.yml`). Remove that block if
you just want the API container up without hardware access.

### Running Tests

```bash
pytest tests/test_validation.py   # pure logic — runs anywhere
pytest tests/test_hardware.py     # requires a real NVIDIA GPU
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Confirms the server is running |
| POST | `/validate` | Collects a metrics snapshot, checks thresholds + regression, runs Claude analysis |
| POST | `/run/{workload_type}` | Runs a workload (`light`, `medium`, or `stress`), then the full validation pipeline |
| GET | `/results` | Returns the 20 most recent validation runs |
| GET | `/results/{run_id}` | Returns logged metrics and the Claude report for a specific run |
| GET | `/regression` | Compares current metrics against historical averages |

**Workload types:** `light` (10s, 1024×1024), `medium` (30s, 4096×4096), `stress` (60s, 8192×8192)

## Validation Thresholds (RTX 3090 Ti)

| Metric | Threshold |
|---|---|
| Max temperature | 83°C |
| Max memory utilization | 95% |
| Min GPU utilization | 85% |
| Max power draw | 445W |
| Min graphics clock | 1500 MHz |

A metric is flagged as a **regression** when it deviates more than 10% from its historical average
across all past runs.

## Database

Results persist to a local SQLite database (`gpu_validation.db`) across four tables:

- `gpu_metrics` — per-run metric snapshots
- `validation_results` — threshold check results
- `workload_results` — compute workload benchmarks
- `analyst_reports` — Claude-generated diagnostic text

## Example Output

`POST /run/light`:

```json
{
  "run_id": "8f1a2c3d-6b7e-4f21-9a10-4e5f6a7b8c9d",
  "metrics": {
    "timestamp": 1752521400.12,
    "gpu_utilization_pct": 97.0,
    "memory_used_mb": 4102.3,
    "memory_total_mb": 24564.0,
    "temperature_c": 71.0,
    "power_draw_w": 328.4,
    "fan_speed_pct": 58.0,
    "clock_graphics_mhz": 1875.0,
    "clock_memory_mhz": 9751.0
  },
  "validation": {
    "overall_passed": true,
    "checks": {
      "temperature_c": { "passed": true, "value": 71.0, "threshold": 83.0, "rule": "<=" },
      "memory_utilization_pct": { "passed": true, "value": 16.7, "threshold": 95.0, "rule": "<=" },
      "gpu_utilization_pct": { "passed": true, "value": 97.0, "threshold": 85.0, "rule": ">=" },
      "power_draw_w": { "passed": true, "value": 328.4, "threshold": 445.0, "rule": "<=" },
      "clock_graphics_mhz": { "passed": true, "value": 1875.0, "threshold": 1500.0, "rule": ">=" }
    }
  },
  "regression": {
    "regression_detected": false,
    "regressions": []
  },
  "workload": {
    "workload_name": "light",
    "duration_seconds": 10,
    "operations_per_second": 14.2,
    "peak_memory_used_mb": 4102.3
  },
  "analyst_report": "GPU health looks nominal for this light workload run. Utilization (97%) and clock speed (1875 MHz) confirm the GPU was fully engaged with no throttling. Temperature (71°C) sits well under the 83°C ceiling and power draw (328W) is comfortably below the 445W limit. No regressions against historical averages. No follow-up action needed."
}
```
