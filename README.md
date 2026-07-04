# GPU Validation Harness

A testing and validation framework for the NVIDIA RTX 3090 Ti, built to programmatically stress the GPU, collect real-time hardware metrics, validate performance against defined baselines, detect regressions, and generate AI-powered diagnostic reports via the Anthropic Claude API.

## Overview

This project provides two modes of operation:

**API mode** — start a FastAPI server to trigger and retrieve benchmark runs over HTTP:
```bash
uvicorn src.api:app --reload --port 8000
```

Then open `http://localhost:8000/docs` for an interactive UI to test every endpoint.

**CLI mode** — run a workload and print a report directly to terminal (coming soon):
```bash
python main.py --workload light
python main.py --workload medium
python main.py --workload stress
```

## Tech Stack

| Component | Library |
|---|---|
| GPU metrics | `nvidia-ml-py` (pynvml / NVIDIA NVML) |
| GPU compute workloads | `cupy` |
| API layer | `fastapi` + `uvicorn` |
| Persistence | `sqlite3` (stdlib) |
| AI analysis | `anthropic` (Claude API) |
| Tests | `pytest` |

## Project Structure

```
gpu-validation-harness/
  src/
    instrumentation.py  # pynvml metric collection (utilization, memory, temp, power, clocks)
    workload.py         # GPU compute workloads (light / medium / stress matrix multiplication)
    validator.py        # threshold validation and regression detection
    database.py         # SQLite schema and result logging
    analyst.py          # Anthropic Claude API integration for diagnostic reports
    api.py              # FastAPI REST endpoints
  tests/
    test_instrumentation.py
    test_validator.py
  main.py               # CLI entry point (coming soon)
  requirements.txt
  .env                  # ANTHROPIC_API_KEY (not committed)
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Confirms server is running |
| POST | `/validate` | Collects metrics, validates against thresholds, runs Claude analysis |
| POST | `/run/{workload_type}` | Runs a workload then full validation pipeline |
| GET | `/results` | Returns last 20 validation runs |
| GET | `/results/{run_id}` | Returns metrics and analyst report for a specific run |
| GET | `/regression` | Checks current metrics against historical averages |

**Workload types:** `light` (10s, 1024x1024), `medium` (30s, 4096x4096), `stress` (60s, 8192x8192)

## Metrics Collected

Each run captures the following per reading:

- GPU core utilization (%)
- VRAM used / total (MB)
- Core temperature (°C)
- Power draw (W)
- Fan speed (%)
- Graphics clock (MHz)
- Memory clock (MHz)

## Validation Thresholds (RTX 3090 Ti)

| Metric | Threshold |
|---|---|
| Max temperature | 83°C |
| Max memory utilization | 95% |
| Min GPU utilization | 85% |
| Max power draw | 350W |
| Min graphics clock | 1500 MHz |

Regression is flagged when any metric deviates more than 10% from its historical average across all past runs.

## Database

Results are persisted to a local SQLite database (`gpu_validation.db`) across four tables:

- `gpu_metrics` — per-run metric snapshots
- `validation_results` — threshold check results
- `workload_results` — compute workload benchmarks
- `analyst_reports` — Claude-generated diagnostic text

## Module Status

| Module | Status |
|---|---|
| `instrumentation.py` | Complete |
| `workload.py` | Complete |
| `validator.py` | Complete |
| `database.py` | Complete |
| `analyst.py` | Complete |
| `api.py` | Complete |
| `main.py` | In progress |
| `tests/test_validator.py` | In progress |

## Setup

### Requirements
- Windows or Linux with NVIDIA drivers installed
- CUDA Toolkit 13.x
- Python 3.10+

### Install

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

python -m pip install -r requirements.txt
```

### Environment

Create a `.env` file in the project root:
```
ANTHROPIC_API_KEY=your_key_here
```

### Start the API

```bash
uvicorn src.api:app --reload --port 8000
```

## Running Tests

```bash
pytest tests/
```
