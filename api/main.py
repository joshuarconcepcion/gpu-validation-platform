"""FastAPI HTTP layer. This is the composition root: it is the only module
allowed to import from hardware, validation, storage, agents, and rag
together, since its job is to wire those independent modules into request
handlers."""

import functools
import time
import uuid
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import StreamingResponse

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
from rag.embedder import get_embeddings
from rag.ingestion import documents_from_run, documents_from_all_runs
from rag.store import add_documents, get_store
from rag.retriever import get_retriever
from rag.pipeline import query as run_rag_query, stream_query as run_rag_stream_query

app = FastAPI(title="GPU Validation Platform")

init_db()


@functools.lru_cache(maxsize=1)
def _embeddings():
    """Load the HuggingFace embedding model once and reuse it across requests."""
    return get_embeddings()


def _rag_store():
    """Open the persistent Chroma store, creating it on first use."""
    return get_store(_embeddings())

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


@app.post("/rag/ingest/{run_id}")
def rag_ingest_run(run_id: str):
    """Embed and store the failed-check events for one validation run."""
    documents = documents_from_run(run_id)
    if not documents:
        raise HTTPException(status_code=404, detail=f"No failed checks found for run_id: {run_id}")
    add_documents(_rag_store(), documents)
    return {"run_id": run_id, "documents_ingested": len(documents)}


@app.post("/rag/ingest/all")
def rag_ingest_all():
    """Embed and store failed-check events for every validation run in the database."""
    documents = documents_from_all_runs()
    add_documents(_rag_store(), documents)
    return {"documents_ingested": len(documents)}


@app.get("/rag/search")
def rag_search(q: str, k: int = 5, gpu_model: str | None = None):
    """Search the vector store for historical failures similar to the query text."""
    retriever = get_retriever(_rag_store(), k=k, gpu_model=gpu_model)
    results = retriever.invoke(q)
    return {
        "query": q,
        "results": [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in results
        ],
    }


@app.post("/rag/query")
def rag_query_endpoint(
    question: str = Body(..., embed=True),
    session_id: str = Body("default", embed=True),
):
    """Ask the RAG chain a diagnostic question, grounded in similar past failures."""
    retriever = get_retriever(_rag_store())
    answer = run_rag_query(question, retriever, session_id=session_id)
    return {"question": question, "session_id": session_id, "answer": answer}


@app.post("/rag/query/stream")
def rag_query_stream_endpoint(
    question: str = Body(..., embed=True),
    session_id: str = Body("default", embed=True),
):
    """Same as /rag/query, but streams the answer back as plain text chunks as
    Claude generates them, instead of waiting for the full response."""
    retriever = get_retriever(_rag_store())
    return StreamingResponse(
        run_rag_stream_query(question, retriever, session_id=session_id),
        media_type="text/plain",
    )
