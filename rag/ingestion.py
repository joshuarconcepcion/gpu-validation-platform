"""Reads validation failures from SQLite and converts them into LangChain
Document objects for embedding into the RAG vector store.

We build langchain_core.documents.Document objects (rather than plain dicts)
because every downstream LangChain component (i.e. the embedder, the Chroma
store, and the retriever) is built around that type. Keeping the same type
end-to-end is what makes those components swappable independently.
"""

import json

from langchain_core.documents import Document

from storage import db

GPU_MODEL = "RTX 3090 Ti"  # single-GPU platform per CLAUDE.md; not tracked in the metrics schema itself

# A failed check counts as "critical" once its deviation from threshold clears
# this bar -- the same 10% line validation.benchmarks.REGRESSION_THRESHOLD uses
# for regression detection. Kept as a local constant (rather than importing
# validation.benchmarks) so this module still depends on only one sibling
# module (storage), matching the rest of the platform's import discipline.
SEVERITY_THRESHOLD_PCT = 10.0 


def _deviation_pct(value: float, threshold: float, rule: str) -> float:
    """Return how far a failed check's value sits past its threshold, as a positive percentage."""
    if threshold == 0: # division by 0 guard for possible config changes
        return 0.0
    if rule == "<=": # ceiling metrics (i.e. temp, power draw, memory util)
        return (value - threshold) / threshold * 100
    return (threshold - value) / threshold * 100 # floor metrics (i.e. gpu util, graphics clock)


def _severity(deviation_pct: float) -> str:
    """Classify a failed check as 'warning' or 'critical' based on deviation size."""
    return "critical" if deviation_pct >= SEVERITY_THRESHOLD_PCT else "warning" # classifies severity based on threshold percent


def _failure_document(run_id: str, timestamp: float, metric_name: str, check: dict) -> Document:
    """Build a Document describing one failed threshold check from a validation run; transforms SQL into LLM-readable sentence"""
    value = check["value"]
    threshold = check["threshold"]
    rule = check.get("rule", "<=")
    deviation_pct = round(_deviation_pct(value, threshold, rule), 2)
    severity = _severity(deviation_pct)

    page_content = ( # assembles all values above into plain-English sentence to embed
        f"On run {run_id} ({GPU_MODEL}) at timestamp {timestamp}, metric '{metric_name}' "
        f"measured {value}, violating its threshold of '{rule} {threshold}' by {deviation_pct}%. "
        f"Severity: {severity}."
    )

    return Document( # returns Document object (langchain)
        page_content=page_content, # page_content for semantic search
        metadata={ # metadata for exact-match filtering
            "run_id": run_id,
            "metric_name": metric_name,
            "gpu_model": GPU_MODEL,
            "timestamp": timestamp,
            "deviation_pct": deviation_pct,
            "severity": severity,
        },
    )


def _documents_from_result(result: dict) -> list[Document]:
    """Build a Document for every failed check in one validation_results row."""
    checks = json.loads(result["results_json"])
    documents = []
    for metric_name, check in checks.items():
        if not check.get("passed", True): # turns all failed checks into Documents and appends them to a list
            documents.append(_failure_document(result["run_id"], result["timestamp"], metric_name, check))
    return documents


def documents_from_run(run_id: str) -> list[Document]:
    """Build failure Documents for every failed threshold check in one validation run."""
    result = db.get_validation_result_for_run(run_id)
    if result is None:
        return []
    return _documents_from_result(result)


def documents_from_all_runs() -> list[Document]:
    """Build failure Documents for every failed threshold check across all validation runs."""
    documents = []
    for result in db.get_all_validation_results():
        documents.extend(_documents_from_result(result))
    return documents
