"""Tests for the RAG module: document construction from SQLite data,
retriever similarity ordering, and the LCEL chain with a mocked ChatAnthropic.

No real embedding model or Anthropic API call is used -- a small deterministic
fake Embeddings implementation and LangChain's FakeListChatModel stand in for
them, so these tests run offline and fast.
"""

import json

import pytest
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel

import rag.pipeline as pipeline
from rag.ingestion import _documents_from_result, documents_from_all_runs, documents_from_run
from rag.retriever import get_retriever
from storage import db

# Small fixed vocabulary so a fake embedding vector encodes which keywords a
# text mentions -- similarity then naturally ranks documents sharing the
# query's keyword highest, without needing a real embedding model.
_VOCAB = ["temperature", "power", "memory", "utilization"]


class _FakeEmbeddings(Embeddings):
    """Deterministic bag-of-keywords embeddings, standing in for HuggingFaceEmbeddings in tests."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        lowered = text.lower()
        return [float(lowered.count(word)) for word in _VOCAB]


def _make_validation_row(run_id: str, timestamp: float, checks: dict) -> dict:
    return {
        "run_id": run_id,
        "timestamp": timestamp,
        "overall_passed": 0,
        "results_json": json.dumps(checks),
    }


class TestDocumentCreation:
    """Document construction from mock SQLite validation_results rows."""

    def test_failed_check_becomes_a_document(self):
        row = _make_validation_row("run-1", 1700000000.0, {
            "temperature_c": {"passed": False, "value": 90.0, "threshold": 83.0, "rule": "<="},
        })

        documents = _documents_from_result(row)

        assert len(documents) == 1
        doc = documents[0]
        assert "run-1" in doc.page_content
        assert "temperature_c" in doc.page_content
        assert doc.metadata["run_id"] == "run-1"
        assert doc.metadata["metric_name"] == "temperature_c"
        assert doc.metadata["gpu_model"] == "RTX 3090 Ti"
        assert doc.metadata["timestamp"] == 1700000000.0

    def test_passed_checks_produce_no_documents(self):
        row = _make_validation_row("run-2", 1700000001.0, {
            "temperature_c": {"passed": True, "value": 70.0, "threshold": 83.0, "rule": "<="},
        })

        assert _documents_from_result(row) == []

    def test_deviation_and_severity_for_max_rule(self):
        # 90 vs an 83 ceiling is a ~8.4% overshoot -- under the 10% severity bar
        row = _make_validation_row("run-3", 1700000002.0, {
            "temperature_c": {"passed": False, "value": 90.0, "threshold": 83.0, "rule": "<="},
        })

        doc = _documents_from_result(row)[0]

        assert doc.metadata["deviation_pct"] == pytest.approx(8.43, abs=0.01)
        assert doc.metadata["severity"] == "warning"

    def test_deviation_and_severity_for_min_rule(self):
        # 70 vs an 85 floor is a ~17.6% shortfall -- over the 10% severity bar
        row = _make_validation_row("run-4", 1700000003.0, {
            "gpu_utilization_pct": {"passed": False, "value": 70.0, "threshold": 85.0, "rule": ">="},
        })

        doc = _documents_from_result(row)[0]

        assert doc.metadata["deviation_pct"] == pytest.approx(17.65, abs=0.01)
        assert doc.metadata["severity"] == "critical"

    def test_multiple_failed_checks_produce_multiple_documents(self):
        row = _make_validation_row("run-5", 1700000004.0, {
            "temperature_c": {"passed": False, "value": 90.0, "threshold": 83.0, "rule": "<="},
            "power_draw_w": {"passed": False, "value": 500.0, "threshold": 445.0, "rule": "<="},
            "clock_graphics_mhz": {"passed": True, "value": 1800.0, "threshold": 1500.0, "rule": ">="},
        })

        documents = _documents_from_result(row)

        assert len(documents) == 2
        assert {d.metadata["metric_name"] for d in documents} == {"temperature_c", "power_draw_w"}

    @pytest.fixture(autouse=True)
    def _isolated_db(self, tmp_path, monkeypatch):
        monkeypatch.setattr(db, "DB_PATH", tmp_path / "test_gpu_validation.db")
        db.init_db()

    def test_documents_from_run_reads_from_sqlite(self):
        db.log_validation_result("run-6", False, {
            "temperature_c": {"passed": False, "value": 90.0, "threshold": 83.0, "rule": "<="},
        })

        documents = documents_from_run("run-6")

        assert len(documents) == 1
        assert documents[0].metadata["run_id"] == "run-6"

    def test_documents_from_run_missing_run_returns_empty(self):
        assert documents_from_run("nonexistent-run") == []

    def test_documents_from_all_runs_covers_every_run(self):
        db.log_validation_result("run-7", False, {
            "temperature_c": {"passed": False, "value": 90.0, "threshold": 83.0, "rule": "<="},
        })
        db.log_validation_result("run-8", False, {
            "power_draw_w": {"passed": False, "value": 500.0, "threshold": 445.0, "rule": "<="},
        })

        documents = documents_from_all_runs()

        assert {d.metadata["run_id"] for d in documents} == {"run-7", "run-8"}


class TestRetrieverSimilarityOrdering:
    """MMR retriever ranks the most similar historical failure first."""

    @pytest.fixture
    def store(self, tmp_path):
        from langchain_core.documents import Document

        documents = [
            Document(
                page_content="GPU temperature exceeded threshold, overheating detected.",
                metadata={"run_id": "run-temp", "gpu_model": "RTX 3090 Ti", "metric_name": "temperature_c"},
            ),
            Document(
                page_content="GPU power draw exceeded threshold, power supply issue.",
                metadata={"run_id": "run-power", "gpu_model": "RTX 3090 Ti", "metric_name": "power_draw_w"},
            ),
            Document(
                page_content="GPU memory utilization exceeded threshold, VRAM pressure.",
                metadata={"run_id": "run-memory", "gpu_model": "RTX 3090 Ti", "metric_name": "memory_used_mb"},
            ),
        ]
        return Chroma.from_documents(
            documents=documents,
            embedding=_FakeEmbeddings(),
            collection_name="test_failures",
            persist_directory=str(tmp_path / "chroma"),
        )

    def test_top_result_matches_the_query_topic(self, store):
        retriever = get_retriever(store, k=3)

        results = retriever.invoke("temperature problem")

        assert results[0].metadata["run_id"] == "run-temp"

    def test_different_query_returns_different_top_result(self, store):
        retriever = get_retriever(store, k=3)

        results = retriever.invoke("power supply problem")

        assert results[0].metadata["run_id"] == "run-power"

    def test_gpu_model_filter_is_applied(self, store):
        retriever = get_retriever(store, k=3, gpu_model="RTX 4090")

        results = retriever.invoke("temperature problem")

        assert results == []


class TestPipelineWithMockedChatAnthropic:
    """LCEL chain behavior with a mocked ChatAnthropic response."""

    @pytest.fixture
    def retriever(self, tmp_path):
        from langchain_core.documents import Document

        documents = [
            Document(
                page_content="GPU temperature exceeded threshold, overheating detected.",
                metadata={"run_id": "run-temp", "gpu_model": "RTX 3090 Ti", "severity": "critical"},
            ),
        ]
        store = Chroma.from_documents(
            documents=documents,
            embedding=_FakeEmbeddings(),
            collection_name="test_pipeline",
            persist_directory=str(tmp_path / "chroma_pipeline"),
        )
        return get_retriever(store, k=1)

    def test_query_returns_mocked_response(self, monkeypatch, retriever):
        monkeypatch.setattr(
            pipeline,
            "ChatAnthropic",
            lambda **kwargs: FakeListChatModel(responses=["Mocked diagnosis: check thermal paste."]),
        )

        answer = pipeline.query("temperature problem", retriever, session_id="test-session")

        assert answer == "Mocked diagnosis: check thermal paste."

    def test_fallback_notice_when_no_relevant_matches(self, monkeypatch, retriever):
        monkeypatch.setattr(
            pipeline,
            "ChatAnthropic",
            lambda **kwargs: FakeListChatModel(responses=["Mocked ungrounded diagnosis."]),
        )
        monkeypatch.setattr(pipeline, "has_relevant_matches", lambda *a, **k: False)

        answer = pipeline.query("completely unrelated query about ducks", retriever)

        assert answer.startswith(pipeline.NO_MATCH_FALLBACK)
        assert "Mocked ungrounded diagnosis." in answer

    def test_session_history_persists_across_queries(self, monkeypatch, retriever):
        monkeypatch.setattr(
            pipeline,
            "ChatAnthropic",
            lambda **kwargs: FakeListChatModel(responses=["First answer.", "Second answer."]),
        )

        pipeline.query("What is wrong?", retriever, session_id="follow-up-session")
        pipeline.query("Can you say more?", retriever, session_id="follow-up-session")

        history = pipeline._get_session_history("follow-up-session")
        assert len(history.messages) == 4  # 2 human + 2 AI messages

    def test_stream_query_yields_the_full_answer_across_chunks(self, monkeypatch, retriever):
        monkeypatch.setattr(
            pipeline,
            "ChatAnthropic",
            lambda **kwargs: FakeListChatModel(responses=["Streamed diagnosis text."]),
        )

        chunks = list(pipeline.stream_query("temperature problem", retriever, session_id="stream-session"))

        assert "".join(chunks) == "Streamed diagnosis text."

    def test_stream_query_fallback_notice_is_first_chunk(self, monkeypatch, retriever):
        monkeypatch.setattr(
            pipeline,
            "ChatAnthropic",
            lambda **kwargs: FakeListChatModel(responses=["Ungrounded streamed answer."]),
        )
        monkeypatch.setattr(pipeline, "has_relevant_matches", lambda *a, **k: False)

        chunks = list(pipeline.stream_query("completely unrelated query about ducks", retriever))

        assert chunks[0] == pipeline.NO_MATCH_FALLBACK
        assert "".join(chunks[1:]) == "Ungrounded streamed answer."
