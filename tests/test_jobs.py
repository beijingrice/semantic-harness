from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app import jobs
from app.api import app
from app.backends import BackendError
from app.config import ConfigError, Settings
from app.llm import LLMError
from app.models import CheckRequest, CheckResult

REQUEST = CheckRequest(english="The report requires a date.", chinese="报告必须包含日期。", mode="primary")
RESULT = CheckResult(
    verdict="PASS",
    model="main",
    verifier_model="",
    english_keywords=[],
    chinese_keywords=[],
    links=[],
    missing_in_chinese=[],
    missing_in_english=[],
    conflicts=[],
    review=None,
    review_issues=[],
    mode="primary",
    deterministic_issues=[],
    initial_verdict="PASS",
    summary="Requirements match.",
)


@pytest.fixture
def fake_check(monkeypatch):
    close = Mock()
    check = Mock(return_value=RESULT)
    monkeypatch.setattr(jobs, "ChatClient", lambda settings: SimpleNamespace(settings=settings, close=close))
    monkeypatch.setattr(jobs.Checker, "check", check)
    return check, close


@pytest.fixture
def manager(monkeypatch):
    manager = jobs.JobManager(max_workers=1, max_pending=2)
    monkeypatch.setattr(jobs, "_manager", manager)
    yield manager
    manager._executor.shutdown()


def test_job_lifecycle_and_worker_limit(manager, fake_check):
    check, close = fake_check
    started = [Event(), Event()]
    release = [Event(), Event()]
    second_request = REQUEST.model_copy(update={"english": "Another requirement."})

    def slow_check(english, *_):
        index = 0 if english == REQUEST.english else 1
        started[index].set()
        assert release[index].wait(1)
        return RESULT

    check.side_effect = slow_check
    with TestClient(app) as client:
        try:
            first_id, first_future = manager.submit_with_future(REQUEST)
            assert started[0].wait(1)
            assert manager.get(first_id)["status"] == "running"

            second_id, second_future = manager.submit_with_future(second_request)
            url = f"/api/jobs/{second_id}"
            response = client.get(url)
            assert response.status_code == 200
            queued = response.json()
            assert queued["status"] == "queued" and queued["position"] == 0
            assert queued["started_at"] is None and queued["finished_at"] is None
            assert queued["result"] is None and queued["error"] is None
            assert not started[1].is_set()

            release[0].set()
            assert started[1].wait(1)
            running = client.get(url).json()
            assert running["status"] == "running" and running["position"] is None
            assert running["created_at"] <= running["started_at"]
            assert running["finished_at"] is None

            release[1].set()
            assert first_future.result(timeout=1) == second_future.result(timeout=1) == RESULT
            done = client.get(url).json()
            assert done["id"] == second_id and done["status"] == "done"
            assert done["started_at"] <= done["finished_at"]
            assert done["result"] == RESULT.model_dump(mode="json")
            assert done["error"] is None
            assert close.call_count == 2
        finally:
            for event in release:
                event.set()


def test_queue_full_is_429_for_both_endpoints(manager, fake_check):
    check, _ = fake_check
    started, release = Event(), Event()

    def slow_check(*_):
        started.set()
        assert release.wait(1)
        return RESULT

    check.side_effect = slow_check
    with TestClient(app) as client:
        try:
            manager.submit(REQUEST)
            assert started.wait(1)
            response = client.post("/api/jobs", json=REQUEST.model_dump())
            assert response.status_code == 202
            job_id = response.json()["job_id"]
            assert response.json()["status"] == "queued"
            assert len(job_id) == 32 and job_id == f"{int(job_id, 16):032x}"
            assert client.get(f"/api/jobs/{job_id}").json()["status"] == "queued"
            with pytest.raises(jobs.QueueFull):
                manager.submit(REQUEST)
            for endpoint in ("/api/jobs", "/api/check"):
                response = client.post(endpoint, json=REQUEST.model_dump())
                assert response.status_code == 429
                assert isinstance(response.json()["detail"], str)
        finally:
            release.set()


@pytest.mark.parametrize(
    "exception,kind,message",
    [
        (ConfigError("bad config"), "config", "配置错误：bad config"),
        (BackendError("bad backend"), "config", "配置错误：bad backend"),
        (LLMError("model unavailable"), "model", "model unavailable"),
        (RuntimeError("private details"), "internal", "检查时发生内部错误。"),
    ],
)
def test_job_errors_are_classified_and_client_is_closed(manager, fake_check, caplog, exception, kind, message):
    check, close = fake_check
    check.side_effect = exception
    job_id, future = manager.submit_with_future(REQUEST)
    future.result(timeout=1)
    job = TestClient(app).get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "error"
    assert job["error"] == {"kind": kind, "message": message}
    assert job["result"] is None
    assert job["finished_at"] is not None
    close.assert_called_once_with()
    if kind == "internal":
        assert "private details" in caplog.text
        assert "Traceback" in caplog.text


def test_finished_job_ttl(manager, fake_check, monkeypatch):
    now = 100.0
    monkeypatch.setattr(jobs, "monotonic", lambda: now)
    job_id, future = manager.submit_with_future(REQUEST)
    future.result(timeout=1)
    now += 3599
    assert manager.get(job_id)["status"] == "done"
    now += 2
    assert TestClient(app).get(f"/api/jobs/{job_id}").status_code == 404
    assert manager.get(job_id) is None


def test_unknown_job_is_404(manager):
    assert TestClient(app).get("/api/jobs/unknown").status_code == 404


def test_unknown_job_before_manager_initialization_with_invalid_settings(monkeypatch):
    monkeypatch.setattr(jobs, "_manager", None)
    monkeypatch.setenv("LM_MAX_CONCURRENT_CHECKS", "0")
    assert TestClient(app).get("/api/jobs/unknown").status_code == 404


def test_max_concurrent_checks_setting_and_submit_validation(manager, fake_check, monkeypatch):
    assert Settings().max_concurrent_checks == 2
    monkeypatch.setenv("LM_MAX_CONCURRENT_CHECKS", "3")
    assert Settings().max_concurrent_checks == 3
    monkeypatch.setenv("LM_MAX_CONCURRENT_CHECKS", "0")
    response = TestClient(app).post("/api/jobs", json=REQUEST.model_dump())
    assert response.status_code == 500
    assert response.json()["detail"].startswith("配置错误：LM_MAX_CONCURRENT_CHECKS")
    fake_check[0].assert_not_called()


def test_synchronous_check_returns_check_result(manager, fake_check):
    check, _ = fake_check
    response = TestClient(app).post("/api/check", json=REQUEST.model_dump())
    assert response.status_code == 200
    assert response.json() == RESULT.model_dump(mode="json")
    check.assert_called_once_with(REQUEST.english, REQUEST.chinese, None, None, REQUEST.mode)


def test_synchronous_check_model_error_is_502(manager, fake_check):
    check, _ = fake_check
    check.side_effect = LLMError("model unavailable")
    response = TestClient(app).post("/api/check", json=REQUEST.model_dump())
    assert response.status_code == 502
    assert response.json() == {"detail": "model unavailable"}
