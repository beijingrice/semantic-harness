import json
import sys
import time
from pathlib import Path
from threading import Barrier, Event, current_thread
from types import SimpleNamespace

import httpx
import pytest
from pydantic import BaseModel

from app.backends import BackendConfig, BackendError
from app.config import Settings
from app.llm import ChatClient, LLMError
from examples import evaluate
from examples.codex_backend import CodexExecBackend


def test_headline_metrics_include_all_cases():
    results = [
        {"id": "a", "category": "mixed", "expected": "PASS", "predicted": "PASS", "correct": True},
        {"id": "b", "category": "mixed", "expected": "PASS", "predicted": "FAIL", "correct": False},
        {"id": "c", "category": "mixed", "expected": "PASS", "predicted": "REVIEW_REQUIRED", "correct": False},
        {"id": "d", "category": "undecided", "expected": "FAIL", "predicted": "ERROR", "correct": False},
        {"id": "e", "category": "mixed", "expected": "FAIL", "predicted": "PASS", "correct": False},
        {"id": "f", "category": "mixed", "expected": "FAIL", "predicted": "FAIL", "correct": True},
    ]
    metrics = {
        "end_to_end_accuracy": 0.3333,
        "auto_decision_rate": 0.6667,
        "review_required_rate": 0.1667,
        "error_rate": 0.1667,
        "accuracy_on_decided": 0.5,
    }
    summary = evaluate.summarize(results)
    assert {key: summary[key] for key in metrics} == metrics
    assert summary["accuracy"] == summary["accuracy_on_decided"]
    assert summary["false_pass"] == summary["false_fail"] == 1
    assert summary["by_category"]["mixed"]["end_to_end_accuracy"] == 0.4
    assert summary["by_category"]["mixed"]["false_fail"] == 1
    assert summary["by_category"]["undecided"]["end_to_end_accuracy"] == 0.0
    assert summary["by_category"]["undecided"]["accuracy"] is None
    empty = evaluate.summarize([])
    assert all(empty[key] is None for key in metrics)


@pytest.fixture
def run_evaluation(monkeypatch, tmp_path):
    def run(workers, verdicts, stop_after_errors=0):
        clients = []
        preflight = []
        first_pair = Barrier(2) if workers == 2 else None
        error_limit_reached = Event()
        rows = [{"id": f"case-{i}", "en": str(i), "zh": "要求", "expected": "PASS"} for i in range(len(verdicts))]
        dataset = tmp_path / "cases.json"
        dataset.write_text(json.dumps(rows), encoding="utf-8")
        report = tmp_path / "report.json"

        class FakeClient:
            def __init__(self, settings):
                self.settings = settings
                self.threads = set()
                self.closed = False
                clients.append(self)

            def choose_model(self, requested):
                preflight.append(("primary", self))
                return "primary"

            def choose_verifier(self, requested):
                preflight.append(("verifier", self))
                return "verifier"

            def validate_model_access(self, primary, verifier):
                assert (primary, verifier) == ("primary", "verifier")
                preflight.append(("validate", self))

            def start_trace(self):
                self.trace = []

            def trace_snapshot(self):
                return self.trace.copy()

            def close(self):
                assert not self.closed
                self.closed = True

        class FakeChecker:
            def __init__(self, client, settings):
                assert settings is clients[0].settings
                assert not hasattr(client, "checker")
                client.checker = self
                self.client = client

            def check(self, english, chinese, primary, verifier, mode):
                assert preflight == [(step, clients[0]) for step in ("primary", "verifier", "validate")]
                assert (primary, verifier, mode) == ("primary", "verifier", "dual")
                index = int(english)
                self.client.threads.add(current_thread())
                self.client.trace.append({"case": f"case-{index}"})
                if first_pair is not None and index < 2:
                    first_pair.wait(timeout=5)
                if stop_after_errors and index == 0:
                    assert error_limit_reached.wait(timeout=5)
                time.sleep(0.030 if index == 0 else 0.002)
                verdict = verdicts[index]
                if verdict == "ERROR":
                    raise LLMError("simulated execution failure")
                return SimpleNamespace(verdict=verdict, model_dump=lambda: {"initial_verdict": verdict})

        monkeypatch.setattr(evaluate, "ChatClient", FakeClient)
        monkeypatch.setattr(evaluate, "Checker", FakeChecker)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "evaluate",
                "--cases",
                str(dataset),
                "--workers",
                str(workers),
                "--stop-after-errors",
                str(stop_after_errors),
                "--out",
                str(report),
            ],
        )
        if stop_after_errors:

            def track_limit(*args, **kwargs):
                print(*args, **kwargs)
                if args[0].startswith("已达到 ERROR 停止阈值"):
                    error_limit_reached.set()

            monkeypatch.setattr(evaluate, "print", track_limit, raising=False)
            with pytest.raises(SystemExit) as exc:
                evaluate.main()
            assert exc.value.code == 1
        else:
            evaluate.main()
        assert len(clients) == workers
        assert all(client.closed and len(client.threads) == 1 for client in clients)
        assert len(set().union(*(client.threads for client in clients))) == workers
        return json.loads(report.read_text(encoding="utf-8"))

    return run


def test_parallel_evaluation_preserves_order_predictions_and_case_traces(run_evaluation, capsys):
    verdicts = ["PASS", "FAIL", "REVIEW_REQUIRED", "PASS"]
    serial = run_evaluation(1, verdicts)
    serial_output = capsys.readouterr().out
    parallel = run_evaluation(2, verdicts)
    parallel_output = capsys.readouterr().out
    expected = [(f"case-{index}", verdict) for index, verdict in enumerate(verdicts)]
    for report, output in [(serial, serial_output), (parallel, parallel_output)]:
        assert [(case["id"], case["predicted"]) for case in report["cases"]] == expected
        assert all(case["requests"] == [{"case": case["id"]}] for case in report["cases"])
        assert (
            "端到端正确率 0.5000 (2/4) | 自动判决 0.7500 | 转人工 0.2500 | ERROR 0.0000 | 误判PASS 0 | 误判FAIL 1"
            in output
        )
    progress = [f"[{index:02d}/04] {case_id} " for index, (case_id, _) in enumerate(expected, 1)]
    assert [serial_output.index(line) for line in progress] == sorted(serial_output.index(line) for line in progress)
    assert all(line in parallel_output for line in progress)


def test_parallel_evaluation_stops_submitting_at_error_limit(run_evaluation):
    report = run_evaluation(2, ["PASS", "ERROR", "PASS", "PASS"], stop_after_errors=1)
    assert [(case["id"], case["predicted"]) for case in report["cases"]] == [("case-0", "PASS"), ("case-1", "ERROR")]
    summary = report["summary"]
    assert summary["configured_total"] == 4
    assert summary["stopped_early"] is True
    assert summary["error_rate"] == 0.5


@pytest.fixture
def fake_codex(monkeypatch, tmp_path):
    binary = tmp_path / "fake codex"
    binary.write_text(
        """#!/bin/sh
set -eu
test -z "$(ls -A)"
cat > "$FAKE_CODEX_DIR/prompt"
printf '%s\\n' "$@" > "$FAKE_CODEX_DIR/argv"
pwd > "$FAKE_CODEX_DIR/cwd"
while [ "$1" != "-o" ]; do shift; done
case "$FAKE_CODEX_MODE" in
    fail)
        printf '%0350d\\n' 0 >&2
        cat "$FAKE_CODEX_DIR/prompt" >&2
        printf '\\nfixture failure\\n' >&2
        exit 7
        ;;
    missing) exit 0 ;;
    empty) printf '  \\n' > "$2"; exit 0 ;;
    timeout) exec sleep 2 ;;
esac
printf '  {"ok": true}\\n' > "$2"
printf 'tokens used\\n123\\n' >&2
""",
        encoding="utf-8",
    )
    binary.chmod(0o755)
    monkeypatch.setenv("CODEX_BIN", str(binary))
    monkeypatch.setenv("FAKE_CODEX_DIR", str(tmp_path))
    monkeypatch.delenv("CODEX_REASONING_EFFORT", raising=False)
    monkeypatch.setenv("FAKE_CODEX_MODE", "success")
    return tmp_path


@pytest.fixture
def codex_payload():
    return {
        "model": "gpt-6-luna@high",
        "messages": [
            {"role": "system", "content": "Test system instructions"},
            {"role": "user", "content": "测试 JSON 请求"},
            {"role": "assistant", "content": "Previous JSON answer"},
            {"role": "user", "content": "Try again"},
        ],
        "max_tokens": 128,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "strict": True,
                "schema": {"type": "object", "properties": {"ok": {"type": "boolean"}}},
            },
        },
    }


def test_codex_backend_renders_prompt_and_routes_effort(fake_codex, codex_payload, monkeypatch):
    monkeypatch.setenv("CODEX_REASONING_EFFORT", "low")
    backend = CodexExecBackend(BackendConfig("", "", 5), None)
    assert backend.models() is None
    response = backend.complete(codex_payload)
    assert response.content == '{"ok": true}'
    assert response.finish_reason == "stop"
    assert response.completion_tokens == 123
    assert response.http_status == 200
    blocks = (fake_codex / "prompt").read_text(encoding="utf-8").split("\n\n")
    assert blocks[0].startswith("You are being used as a plain text-completion model.")
    assert blocks[1:5] == [
        "### SYSTEM\nTest system instructions",
        "### USER\n测试 JSON 请求",
        "### ASSISTANT (your previous answer)\nPrevious JSON answer",
        "### USER\nTry again",
    ]
    assert (
        json.loads(blocks[5].removeprefix("The JSON must satisfy this JSON Schema: "))
        == codex_payload["response_format"]["json_schema"]["schema"]
    )
    argv = (fake_codex / "argv").read_text(encoding="utf-8").splitlines()
    workdir = Path((fake_codex / "cwd").read_text(encoding="utf-8").strip())
    assert argv == [
        "exec",
        "--ephemeral",
        "--skip-git-repo-check",
        "--ignore-user-config",
        "-s",
        "read-only",
        "-m",
        "gpt-6-luna",
        "-c",
        "model_reasoning_effort=high",
        "-o",
        argv[-2],
        "-",
    ]
    assert Path(argv[-2]).resolve() == workdir.resolve() / "last.txt"
    assert not workdir.exists()


def test_codex_failure_redacts_prompt(fake_codex, codex_payload, monkeypatch):
    monkeypatch.setenv("FAKE_CODEX_MODE", "fail")
    backend = CodexExecBackend(BackendConfig("", "", 5), None)
    with pytest.raises(BackendError, match=r"code 7:[\s\S]*fixture failure") as exc:
        backend.complete(codex_payload)
    assert "测试 JSON 请求" not in str(exc.value)
    assert "Try again" not in str(exc.value)
    assert len(str(exc.value)) < 350


@pytest.mark.parametrize(
    ("mode", "message"),
    [("missing", "no readable output file"), ("empty", "empty output file"), ("timeout", "timed out")],
)
def test_codex_reports_execution_problems(fake_codex, codex_payload, monkeypatch, mode, message):
    monkeypatch.setenv("FAKE_CODEX_MODE", mode)
    backend = CodexExecBackend(BackendConfig("", "", 0.05 if mode == "timeout" else 5), None)
    with pytest.raises(BackendError, match=message):
        backend.complete(codex_payload)


def test_codex_missing_binary_and_invalid_effort(fake_codex, codex_payload, monkeypatch):
    monkeypatch.setenv("CODEX_BIN", str(fake_codex / "missing binary"))
    backend = CodexExecBackend(BackendConfig("", "", 5), None)
    with pytest.raises(BackendError, match="executable not found"):
        backend.complete(codex_payload)
    with pytest.raises(BackendError, match="Invalid Codex reasoning effort"):
        backend.complete({**codex_payload, "model": "gpt-6-luna@turbo"})
    monkeypatch.setenv("CODEX_REASONING_EFFORT", "turbo")
    with pytest.raises(BackendError, match="Invalid Codex reasoning effort"):
        backend.complete({**codex_payload, "model": "gpt-6-luna"})


def test_chat_client_structured_through_codex(fake_codex):
    class Answer(BaseModel):
        ok: bool

    def unexpected_http(request):
        pytest.fail(f"Unexpected HTTP request: {request.url}")

    settings = Settings(
        backend="examples.codex_backend:CodexExecBackend",
        model_discovery="off",
        model="gpt-6-luna",
        schema_mode="on",
    )
    with httpx.Client(transport=httpx.MockTransport(unexpected_http)) as http_client:
        client = ChatClient(settings, http_client=http_client)
        try:
            model = client.choose_model()
            assert client.validate_model_access(model) == "skipped"
            answer = client.structured(model, "Output an object.", "Is it OK?", Answer, "answer")
            assert answer == Answer(ok=True)
            assert client.trace_snapshot()[0]["completion_tokens"] == 123
        finally:
            client.close()
    assert "model_reasoning_effort=medium" in (fake_codex / "argv").read_text(encoding="utf-8").splitlines()
