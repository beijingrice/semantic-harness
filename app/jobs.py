"""In-process queue shared by the web and synchronous check endpoints."""

import logging
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from threading import Lock
from time import monotonic, time
from typing import Any, Literal
from uuid import uuid4

from app.backends import BackendError
from app.checker import Checker
from app.config import ConfigError, Settings
from app.llm import ChatClient, LLMError
from app.models import CheckRequest, CheckResult

logger = logging.getLogger(__name__)


class QueueFull(RuntimeError):
    """The maximum number of unfinished checks has been reached."""


@dataclass(frozen=True)
class JobError:
    kind: Literal["config", "model", "internal"]
    message: str


@dataclass(frozen=True)
class JobRecord:
    id: str
    status: Literal["queued", "running", "done", "error"]
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    result: dict[str, Any] | None = None
    error: JobError | None = None
    position: int | None = None


class JobManager:
    def __init__(self, max_workers: int, ttl_seconds: float = 3600, max_pending: int = 50):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._ttl_seconds = ttl_seconds
        self._max_pending = max_pending
        self._lock = Lock()
        self._jobs: dict[str, JobRecord] = {}
        self._queued: dict[str, None] = {}
        self._running = 0
        self._finished: deque[tuple[float, str]] = deque()

    def submit(self, request: CheckRequest) -> str:
        job_id, _ = self.submit_with_future(request)
        return job_id

    def submit_with_future(self, request: CheckRequest) -> tuple[str, Future[CheckResult | JobError]]:
        # Return the future at submission so synchronous callers can await it even after TTL expiry.
        with self._lock:
            self._purge_expired()
            if len(self._queued) + self._running >= self._max_pending:
                raise QueueFull("检查队列已满，请稍后重试。")
            job_id = uuid4().hex
            future = self._executor.submit(self._run, job_id, request)
            self._jobs[job_id] = JobRecord(id=job_id, status="queued", created_at=time())
            self._queued[job_id] = None
        return job_id, future

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            self._purge_expired()
            job = self._jobs.get(job_id)
            if job is not None and job.status == "queued":
                job = replace(job, position=list(self._queued).index(job_id))
        return asdict(job) if job is not None else None

    def _purge_expired(self) -> None:
        cutoff = monotonic() - self._ttl_seconds
        while self._finished and self._finished[0][0] < cutoff:
            _, job_id = self._finished.popleft()
            del self._jobs[job_id]

    def _run(self, job_id: str, request: CheckRequest) -> CheckResult | JobError:
        with self._lock:
            job = replace(self._jobs[job_id], status="running", started_at=time())
            self._jobs[job_id] = job
            del self._queued[job_id]
            self._running += 1

        try:
            client = ChatClient(Settings())
            try:
                result = Checker(client, client.settings).check(
                    request.english, request.chinese, request.model, request.verifier_model, request.mode
                )
            finally:
                client.close()
            job = replace(job, status="done", result=result.model_dump(mode="json"))
            outcome = result
        except (ConfigError, BackendError) as exc:
            outcome = JobError("config", f"配置错误：{exc}")
        except LLMError as exc:
            outcome = JobError("model", str(exc))
        except Exception:
            logger.exception("Check job %s failed", job_id)
            outcome = JobError("internal", "检查时发生内部错误。")

        with self._lock:
            if isinstance(outcome, JobError):
                job = replace(job, status="error", error=outcome)
            self._jobs[job_id] = replace(job, finished_at=time())
            self._running -= 1
            self._finished.append((monotonic(), job_id))
        return outcome


_manager: JobManager | None = None
_manager_lock = Lock()


def get_manager(settings: Settings | None = None) -> JobManager:
    global _manager
    with _manager_lock:
        if _manager is None:
            config = settings if settings is not None else Settings()
            _manager = JobManager(config.max_concurrent_checks)
        return _manager
