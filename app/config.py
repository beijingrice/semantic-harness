"""Configuration shared by the two models; the server can be changed via LM_BASE_URL.

Environment variables are read every time ``Settings()`` is constructed (not
once at import time), so runtime changes and test isolation behave as expected.
Explicit keyword arguments always override the environment.
"""
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


class ConfigError(ValueError):
    """Invalid operator configuration (env var or setting). Never a semantic FAIL."""


def env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _env_number(name: str, default: str, cast, kind: str):
    raw = os.getenv(name, default)
    try:
        value = cast(raw.strip())
    except (TypeError, ValueError):
        raise ConfigError(f"{name} 必须是{kind}，当前值为 {raw!r}") from None
    if value <= 0:
        raise ConfigError(f"{name} 必须大于 0，当前值为 {raw!r}")
    return value


def env_int(name: str, default: str) -> int:
    return _env_number(name, default, int, "正整数")


def env_float(name: str, default: str) -> float:
    return _env_number(name, default, float, "正数")


def _str(name: str, default: str = "", *, lower: bool = False, rstrip_slash: bool = False):
    def read() -> str:
        value = os.getenv(name, default).strip()
        if lower:
            value = value.lower()
        if rstrip_slash:
            value = value.rstrip("/")
        return value
    return field(default_factory=read)


def _raw(name: str, default: str = ""):
    return field(default_factory=lambda: os.getenv(name, default))


def _bool(name: str, default: bool):
    return field(default_factory=lambda: env_bool(name, default))


def _int(name: str, default: str):
    return field(default_factory=lambda: env_int(name, default))


def _float(name: str, default: str):
    return field(default_factory=lambda: env_float(name, default))


@dataclass(frozen=True)
class Settings:
    base_url: str = _str("LM_BASE_URL", "http://127.0.0.1:1234/v1", rstrip_slash=True)
    model: str = _str("LM_MODEL")
    verifier_model: str = _str("LM_VERIFIER_MODEL")
    api_key: str = _raw("LM_API_KEY", "lm-studio")
    timeout_seconds: float = _float("LM_TIMEOUT_SECONDS", "180")
    max_tokens: int = _int("LM_MAX_TOKENS", "3072")
    # v0.2.16 provider abstraction. The legacy LM_BASE_URL/API_KEY continue to work.
    backend: str = _str("LM_BACKEND", "openai")
    verifier_backend: str = _str("LM_VERIFIER_BACKEND")
    verifier_base_url: str = _str("LM_VERIFIER_BASE_URL", rstrip_slash=True)
    verifier_api_key: str = _raw("LM_VERIFIER_API_KEY")
    # legacy: honor LM_JSON_SCHEMA; on: require native; off: prompt-only;
    # auto: attempt native once, fall back ONLY on explicit schema rejection.
    schema_mode: str = _str("LM_SCHEMA_MODE", "legacy", lower=True)
    # required: must enumerate models; optional: verify when list supported;
    # off: do not call /models. Useful for managed deployments.
    model_discovery: str = _str("LM_MODEL_DISCOVERY", "required", lower=True)
    generation_recovery: bool = _bool("LM_GENERATION_RECOVERY", True)
    # extraction (default) or direct_hardened (experimental paragraph-direct).
    pipeline: str = _str("LM_PIPELINE", "extraction", lower=True)
    # v0.2.9.2: source-only fast paths for a cohesive short clause or safe two-clause pairs.
    extraction_fast_path: bool = _bool("LM_EXTRACTION_FAST_PATH", True)
    # Independent bounded generation for non-fast-path short single paragraphs.
    # Long/multi-paragraph sources retain LM_MAX_TOKENS.
    extraction_max_tokens: int = _int("LM_EXTRACTION_MAX_TOKENS", "1536")
    json_schema: bool = _bool("LM_JSON_SCHEMA", False)
    max_keywords: int = _int("MAX_KEYWORDS", "30")
    # coarse is the v0.2.5+ default; proposition retains v0.2.4 for A/B.
    extraction_prompt: str = _str("LM_EXTRACTION_PROMPT", "coarse", lower=True)
    evidence_mode: str = _str("LM_EVIDENCE_MODE", "paragraph", lower=True)
    coverage_audit: bool = _bool("LM_COVERAGE_AUDIT", False)
    atomic_refinement: bool = _bool("LM_ATOMIC_REFINEMENT", False)
