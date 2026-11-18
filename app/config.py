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


# Allowed values of the enumerated settings. Validated once, when Settings is built.
CHOICES = {
    "pipeline": ("extraction", "direct_hardened"),
    "evidence_mode": ("paragraph", "quote"),
    "extraction_prompt": ("coarse", "proposition", "legacy"),
    "schema_mode": ("legacy", "on", "off", "auto"),
    "model_discovery": ("required", "optional", "off"),
}
ENV_NAMES = {
    "pipeline": "LM_PIPELINE",
    "evidence_mode": "LM_EVIDENCE_MODE",
    "extraction_prompt": "LM_EXTRACTION_PROMPT",
    "schema_mode": "LM_SCHEMA_MODE",
    "model_discovery": "LM_MODEL_DISCOVERY",
}


@dataclass(frozen=True)
class Settings:
    # --- Backend connection
    base_url: str = _str("LM_BASE_URL", "http://127.0.0.1:1234/v1", rstrip_slash=True)
    model: str = _str("LM_MODEL")
    verifier_model: str = _str("LM_VERIFIER_MODEL")
    api_key: str = _raw("LM_API_KEY", "lm-studio")
    timeout_seconds: float = _float("LM_TIMEOUT_SECONDS", "180")
    max_tokens: int = _int("LM_MAX_TOKENS", "3072")
    backend: str = _str("LM_BACKEND", "openai")
    # Optional separate verifier endpoint; empty values fall back to the primary.
    verifier_backend: str = _str("LM_VERIFIER_BACKEND")
    verifier_base_url: str = _str("LM_VERIFIER_BASE_URL", rstrip_slash=True)
    verifier_api_key: str = _raw("LM_VERIFIER_API_KEY")
    # legacy: follow LM_JSON_SCHEMA; on: always native JSON Schema; off: prompt-only;
    # auto: try native, fall back once only when the provider explicitly rejects it.
    schema_mode: str = _str("LM_SCHEMA_MODE", "legacy", lower=True)
    # required: the model list must be readable; optional: check it when available;
    # off: never call /models (managed APIs).
    model_discovery: str = _str("LM_MODEL_DISCOVERY", "required", lower=True)
    # One bounded recovery request after a truncated or empty generation.
    generation_recovery: bool = _bool("LM_GENERATION_RECOVERY", True)
    json_schema: bool = _bool("LM_JSON_SCHEMA", False)

    # --- Pipeline selection
    pipeline: str = _str("LM_PIPELINE", "extraction", lower=True)
    # paragraph: models cite paragraph IDs; quote: models retype exact quotes (A/B only).
    evidence_mode: str = _str("LM_EVIDENCE_MODE", "paragraph", lower=True)

    # --- Extraction pipeline options
    # Skip model extraction for short sources whose clauses can be anchored verbatim.
    extraction_fast_path: bool = _bool("LM_EXTRACTION_FAST_PATH", True)
    # Token cap for extracting a short single paragraph; longer sources use LM_MAX_TOKENS.
    extraction_max_tokens: int = _int("LM_EXTRACTION_MAX_TOKENS", "1536")
    max_keywords: int = _int("MAX_KEYWORDS", "30")
    # coarse is the default; proposition and legacy are kept for A/B comparison.
    extraction_prompt: str = _str("LM_EXTRACTION_PROMPT", "coarse", lower=True)
    coverage_audit: bool = _bool("LM_COVERAGE_AUDIT", False)
    atomic_refinement: bool = _bool("LM_ATOMIC_REFINEMENT", False)

    def __post_init__(self):
        for name, allowed in CHOICES.items():
            value = getattr(self, name)
            if value not in allowed:
                hint = "（早期的 direct* 实验模式已在 v0.3.0 移除，可从 git tag v0.2.16 找回）" if name == "pipeline" else ""
                raise ConfigError(f"{ENV_NAMES[name]} 只能为 {' / '.join(allowed)}，当前值为 {value!r}{hint}")
        if self.pipeline == "direct_hardened" and self.evidence_mode != "paragraph":
            raise ConfigError("LM_PIPELINE=direct_hardened 要求 LM_EVIDENCE_MODE=paragraph")
