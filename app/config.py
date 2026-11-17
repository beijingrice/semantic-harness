"""Configuration shared by the two models; the server can be changed via LM_BASE_URL."""
from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()


def env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    base_url: str = os.getenv("LM_BASE_URL", "http://127.0.0.1:1234/v1").rstrip("/")
    model: str = os.getenv("LM_MODEL", "").strip()
    verifier_model: str = os.getenv("LM_VERIFIER_MODEL", "").strip()
    api_key: str = os.getenv("LM_API_KEY", "lm-studio")
    timeout_seconds: float = float(os.getenv("LM_TIMEOUT_SECONDS", "180"))
    max_tokens: int = int(os.getenv("LM_MAX_TOKENS", "3072"))
    # v0.2.16 provider abstraction. The legacy LM_BASE_URL/API_KEY continue to work.
    backend: str = os.getenv("LM_BACKEND", "openai").strip()
    verifier_backend: str = os.getenv("LM_VERIFIER_BACKEND", "").strip()
    verifier_base_url: str = os.getenv("LM_VERIFIER_BASE_URL", "").strip().rstrip("/")
    verifier_api_key: str = os.getenv("LM_VERIFIER_API_KEY", "")
    # legacy: honor LM_JSON_SCHEMA; on: require native; off: prompt-only;
    # auto: attempt native once, fall back ONLY on explicit schema rejection.
    schema_mode: str = os.getenv("LM_SCHEMA_MODE", "legacy").strip().lower()
    # required: must enumerate models; optional: verify when list supported;
    # off: do not call /models. Useful for managed deployments.
    model_discovery: str = os.getenv("LM_MODEL_DISCOVERY", "required").strip().lower()
    generation_recovery: bool = env_bool("LM_GENERATION_RECOVERY", True)
    # v0.2.14: additionally supports experimental direct_relations.
    pipeline: str = os.getenv("LM_PIPELINE", "extraction").strip().lower()
    # v0.2.9.2: source-only fast paths for a cohesive short clause or safe two-clause pairs.
    extraction_fast_path: bool = env_bool("LM_EXTRACTION_FAST_PATH", True)
    # Independent bounded generation for non-fast-path short single paragraphs.
    # Long/multi-paragraph sources retain LM_MAX_TOKENS.
    extraction_max_tokens: int = int(os.getenv("LM_EXTRACTION_MAX_TOKENS", "1536"))
    json_schema: bool = env_bool("LM_JSON_SCHEMA")
    max_keywords: int = int(os.getenv("MAX_KEYWORDS", "30"))
    # coarse is the v0.2.5+ default; proposition retains v0.2.4 for A/B.
    extraction_prompt: str = os.getenv("LM_EXTRACTION_PROMPT", "coarse").strip().lower()
    evidence_mode: str = os.getenv("LM_EVIDENCE_MODE", "paragraph").strip().lower()
    coverage_audit: bool = env_bool("LM_COVERAGE_AUDIT", False)
    atomic_refinement: bool = env_bool("LM_ATOMIC_REFINEMENT", False)
