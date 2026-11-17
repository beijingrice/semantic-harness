"""Backend-independent structured generation and source-validated retry orchestration."""
import json
import re
import time
from typing import Any, Callable

import httpx
from pydantic import BaseModel, ValidationError

from app.backends import (
    BackendConfig,
    BackendError,
    BackendResponse,
    LLMBackend,
    UnsupportedNativeSchema,
    build_backend,
)
from app.config import ConfigError, Settings


class LLMError(RuntimeError):
    """模型断开、回复不可解析或协议不符合预期时抛出；绝不自动 PASS。"""


def parse_json_object(raw: str) -> dict:
    """允许 JSON 代码围栏/少量自然语言/think 标签，但不修补模型臆造的 JSON。"""
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.S | re.I).strip()
    if not cleaned:
        raise LLMError("模型回复为空（可能耗尽输出 token 或只输出了思考内容）")
    try:
        value = json.loads(cleaned)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    # 找第一段有效的 JSON object（不使用贪婪正则）
    decoder = json.JSONDecoder()
    for index, ch in enumerate(cleaned):
        if ch != "{":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[index:])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    raise LLMError("模型未返回有效 JSON 对象；可提高 LM_MAX_TOKENS 或开启 LM_JSON_SCHEMA")


class GenerationAnomaly(LLMError):
    """An incomplete generation, not a semantic verdict or ordinary schema error."""
    def __init__(self, category: str, message: str):
        super().__init__(message)
        self.category = category


class ChatClient:
    def __init__(self, settings: Settings, http_client: httpx.Client | None = None,
                 verifier_http_client: httpx.Client | None = None):
        self.settings = settings
        if settings.schema_mode not in ("legacy", "on", "off", "auto"):
            raise ConfigError("LM_SCHEMA_MODE must be legacy/on/off/auto")
        if settings.model_discovery not in ("required", "optional", "off"):
            raise ConfigError("LM_MODEL_DISCOVERY must be required/optional/off")
        self.http = http_client or httpx.Client(timeout=settings.timeout_seconds)
        self.owns_http = http_client is None
        separate = bool(settings.verifier_base_url or settings.verifier_api_key or settings.verifier_backend)
        self.verifier_http = verifier_http_client or self.http
        self._owns_verifier_http = False
        if separate and verifier_http_client is None and http_client is None:
            self.verifier_http = httpx.Client(timeout=settings.timeout_seconds)
            self._owns_verifier_http = True
        primary_cfg = BackendConfig(settings.base_url, settings.api_key, settings.timeout_seconds)
        verifier_cfg = BackendConfig(settings.verifier_base_url or settings.base_url,
                                     settings.verifier_api_key if settings.verifier_api_key else settings.api_key,
                                     settings.timeout_seconds)
        self.primary_backend = build_backend(settings.backend, primary_cfg, self.http)
        self.verifier_backend = (build_backend(settings.verifier_backend or settings.backend, verifier_cfg,
                                               self.verifier_http) if separate else self.primary_backend)
        self._runtime_verifier_model = settings.verifier_model
        self._runtime_primary_model = settings.model
        self._native_support: dict[tuple[str, str], bool] = {}
        # Request telemetry scoped per case. No raw prompt/response is persisted.
        self.request_trace: list[dict[str, Any]] = []

    def start_trace(self) -> None:
        self.request_trace.clear()

    def trace_snapshot(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self.request_trace]

    def _role_for(self, model: str) -> str:
        if self.verifier_backend is not self.primary_backend and model == self._runtime_verifier_model:
            return "verifier"
        return "primary"

    def _backend_for(self, model: str) -> LLMBackend:
        return self.verifier_backend if self._role_for(model) == "verifier" else self.primary_backend

    def _native_enabled(self, model: str) -> bool:
        mode = self.settings.schema_mode
        if mode == "off":
            return False
        if mode == "legacy":
            return self.settings.json_schema
        if mode == "on":
            return True
        return self._native_support.get((self._role_for(model), model), True)

    def _completion(self, *, payload: dict[str, Any], model: str,
                    schema_name: str, attempt: int, recovery: bool) -> str:
        """One API operation; backend normalizes response before semantic parsing."""
        began = time.monotonic()
        record: dict[str, Any] = {
            "stage": getattr(self, "_current_stage", schema_name), "schema": schema_name,
            "model": model, "attempt": attempt, "recovery": recovery,
            "backend_role": self._role_for(model),
            "schema_mode": "json_schema" if "response_format" in payload else "prompt_only",
            "max_tokens": payload.get("max_tokens"),
        }
        self.request_trace.append(record)
        try:
            data = self._backend_for(model).complete(payload)
            if not isinstance(data, BackendResponse):
                raise BackendError("Adapter must return BackendResponse")
            content = data.content
            reasoning = data.reasoning_content
            record.update({
                "http_status": data.http_status,
                "finish_reason": data.finish_reason,
                "prompt_tokens": data.prompt_tokens,
                "completion_tokens": data.completion_tokens,
                "reasoning_tokens": data.reasoning_tokens,
                "content_chars": len(content) if isinstance(content, str) else 0,
                "reasoning_chars": len(reasoning) if isinstance(reasoning, str) else 0,
                "response_source": "content",
            })
            if data.finish_reason == "length":
                record["outcome"] = "truncated"
                raise GenerationAnomaly("truncated",
                    f"模型输出被 max_tokens 截断；finish_reason=length stage={schema_name} model={model} "
                    f"completion_tokens={record['completion_tokens']} "
                    f"reasoning_tokens={record['reasoning_tokens']} "
                    f"content_chars={record['content_chars']} "
                    f"reasoning_chars={record['reasoning_chars']} "
                    f"max_tokens={record['max_tokens']}。")
            if not isinstance(content, str):
                # Null content is allowed by a few OpenAI-compatible APIs.
                content = "" if content is None else content
                if not isinstance(content, str):
                    record["outcome"] = "missing_content"
                    raise GenerationAnomaly("empty", "模型响应缺少 content 字符串")
            if not content.strip():
                # Backward compatible LM Studio quirk. Never trust freeform CoT.
                if "response_format" in payload and isinstance(reasoning, str):
                    try:
                        recovered = json.loads(reasoning.strip())
                    except (ValueError, TypeError):
                        recovered = None
                    if isinstance(recovered, dict):
                        content = reasoning.strip()
                        record["response_source"] = "reasoning_json"
                if not content.strip():
                    record["response_source"] = "none"
                    record["outcome"] = "empty_content"
                    raise GenerationAnomaly("empty", "模型返回空 content，且没有可恢复的完整 schema JSON")
            record["outcome"] = "received"
            return content
        except GenerationAnomaly:
            raise
        except UnsupportedNativeSchema:
            record["outcome"] = "native_schema_unsupported"
            raise
        except (BackendError, ValueError, KeyError, TypeError, IndexError) as exc:
            record["outcome"] = "http_or_protocol_error"
            record["error_type"] = type(exc).__name__
            raise LLMError(f"模型 API 请求失败；stage={schema_name} model={model} "
                           f"error_type={type(exc).__name__}: {exc}") from exc
        finally:
            record["duration_ms"] = round((time.monotonic() - began) * 1000, 1)

    def close(self) -> None:
        if self._owns_verifier_http:
            self.verifier_http.close()
        if self.owns_http:
            self.http.close()

    def _headers(self) -> dict[str, str]:
        # Legacy compatibility; adapter owns actual credentials.
        return {"Authorization": f"Bearer {self.settings.api_key}"} if self.settings.api_key else {}

    def models(self) -> list[str]:
        try:
            values = self.primary_backend.models()
            if values is None:
                raise BackendError("Model discovery not supported")
            return values
        except BackendError as exc:
            raise LLMError(f"无法获取主模型清单：{exc}") from exc

    def validate_model_access(self, primary: str, verifier: str = "") -> str:
        """Preflight without assuming managed APIs expose GET /models."""
        policy = self.settings.model_discovery
        if policy == "off":
            return "skipped"
        cached_discovery: dict[int, list[str] | None] = {}
        for role, backend, model in (("primary", self.primary_backend, primary),
                                     ("verifier", self.verifier_backend, verifier)):
            if not model:
                continue
            try:
                cache_key = id(backend)
                if cache_key in cached_discovery:
                    names = cached_discovery[cache_key]
                else:
                    names = backend.models()
                    cached_discovery[cache_key] = names
                if names is None:
                    raise BackendError("/models unavailable")
            except BackendError as exc:
                if policy == "required":
                    raise LLMError(f"{role} model discovery failed: {exc}") from exc
                cached_discovery[id(backend)] = None
                continue
            if model not in names:
                raise LLMError(f"{role} model {model!r} not in provider's /models result: {names}")
        return "verified_or_optional"

    def choose_model(self, requested: str | None = None) -> str:
        if requested and requested.strip():
            self._runtime_primary_model = requested.strip()
            return self._runtime_primary_model
        if self.settings.model:
            self._runtime_primary_model = self.settings.model
            return self.settings.model
        ids = self.models()
        if not ids:
            raise LLMError("LM Studio 没有可用模型。请先在 LM Studio 中加载模型。")
        if len(ids) != 1:
            raise LLMError("检测到多个模型：请在网页选择模型，或在 .env 中设置 LM_MODEL。")
        self._runtime_primary_model = ids[0]
        return ids[0]

    def choose_verifier(self, requested: str | None = None) -> str:
        value = (requested or self.settings.verifier_model).strip()
        if not value:
            raise LLMError("未配置复核模型：在 .env 设置 LM_VERIFIER_MODEL，或使用 --verifier-model / 网页下拉框。")
        self._runtime_verifier_model = value
        return value

    def structured(self, model: str, system: str, user: str, schema_model: type[BaseModel],
                   schema_name: str, validate_result: Callable[[BaseModel], None] | None = None,
                   max_tokens: int | None = None, max_attempts: int = 3) -> BaseModel:
        """要求模型输出符合 schema 的 JSON；格式错误时至多进行两次修正重试。

        仅修复格式问题；绝不把一条裸 link 自动包装后直接认定 PASS。
        对长度截断或空回答可选一次备用生成，仍严格验证 Schema 和来源。
        """
        if max_attempts not in (1, 2, 3):
            raise ValueError("max_attempts must be 1, 2, or 3")
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens if max_tokens is not None else self.settings.max_tokens,
            "stream": False,
        }
        expected_schema = schema_model.model_json_schema()
        if self._native_enabled(model):
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": expected_schema,
                },
            }

        # Original schema/source feedback policy is unchanged (max 3 attempts).
        # Generation anomalies are a different failure class: perform at most
        # ONE targeted, separately validated recovery call, never identical replay.
        for attempt in range(max_attempts):
            try:
                content = self._completion(payload=payload, model=model,
                    schema_name=schema_name, attempt=attempt + 1, recovery=False)
            except UnsupportedNativeSchema as schema_exc:
                if self.settings.schema_mode != "auto" or "response_format" not in payload:
                    raise LLMError(f"{schema_exc}; configure LM_SCHEMA_MODE=auto or off") from schema_exc
                # Exactly ONE downgrade, only after explicit HTTP schema rejection.
                self._native_support[(self._role_for(model), model)] = False
                payload.pop("response_format", None)
                messages.append({"role": "user", "content": (
                    "The API cannot enforce response_format=json_schema. "
                    "Return only a valid JSON object satisfying this schema, with no prose: "
                    + json.dumps(expected_schema, ensure_ascii=False)
                )})
                try:
                    content = self._completion(payload=payload, model=model,
                        schema_name=schema_name, attempt=attempt + 1, recovery=False)
                except UnsupportedNativeSchema as exc:
                    raise LLMError(f"Provider rejected prompt-only fallback: {exc}") from exc
            except GenerationAnomaly as initial_exc:
                if not self.settings.generation_recovery:
                    raise LLMError(f"{initial_exc}；LM_GENERATION_RECOVERY=false") from initial_exc
                recovery_payload = dict(payload)
                recovery_payload.pop("response_format", None)
                recovery_payload["messages"] = [*messages, {
                    "role": "user",
                    "content": (
                        "The previous generation ended without a complete final JSON answer. "
                        "Return the COMPLETE requested JSON object now, directly and concisely, "
                        "without a reasoning preamble, code fence, or explanatory prose. "
                        "Preserve all required fields and source paragraph IDs."
                    ),
                }]
                try:
                    content = self._completion(payload=recovery_payload, model=model,
                        schema_name=schema_name, attempt=attempt + 1, recovery=True)
                except LLMError as recovery_exc:
                    raise LLMError(
                        f"生成异常恢复失败；original={initial_exc}; recovery={recovery_exc}"
                    ) from recovery_exc
                # One recovery call may not trigger another corrective retry.
                # The normal schema and validate_result checks below still apply.
                try:
                    validated = schema_model.model_validate(parse_json_object(content))
                    if validate_result is not None:
                        validate_result(validated)
                    self.request_trace[-1]["outcome"] = "recovered_validated"
                    return validated
                except (LLMError, ValidationError) as validation_exc:
                    self.request_trace[-1]["outcome"] = "recovery_validation_failed"
                    raise LLMError(
                        f"生成恢复产物未通过原 Schema / 证据校验：{validation_exc}"
                    ) from validation_exc

            try:
                validated = schema_model.model_validate(parse_json_object(content))
                if validate_result is not None:
                    validate_result(validated)
                self.request_trace[-1]["outcome"] = "validated"
                return validated
            except (LLMError, ValidationError) as exc:
                if self.request_trace:
                    self.request_trace[-1]["outcome"] = "schema_or_grounding_invalid"
                    self.request_trace[-1]["validation_error_type"] = type(exc).__name__
                if attempt == max_attempts - 1:
                    raise LLMError(
                        f"stage={schema_name} model={model} attempts={max_attempts}；"
                        f"模型{'连续三次' if max_attempts == 3 else f'在 {max_attempts} 次尝试中'}未能返回符合输出 Schema / 原文证据要求的结果（并非 FAIL）："
                        f"{exc}"
                    ) from exc
                # 不能仅仅把 {en_ids, zh_ids, ...} 包装为 {links:[...]}，
                # 因为原始配对本身可能是错的：要求模型重新核对对应关系。
                messages.append({"role": "assistant", "content": content})
                if schema_name == "keyword_matching":
                    format_hint = (
                        'Top level MUST be {"links":[{"en_ids":["en_1"],'
                        '"zh_ids":["zh_1"],"relation":"equivalent","reason":"..."}]}. '
                        'If none match, return {"links":[]}. '
                        'Reconsider the keyword inventories: do NOT group unrelated '
                        'keywords into one link or fabricate missing counterparts.'
                    )
                elif schema_name in ("keyword_extraction", "keyword_extraction_fallback"):
                    format_hint = (
                        'Top level MUST be {"units":[{"kind":"concept",'
                        '"claim":"...","evidence":"..."}]}. '
                        'evidence must be ONE exact, contiguous substring of original text. '
                        'For a list "image, audio, text samples", quote "image", not "image samples".'
                    )
                elif schema_name == "atomic_refinement":
                    format_hint = (
                        'Return exactly {"units":[{"kind":"enumerated_item",'
                        '"claim":"...","evidence":"..."}]}. '
                        'This is a COMPLETE replacement inventory, not additional missing units. '
                        'Each independently omittable item needs its own unit. '
                        'All evidence must be exact contiguous source substrings (spaces at '
                        'Chinese/ASCII boundaries will be normalized by the caller).'
                    )
                elif schema_name == "coverage_audit":
                    format_hint = (
                        'Top level MUST be {"missing_units":[{"kind":"concept",'
                        '"claim":"...","evidence":"..."}]}. '
                        'Return {"missing_units":[]} if nothing MATERIAL is missing. '
                        'Each evidence MUST be an exact contiguous substring of source. '
                        'Never repeat existing units or invent facts.'
                    )
                elif schema_name == "alignment_evidence_check":
                    format_hint = (
                        'Return exactly {"status":"supported|unsupported|uncertain",'
                        '"en_evidence":"...","zh_evidence":"...","reason":"..."}. '
                        'Supported requires two verbatim ORIGINAL quotes proving the '
                        'SAME disputed fact. If they cannot be found, return unsupported or uncertain.'
                    )
                elif schema_name in ("paragraph_extraction", "paragraph_extraction_fallback"):
                    # Correct the *actual* paragraph wire format. Never send
                    # legacy quote / issue-array instructions to this stage.
                    try:
                        allowed = json.loads(user).get("allowed_paragraph_ids", [])
                    except (ValueError, TypeError):
                        allowed = []
                    format_hint = (
                        'Return exactly one object with "units" (array). '
                        'Each unit must contain only "kind", "claim", and '
                        '"paragraph_ids" (an array of strings). '
                        'Choose paragraph IDs only from the allowed input list: '
                        f'{json.dumps(allowed, ensure_ascii=False)}. '
                        'Never output any evidence string, character offset, '
                        'issues, resolved_missing, or resolved_conflicts. '
                        'Return {"units":[]} only if no meaningful fact exists.'
                    )
                elif schema_name == "paragraph_review":
                    format_hint = (
                        'Return exactly {"verdict":"PASS|FAIL|UNSURE",'
                        '"reason":"...","issues":[]}. '
                        'If FAIL, issues must contain objects with kind, reason, '
                        'en_paragraph_ids and zh_paragraph_ids. '
                        'Choose IDs only from english_paragraphs and chinese_paragraphs. '
                        'Do not output literal evidence quotes.'
                    )
                elif schema_name == "paragraph_arbitration":
                    format_hint = (
                        'Return exactly {"verdict":"PASS|FAIL|UNSURE",'
                        '"issue_kind":"none|missing_in_chinese|missing_in_english|conflict",'
                        '"en_paragraph_ids":[],"zh_paragraph_ids":[],"reason":"..."}. '
                        'For FAIL with conflict cite BOTH actual source paragraph IDs; '
                        'for initial FAIL to PASS cite BOTH original language sources. '
                        'Do not output literal evidence quotes.'
                    )
                elif schema_name == "paragraph_dispute_recheck":
                    format_hint = (
                        'Return exactly {"issues":[{"issue_index":0,'
                        '"status":"refuted|sustained|uncertain",'
                        '"basis":"formal_numeric|other",'
                        '"en_paragraph_ids":[],"zh_paragraph_ids":[],"reason":"..."}],'
                        '"reason":"..."}. '
                        'Assess EVERY reviewer issue index exactly once, using source '
                        'paragraph IDs. A refuted issue must cite BOTH sources. '
                        'Do not return a generic issue list or evidence quotes.'
                    )
                elif schema_name == "paragraph_alignment_support":
                    format_hint = (
                        'Return exactly {"status":"supported|unsupported|uncertain",'
                        '"en_paragraph_ids":[],"zh_paragraph_ids":[],"reason":"..."}. '
                        'Only supported can assert that every disputed fact aligns; '
                        'in that case cite BOTH source paragraph IDs. '
                        'Do not output literal evidence quotes.'
                    )
                elif schema_name == "independent_review":
                    format_hint = (
                        'Return exactly {"verdict":"PASS","reason":"...","issues":[]} '
                        'or FAIL with quoted grounded issues. No ID repairs or resolved arrays.'
                    )
                elif schema_name == "dispute_arbitration":
                    format_hint = (
                        'Return exactly a JSON object with verdict, issue_kind, en_evidence, '
                        'zh_evidence and reason; do not invent quotes.'
                    )
                else:
                    format_hint = (
                        'Top level MUST contain ALL THREE ARRAYS: '
                        '{"issues":[],"resolved_missing":[],"resolved_conflicts":[]}. '
                        'Every nonempty quoted evidence must be an exact substring of original text.'
                    )
                messages.append({
                    "role": "user",
                    "content": (
                        "Your previous response FAILED JSON schema or source-reference validation. "
                        "Regenerate the FULL answer, not merely an outer wrapper. "
                        "Respond with ONE JSON object matching the schema EXACTLY; "
                        "no extra top-level fields, no prose, no code fences. "
                        f"{format_hint} "
                        f"Specific validation error: {str(exc)}. "
                        f"Expected JSON schema: {json.dumps(expected_schema, ensure_ascii=False)}"
                    ),
                })
        raise AssertionError("unreachable")
