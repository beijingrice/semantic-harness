"""Bilingual key fact checker, v0.2.7.

Default: paragraph-ID grounding with model-independent prompts.
Legacy quote mode remains available for controlled A/B comparisons.
Semantic truth is NOT guaranteed by schemas or source IDs.
"""
from __future__ import annotations
import re

from app.config import ConfigError, Settings
from app.evidence import resolve_evidence
from app.llm import ChatClient, LLMError
from app.models import (AlignmentEvidenceDraft, ArbitrationDraft, CheckResult, CoverageAuditDraft, ExtractedDraft, KeyUnit, Link,
                        MatchDraft, ReviewDraft, ReviewIssue)
from app.prompts import (ARBITRATION_SYSTEM, ATOMIC_REFINEMENT_SYSTEM,
                         COVERAGE_AUDIT_SYSTEM, EVIDENCE_SUPPORT_SYSTEM,
                         EXTRACTION_SYSTEM, EXTRACTION_SYSTEM_LEGACY, EXTRACTION_SYSTEM_COARSE,
                         FALLBACK_EXTRACTION_SYSTEM_COARSE,
                         FALLBACK_EXTRACTION_SYSTEM, MATCH_SYSTEM, REVIEW_SYSTEM,
                         arbitrate_user, atomic_refinement_user, coverage_audit_user,
                         evidence_support_user, extract_user,
                         fallback_extract_user, match_user, review_user)
from app.validators import clear_numeric_conflicts, clear_inclusion_conflicts


class PipelineStageError(LLMError):
    """An ERROR whose originating model stage is accessible to batch evaluation."""
    def __init__(self, stage: str, message: str):
        super().__init__(f"[{stage}] {message}")
        self.stage = stage


class Checker:
    # These conservative guards catch *introduced* high-impact modalities in a
    # paraphrased claim. A claim is never treated as more authoritative than its
    # exact source evidence. No cross-language word-by-word translation rules here.
    _CLAIM_FORCE = re.compile(
        r"\b(?:must|shall|mandatory|required|compulsory|obligatory|have to|has to)\b|"
        r"必须|务必|强制|须|应当|应包含|应提交", re.I
    )
    _SOURCE_FORCE = re.compile(
        r"\b(?:must|shall|mandatory|required|compulsory|obligatory|have to|has to|"
        r"should|need|needs|necessary)\b|必须|务必|强制|须|应|需|要求|不得|不可", re.I
    )
    _CLAIM_EXCLUSIVE = re.compile(r"\b(?:only|solely|exclusively)\b|仅|唯一|只允许")
    _SOURCE_EXCLUSIVE = re.compile(r"\b(?:only|solely|exclusively)\b|仅|只|唯一|独", re.I)
    _CLAIM_NEGATION = re.compile(r"\b(?:not|never|without|cannot|prohibited|forbidden|no)\b|"
                                  r"不得|禁止|没有|未|无需|不", re.I)
    _SOURCE_NEGATION = re.compile(r"\b(?:not|never|without|cannot|prohibited|forbidden|no)\b|"
                                   r"不得|禁止|没有|未|无需|不|无|非", re.I)
    _ARABIC_NUMBER = re.compile(r"(?<![a-zA-Z0-9])\d+(?:\.\d+)?(?![a-zA-Z0-9])")

    @classmethod
    def clean_unsupported_claim(cls, source: str, unit, stage: str, events: list[str]) -> None:
        """Prevent unsupported new obligations/exclusivity in coarse claims.

        Do NOT reject an otherwise valid model response or infer a translation.
        For short inputs, a verbatim original-language source is a safe claim.
        For longer inputs, preserve the record but mark it for original-first
        downstream comparison (which must not rely on the suspect paraphrase).
        """
        force = bool(cls._CLAIM_FORCE.search(unit.claim) and
                     not cls._SOURCE_FORCE.search(source))
        exclusive = bool(cls._CLAIM_EXCLUSIVE.search(unit.claim) and
                         not cls._SOURCE_EXCLUSIVE.search(source))
        negative = bool(cls._CLAIM_NEGATION.search(unit.claim) and
                        not cls._SOURCE_NEGATION.search(source))
        # A newly invented numeric literal is high risk; never "correct" a
        # source's numerical content by accepting the model's paraphrase.
        numbers_in_claim = set(cls._ARABIC_NUMBER.findall(unit.claim))
        numbers_in_source = set(cls._ARABIC_NUMBER.findall(source))
        numeric = bool(numbers_in_claim - numbers_in_source)
        if not (force or exclusive or negative or numeric):
            return
        reason = ("unsupported obligation" if force else
                  "unsupported exclusivity" if exclusive else
                  "unsupported negation" if negative else "unsupported numeric detail")
        if len(source) <= 220:
            unit.claim = source
            events.append(f"{stage}: replaced {reason} in claim with verbatim source")
        elif len(unit.evidence) <= 220:
            unit.claim = unit.evidence
            events.append(f"{stage}: replaced {reason} in claim with grounded evidence")
        else:
            events.append(f"{stage}: potential {reason}; downstream must prioritize originals")

    def __init__(self, client: ChatClient, settings: Settings):
        self.client = client
        self.settings = settings
        if settings.evidence_mode not in ("paragraph", "quote"):
            raise ConfigError("LM_EVIDENCE_MODE 只能为 paragraph 或 quote")
        if settings.extraction_prompt not in ("legacy", "proposition", "coarse"):
            raise ConfigError("LM_EXTRACTION_PROMPT 只能为 legacy、proposition 或 coarse")

    @staticmethod
    def coverage_audit_reasons(text: str, units: list[KeyUnit]) -> list[str]:
        """Cheap, conservative TRIGGERS, not judgments of completeness.

        False negatives here are possible and are measured by extraction-only tests.
        An audit itself is always source-only; it cannot decide bilingual PASS/FAIL.
        """
        if not text.strip():
            return []
        reasons: list[str] = []
        if not units:
            return ["nonempty_source_but_zero_units"]

        # Coordinated claims can hide independently omittable list items.
        # One 'and' / '与' already warrants an audit if only one umbrella unit
        # was returned; three-item lists warrant an audit if fewer than 3 units.
        en_connectors = len(re.findall(r"\b(?:and|or)\b", text, re.I))
        # 包括中文的 "，并在"；此前未检测，导致训练/测试角色未拆分。
        zh_connectors = len(re.findall(r"、|以及|并且|，并|和|与|及|同时", text))
        en_commas = text.count(",")
        has_coordination = bool(en_connectors or zh_connectors or en_commas or "；" in text)
        # Heuristic lower-bound for independently omittable list members.
        # There may also be commas between clauses, so a model check is still needed.
        en_min = en_commas + (1 if en_connectors else 0) + 1 if en_commas else 1 + en_connectors
        zh_min = zh_connectors + 1
        estimated_count = max(en_min, zh_min, 2 if "；" in text else 1)
        if has_coordination and len(units) < estimated_count:
            reasons.append("possible_collapsed_enumeration")

        # If extracted units only name datasets without encoding their different
        # TRAIN/TEST or similar semantic roles, a source-only audit is justified.
        role_tokens = re.findall(r"\b(?:train(?:ed|ing)?|test(?:ed|ing)?|evaluat(?:ed|ion|ing)?)\b|训练|测试|评估",
                                 text, re.I)
        unit_claims = " ".join(f"{unit.claim} {unit.evidence}" for unit in units).lower()
        if role_tokens and all(token.lower() not in unit_claims for token in role_tokens):
            reasons.append("possible_missing_action_or_role")

        # If a number in the source does not even occur in a source quote, units
        # may have lost a bound/version. No attempt to validate numerical semantics.
        digits = re.findall(r"(?<![A-Za-z0-9])\d+(?:\.\d+)?%?", text)
        quoted = " ".join(unit.evidence for unit in units)
        if digits and any(value not in quoted for value in digits):
            reasons.append("numeric_text_outside_extracted_evidence")

        # Very short quotes from a long multi-clause source are worth checking.
        if len(text) >= 170:
            covered_chars = [False] * len(text)
            for unit in units:
                begin = text.find(unit.evidence)
                if begin >= 0:
                    for j in range(begin, begin + len(unit.evidence)):
                        covered_chars[j] = True
            if sum(covered_chars) / len(text) < 0.35:
                reasons.append("low_evidence_coverage_for_long_source")
        return reasons

    @staticmethod
    def atomic_refinement_needed(text: str, units: list[KeyUnit]) -> bool:
        """Cheap suspicion only; models still decide whether a list is truly splittable."""
        if not units or not text.strip():
            return False
        if "possible_collapsed_enumeration" not in Checker.coverage_audit_reasons(text, units):
            return False
        # A numerical interval and a comparison are indivisible relations, not lists.
        if re.search(r"\bbetween\b|\bfrom\b.{1,40}\bto\b|介于|范围为|区间|之间", text, re.I):
            return False
        if re.search(r"\b(?:higher|lower|greater|less|more)\s+than\b|高于|低于|优于|劣于", text, re.I):
            return False
        return True

    def _structured(self, stage: str, diagnostics: list[str], model: str,
                    system: str, user: str, schema, schema_name: str,
                    validate_result=None, max_tokens: int | None = None,
                    max_attempts: int | None = None):
        try:
            # Request telemetry names semantic pipeline stages independently of
            # the wire schema (e.g. English vs Chinese extraction).
            self.client._current_stage = stage
            kwargs = {"validate_result": validate_result}
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            if max_attempts is not None:
                kwargs["max_attempts"] = max_attempts
            answer = self.client.structured(model, system, user, schema, schema_name,
                                            **kwargs)
            diagnostics.append(f"{stage}: OK ({model})")
            return answer
        except LLMError as exc:
            raise PipelineStageError(stage, str(exc)) from exc

    def extract(self, text: str, language: str, prefix: str, model: str,
                fallback_model: str | None = None,
                diagnostics: list[str] | None = None) -> list[KeyUnit]:
        """Extract meaning-preserving facts; audit suspicious gaps, not every source.

        Legacy mode exactly preserves the old extraction retry behavior for A/B.
        New mode uses one first pass, a source-only omissions audit only when
        triggered, and a genuinely different fallback for persistently empty output.
        """
        events = diagnostics if diagnostics is not None else []
        if self.settings.evidence_mode == "paragraph":
            from app.paragraph_pipeline import extract_paragraph
            return extract_paragraph(self, text, language, prefix, model,
                                     fallback_model, events)
        stage = f"extract_{language.lower()}"

        def validate(answer: ExtractedDraft) -> None:
            if len(answer.units) > self.settings.max_keywords:
                raise LLMError(f"{language} 提取 {len(answer.units)} 项，超过 MAX_KEYWORDS={self.settings.max_keywords}。请拆分文本。")
            for unit in answer.units:
                grounded = resolve_evidence(text, unit.evidence)
                if grounded is None:
                    raise LLMError(f"{language} 原文证据不是精确连续子串（仅允许中英数字边界空白差异）：{unit.evidence!r}")
                if grounded != unit.evidence:
                    events.append(f"{stage}: normalized quote whitespace to original source")
                    unit.evidence = grounded
                if self.settings.extraction_prompt == "coarse":
                    self.clean_unsupported_claim(text, unit, stage, events)

        def request(system: str, target_model: str = model, fallback: bool = False) -> ExtractedDraft:
            return self._structured(
                stage + ("_fallback" if fallback else ""), events,
                target_model, system,
                fallback_extract_user(text, language, self.settings.max_keywords) if fallback
                else extract_user(text, language, self.settings.max_keywords),
                ExtractedDraft, "keyword_extraction_fallback" if fallback else "keyword_extraction",
                validate_result=validate,
            )

        if self.settings.extraction_prompt == "legacy":
            result = request(EXTRACTION_SYSTEM_LEGACY)
            if not result.units and text.strip():
                events.append(f"{stage}: empty first extract; retrying with explicit instructions")
                result = request(EXTRACTION_SYSTEM_LEGACY + "\nIMPORTANT: The previous extraction was EMPTY. "
                                 "Re-read the original text and extract any real independently "
                                 "testable key facts or obligations; do not invent facts.")
        else:
            system = EXTRACTION_SYSTEM_COARSE if self.settings.extraction_prompt == "coarse" else EXTRACTION_SYSTEM
            result = request(system)
            primary_units = [KeyUnit(id=f"{prefix}_{i}", **u.model_dump())
                             for i, u in enumerate(result.units, 1)]
            refinement_attempted = False
            if (self.settings.extraction_prompt == "proposition" and
                    self.settings.atomic_refinement and
                    self.atomic_refinement_needed(text, primary_units)):
                refinement_attempted = True
                events.append(f"{stage}: atomic refinement triggered")

                def validate_refinement(answer: ExtractedDraft) -> None:
                    validate(answer)
                    # Duplicates and additional umbrella entries are not useful.
                    if len({u.evidence for u in answer.units}) != len(answer.units):
                        raise LLMError("原子化提取含重复证据；请逐项提供独立引用")

                try:
                    refined: ExtractedDraft = self._structured(
                        stage + "_atomic_refinement", events, model,
                        ATOMIC_REFINEMENT_SYSTEM,
                        atomic_refinement_user(text, language,
                                               [u.model_dump() for u in primary_units],
                                               self.settings.max_keywords),
                        ExtractedDraft, "atomic_refinement", validate_result=validate_refinement)
                    if len(refined.units) > len(result.units):
                        result = refined
                        primary_units = [KeyUnit(id=f"{prefix}_{i}", **u.model_dump())
                                         for i, u in enumerate(result.units, 1)]
                        events.append(f"{stage}: atomic refinement replaced umbrella with {len(result.units)} units")
                    else:
                        events.append(f"{stage}: atomic refinement returned no useful split; original retained")
                except PipelineStageError as exc:
                    # Enhancement failure is not a reason to turn an otherwise grounded
                    # extraction into ERROR; retain original and report the limitation.
                    events.append(f"{stage}: optional atomic refinement failed; original retained ({str(exc)[:180]})")

            reasons = self.coverage_audit_reasons(text, primary_units)
            if self.settings.extraction_prompt == "coarse":
                # In a coarse inventory a full, grounded list is a valid unit.
                # Do not waste calls trying to atomize it or treat it as a gap.
                reasons = [r for r in reasons if r != "possible_collapsed_enumeration"]
            # Atomic refinement already inspected coordination. Do not pay for a
            # second, redundant 'missing fact' audit of the same whole-sentence unit.
            if refinement_attempted:
                reasons = [reason for reason in reasons if reason != "possible_collapsed_enumeration"]
            if self.settings.coverage_audit and reasons:
                events.append(f"{stage}: coverage audit triggered ({','.join(reasons)})")

                def validate_audit(audit: CoverageAuditDraft) -> None:
                    if len(audit.missing_units) + len(primary_units) > self.settings.max_keywords:
                        raise LLMError(f"{language} 审计补充后超过 MAX_KEYWORDS，需拆分原文")
                    for extra in audit.missing_units:
                        grounded = resolve_evidence(text, extra.evidence)
                        if grounded is None:
                            raise LLMError(f"审计原文证据不存在：{extra.evidence!r}")
                        if grounded != extra.evidence:
                            events.append(f"{stage}: normalized audit quote to original source")
                            extra.evidence = grounded

                audit: CoverageAuditDraft = self._structured(
                    stage + "_coverage_audit", events, model,
                    COVERAGE_AUDIT_SYSTEM,
                    coverage_audit_user(text, language,
                                        [u.model_dump() for u in primary_units],
                                        reasons, self.settings.max_keywords),
                    CoverageAuditDraft, "coverage_audit", validate_result=validate_audit,
                )
                existing = {(u.kind, u.claim.casefold(), u.evidence) for u in result.units}
                extras = []
                for extra in audit.missing_units:
                    key = (extra.kind, extra.claim.casefold(), extra.evidence)
                    if key not in existing:
                        extras.append(extra)
                        existing.add(key)
                if extras:
                    result = ExtractedDraft(units=[*result.units, *extras])
                events.append(f"{stage}: coverage audit added {len(extras)} unique facts")
            elif reasons and not self.settings.coverage_audit:
                events.append(f"{stage}: audit disabled; triggers={','.join(reasons)}")

        if not result.units and text.strip():
            # Distinct task, never fabricate semantic content or convert empty -> PASS.
            target = fallback_model or model
            events.append(f"{stage}: " +
                          (f"empty after retry; independent fallback model={target}"
                           if self.settings.extraction_prompt == "legacy" else
                           f"no facts after extraction/audit; fallback model={target}"))
            fallback_prompt = (FALLBACK_EXTRACTION_SYSTEM_COARSE
                               if self.settings.extraction_prompt == "coarse"
                               else FALLBACK_EXTRACTION_SYSTEM)
            result = request(fallback_prompt, target_model=target, fallback=True)
            if not result.units:
                events.append(f"{stage}: fallback produced no fact units")
                # Unlike invented claims, an exact original sentence is grounded.
                # This is a last-resort SOURCE inventory item, not a PASS decision.
                # Only for reasonably substantive short coarse-mode inputs; long
                # inputs must fail closed rather than silently dropping content.
                if (self.settings.extraction_prompt == "coarse" and
                        12 <= len(text.strip()) <= 220 and
                        re.search(r"[A-Za-z\u4e00-\u9fff]", text)):
                    result = ExtractedDraft(units=[{
                        "kind": "concept", "claim": text.strip(),
                        "evidence": text.strip(),
                    }])
                    events.append(f"{stage}: source-anchored recovery after two empty inventories; original comparison required")
        return [KeyUnit(id=f"{prefix}_{i}", **u.model_dump()) for i, u in enumerate(result.units, 1)]

    @staticmethod
    def coverage(en: list[KeyUnit], zh: list[KeyUnit], links: list[Link]):
        en_ids, zh_ids = {u.id for u in en}, {u.id for u in zh}
        linked_en: set[str] = set()
        linked_zh: set[str] = set()
        conflicts: list[Link] = []
        for link in links:
            if not set(link.en_ids) <= en_ids or not set(link.zh_ids) <= zh_ids:
                raise LLMError("匹配模型输出了不存在的关键词 ID，不能据此 PASS。")
            if link.relation in ("equivalent", "plausible"):
                linked_en.update(link.en_ids)
                linked_zh.update(link.zh_ids)
            elif link.relation == "conflict":
                conflicts.append(link)
        return ([u for u in en if u.id not in linked_en and
                 not any(u.id in c.en_ids for c in conflicts)],
                [u for u in zh if u.id not in linked_zh and
                 not any(u.id in c.zh_ids for c in conflicts)], conflicts)

    @staticmethod
    def filter_review_issues(issues: list[ReviewIssue], english: str, chinese: str):
        """Discard UNGROUNDED review quotes as suggestions, never auto FAIL or ERROR.

        This is not a semantic truth validator: a correct quote can have wrong reasoning.
        All disagreements still pass through the adjudicator.
        """
        valid: list[ReviewIssue] = []
        warnings: list[str] = []
        for issue in issues:
            if issue.en_evidence:
                grounded_en = resolve_evidence(english, issue.en_evidence)
                if grounded_en is not None:
                    issue.en_evidence = grounded_en
            if issue.zh_evidence:
                grounded_zh = resolve_evidence(chinese, issue.zh_evidence)
                if grounded_zh is not None:
                    issue.zh_evidence = grounded_zh
            if (issue.en_evidence and issue.en_evidence not in english or
                issue.zh_evidence and issue.zh_evidence not in chinese or
                issue.kind == "missing_in_chinese" and not issue.en_evidence or
                issue.kind == "missing_in_english" and not issue.zh_evidence or
                issue.kind == "conflict" and not (issue.en_evidence and issue.zh_evidence)):
                warnings.append(f"忽略无效复核引用：{issue.reason[:100]}")
            else:
                valid.append(issue)
        return valid, warnings

    @staticmethod
    def validate_arbitration(answer: ArbitrationDraft, english: str, chinese: str,
                             initial_verdict: str = "PASS",
                             allow_missing_pass_evidence: bool = False) -> None:
        if answer.en_evidence:
            grounded_en = resolve_evidence(english, answer.en_evidence)
            if grounded_en is None:
                raise LLMError(f"仲裁英文证据不在原文：{answer.en_evidence!r}")
            answer.en_evidence = grounded_en
        if answer.zh_evidence:
            grounded_zh = resolve_evidence(chinese, answer.zh_evidence)
            if grounded_zh is None:
                raise LLMError(f"仲裁中文证据不在原文：{answer.zh_evidence!r}")
            answer.zh_evidence = grounded_zh
        if answer.verdict == "FAIL":
            if answer.issue_kind == "none":
                raise LLMError("仲裁 FAIL 需要可核对的具体问题类型")
            if answer.issue_kind == "missing_in_chinese" and not answer.en_evidence:
                raise LLMError("仲裁中文缺失需要英文原文证据")
            if answer.issue_kind == "missing_in_english" and not answer.zh_evidence:
                raise LLMError("仲裁英文缺失需要中文原文证据")
            if answer.issue_kind == "conflict" and not (answer.en_evidence and answer.zh_evidence):
                raise LLMError("仲裁冲突需要两侧原文证据")
        elif answer.issue_kind != "none":
            raise LLMError("仲裁 PASS/UNSURE 不能同时声明明确问题")
        if (not allow_missing_pass_evidence and answer.verdict == "PASS"
                and initial_verdict == "FAIL" and not (
                    answer.en_evidence and answer.zh_evidence)):
            raise LLMError("仲裁撤销初判 FAIL 时必须分别引用两侧可核对原文")

    @staticmethod
    def validate_alignment_evidence(answer: AlignmentEvidenceDraft, english: str, chinese: str,
                                    en: list[KeyUnit], zh: list[KeyUnit],
                                    missing_zh: list[KeyUnit], missing_en: list[KeyUnit],
                                    conflicts: list[Link]) -> None:
        """Validate original quote grounding AND coverage of disputed prelim units.

        A quote from the right document is NOT automatically semantically true;
        independent evidence model must also explicitly assert supported.
        This guard prevents irrelevant/generic quotes from masking unmatched items.
        """
        if answer.status != "supported":
            return
        if not answer.en_evidence or not answer.zh_evidence:
            raise LLMError("恢复 PASS 证据需要英文和中文两个非空原文片段")
        grounded_en = resolve_evidence(english, answer.en_evidence)
        grounded_zh = resolve_evidence(chinese, answer.zh_evidence)
        if grounded_en is None or grounded_zh is None:
            raise LLMError("恢复 PASS 的证据并非双语原文中的精确连续子串")
        answer.en_evidence = grounded_en
        answer.zh_evidence = grounded_zh

        def intersects(a: str, b: str) -> bool:
            return a in b or b in a

        # Every concrete source-side item which led to FAIL must be represented.
        required_en = {u.id: u.evidence for u in missing_zh}
        required_zh = {u.id: u.evidence for u in missing_en}
        by_en, by_zh = {u.id: u.evidence for u in en}, {u.id: u.evidence for u in zh}
        for conflict in conflicts:
            for unit_id in conflict.en_ids:
                required_en[unit_id] = by_en[unit_id]
            for unit_id in conflict.zh_ids:
                required_zh[unit_id] = by_zh[unit_id]
        if any(not intersects(quote, answer.en_evidence) for quote in required_en.values()):
            raise LLMError("英文证据未覆盖所有初判争议单元，不得恢复 PASS")
        if any(not intersects(quote, answer.zh_evidence) for quote in required_zh.values()):
            raise LLMError("中文证据未覆盖所有初判争议单元，不得恢复 PASS")

    def check(self, english: str, chinese: str, requested_model: str | None = None,
              requested_verifier: str | None = None, mode: str = "dual") -> CheckResult:
        if self.settings.pipeline not in ("extraction", "direct", "direct_grounded", "direct_verified", "direct_questions", "direct_relations", "direct_hardened"):
            raise ConfigError("LM_PIPELINE 只能为 extraction、direct、direct_grounded、direct_verified 、direct_questions 、direct_relations 或 direct_hardened")
        if self.settings.pipeline == "direct_hardened":
            if self.settings.evidence_mode != "paragraph":
                raise ConfigError("Paragraph-Direct 要求 LM_EVIDENCE_MODE=paragraph")
            from app.hardened_direct import check_hardened
            return check_hardened(self, english, chinese, requested_model, requested_verifier, mode)
        if self.settings.pipeline == "direct_relations":
            if self.settings.evidence_mode != "paragraph":
                raise ConfigError("Paragraph-Direct 要求 LM_EVIDENCE_MODE=paragraph")
            from app.relation_direct import check_relations
            return check_relations(self, english, chinese, requested_model, requested_verifier, mode)
        if self.settings.pipeline == "direct_questions":
            if self.settings.evidence_mode != "paragraph":
                raise ConfigError("Paragraph-Direct 要求 LM_EVIDENCE_MODE=paragraph")
            from app.question_direct import check_questions
            return check_questions(self, english, chinese, requested_model, requested_verifier, mode)
        if self.settings.pipeline == "direct_verified":
            if self.settings.evidence_mode != "paragraph":
                raise ConfigError("Paragraph-Direct 要求 LM_EVIDENCE_MODE=paragraph")
            from app.verified_direct import check_verified
            return check_verified(self, english, chinese, requested_model, requested_verifier, mode)
        if self.settings.pipeline == "direct_grounded":
            if self.settings.evidence_mode != "paragraph":
                raise ConfigError("Paragraph-Direct 要求 LM_EVIDENCE_MODE=paragraph")
            from app.grounded_direct import check_grounded
            return check_grounded(self, english, chinese, requested_model, requested_verifier, mode)
        if self.settings.pipeline == "direct":
            if self.settings.evidence_mode != "paragraph":
                raise ConfigError("Paragraph-Direct 要求 LM_EVIDENCE_MODE=paragraph")
            from app.direct_pipeline import check_direct
            return check_direct(self, english, chinese, requested_model, requested_verifier, mode)
        if self.settings.evidence_mode == "paragraph":
            from app.paragraph_pipeline import check_paragraph
            return check_paragraph(self, english, chinese, requested_model,
                                   requested_verifier, mode)
        if mode not in ("primary", "dual"):
            raise LLMError("mode 只能是 primary 或 dual")
        model = self.client.choose_model(requested_model)
        verifier = ""
        if mode == "dual":
            verifier = self.client.choose_verifier(requested_verifier)
            if model == verifier:
                raise LLMError("主模型与复核模型相同：请选择两个不同的模型。")

        diagnostics: list[str] = []
        fallback_model = verifier if mode == "dual" else model
        en = self.extract(english, "English", "en", model, fallback_model, diagnostics)
        zh = self.extract(chinese, "Chinese", "zh", model, fallback_model, diagnostics)
        if not en and not zh:
            raise PipelineStageError("extract_both", "两侧备用抽取后仍无关键事实，无法可靠判定 PASS。")
        matches: MatchDraft = self._structured(
            "keyword_matching", diagnostics, model, MATCH_SYSTEM,
            match_user(english, chinese, [u.model_dump() for u in en], [u.model_dump() for u in zh]),
            MatchDraft, "keyword_matching",
            validate_result=lambda answer: self.coverage(en, zh, answer.links),
        )
        missing_zh, missing_en, conflicts = self.coverage(en, zh, matches.links)
        guard_issues = (clear_numeric_conflicts(en, zh, matches.links)
                        + clear_inclusion_conflicts(en, zh, matches.links))
        initial_verdict = "FAIL" if (missing_zh or missing_en or conflicts or guard_issues) else "PASS"

        review: ReviewDraft | None = None
        arbitration: ArbitrationDraft | None = None
        review_issues: list[ReviewIssue] = []
        review_warnings: list[str] = []
        verdict = initial_verdict

        if mode == "dual":
            review = self._structured(
                "independent_review", diagnostics, verifier, REVIEW_SYSTEM,
                review_user(english, chinese, [u.model_dump() for u in en],
                            [u.model_dump() for u in zh],
                            [link.model_dump() for link in matches.links], initial_verdict),
                ReviewDraft, "independent_review",
            )
            review_issues, review_warnings = self.filter_review_issues(review.issues, english, chinese)
            if review.verdict == "PASS" and review.issues:
                review_warnings.append("复核结论 PASS 却同时提供问题项；以原文和争议仲裁为准")
                diagnostics.append("independent_review: inconsistent PASS with issues; adjudication required")
            if review.verdict == "FAIL" and not review_issues:
                diagnostics.append("independent_review: FAIL without grounded issues; do not treat reason as proof")
            # A reviewer cannot veto an initial PASS. A reviewer also cannot unilaterally
            # rescue an initial FAIL. Disagreement invokes a FOCUSED source-grounded judge.
            disagreement = (review.verdict != initial_verdict or
                            initial_verdict == "PASS" and bool(review.issues))
            if disagreement and not (guard_issues and initial_verdict == "FAIL"):
                dispute_snapshot = {
                    "missing_in_chinese": [u.model_dump() for u in missing_zh],
                    "missing_in_english": [u.model_dump() for u in missing_en],
                    "conflicts": [c.model_dump() for c in conflicts],
                    "numeric_guard_issues": guard_issues,
                }
                arbitration = self._structured(
                    "dispute_arbitration", diagnostics, model, ARBITRATION_SYSTEM,
                    arbitrate_user(english, chinese, initial_verdict, dispute_snapshot,
                                   review.model_dump()),
                    ArbitrationDraft, "dispute_arbitration",
                    # Recover missing PASS evidence in a SEPARATE, focused step.
                    # Still reject nonexistent evidence and inconsistent issue types.
                    validate_result=lambda answer: self.validate_arbitration(
                        answer, english, chinese, initial_verdict,
                        allow_missing_pass_evidence=True),
                )
                if (initial_verdict == "FAIL" and arbitration.verdict == "PASS"
                        and not (arbitration.en_evidence and arbitration.zh_evidence)):
                    # Both original quotes are required. A second MODEL checks the
                    # semantic assertion and supplies source evidence; Python only
                    # validates actual quotes and coverage, never manufactures them.
                    evidence = self._structured(
                        "alignment_evidence_check", diagnostics, verifier,
                        EVIDENCE_SUPPORT_SYSTEM,
                        evidence_support_user(english, chinese, dispute_snapshot,
                                              review.model_dump(), arbitration.model_dump()),
                        AlignmentEvidenceDraft, "alignment_evidence_check",
                        validate_result=lambda answer: self.validate_alignment_evidence(
                            answer, english, chinese, en, zh, missing_zh, missing_en,
                            conflicts),
                    )
                    if evidence.status != "supported":
                        raise PipelineStageError(
                            "alignment_evidence_check",
                            "双模型无法提供足够证据恢复 PASS（"
                            + evidence.status + "）：" + evidence.reason,
                        )
                    arbitration = arbitration.model_copy(update={
                        "en_evidence": evidence.en_evidence,
                        "zh_evidence": evidence.zh_evidence,
                        "reason": (arbitration.reason + " | 独立证据复核：" + evidence.reason)[:600],
                    })
                    # Keep the original strict acceptance condition.
                    try:
                        self.validate_arbitration(arbitration, english, chinese, initial_verdict)
                    except LLMError as exc:
                        raise PipelineStageError("alignment_evidence_check", str(exc)) from exc
                    diagnostics.append("FAIL→PASS proposal: independently supported by two source quotes")
                if initial_verdict == "FAIL" and arbitration.verdict == "PASS":
                    # Applies also when Qwen supplied both quotes directly.
                    try:
                        self.validate_alignment_evidence(
                            AlignmentEvidenceDraft(
                            status="supported", en_evidence=arbitration.en_evidence,
                            zh_evidence=arbitration.zh_evidence, reason=arbitration.reason,
                            ), english, chinese, en, zh, missing_zh, missing_en, conflicts,
                        )
                    except LLMError as exc:
                        raise PipelineStageError("dispute_arbitration", str(exc)) from exc
                if arbitration.verdict == "UNSURE":
                    if initial_verdict == "PASS":
                        raise LLMError("两模型存在实质异议且仲裁不确定，不能自动 PASS。请人工复核。")
                    verdict = "FAIL"  # An unresolved initial FAIL remains flagged.
                else:
                    verdict = arbitration.verdict
            # Uncertain second opinion without consensus is handled as a disagreement.
            if guard_issues:
                verdict = "FAIL"  # A narrowly proven quantity mismatch cannot be overruled.

        summary = (f"{verdict}：主模型初判 {initial_verdict}；"
                   + ("未启用独立复核。" if mode == "primary" else
                      f"{verifier} 复核 {review.verdict}；"
                      f"{'已执行争议仲裁，结论 ' + arbitration.verdict if arbitration else '无需争议仲裁'}。")
                   + f" 初步缺失 {len(missing_zh) + len(missing_en)} 项、匹配冲突 {len(conflicts)} 项、"
                   f"确定性约束保护 {len(guard_issues)} 项。")
        return CheckResult(
            verdict=verdict, model=model, verifier_model=verifier, mode=mode,
            english_keywords=en, chinese_keywords=zh, links=matches.links,
            missing_in_chinese=missing_zh if verdict == "FAIL" else [],
            missing_in_english=missing_en if verdict == "FAIL" else [],
            conflicts=conflicts if verdict == "FAIL" else [],
            preliminary_missing_in_chinese=missing_zh,
            preliminary_missing_in_english=missing_en, preliminary_conflicts=conflicts,
            review=review, review_issues=review_issues,
            arbitration=arbitration, arbitrated=arbitration is not None,
            review_warnings=review_warnings, deterministic_issues=guard_issues,
            initial_verdict=initial_verdict, summary=summary, diagnostics=diagnostics,
        )
