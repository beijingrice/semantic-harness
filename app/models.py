"""The language-model wire format and the public API data contract.

Every model-produced quote or paragraph ID is checked against the ORIGINAL source text by the pipelines.
Schema validity is not evidence of semantic correctness.
"""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class UnitDraft(StrictModel):
    kind: Literal["concept", "enumerated_item", "entity", "constraint", "relation"]
    claim: str = Field(min_length=1, max_length=220)
    evidence: str = Field(min_length=1, max_length=6000)

    @field_validator("claim", "evidence")
    @classmethod
    def trim(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("empty claim/evidence")
        return value.strip()


class ExtractedDraft(StrictModel):
    units: list[UnitDraft]


class CoverageAuditDraft(StrictModel):
    """Additional source-only facts; NEVER a judgment of bilingual equivalence."""
    missing_units: list[UnitDraft]


class KeyUnit(UnitDraft):
    id: str
    paragraph_ids: list[str] = Field(default_factory=list)


class SourceParagraph(StrictModel):
    id: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    text: str = Field(min_length=1)


class ParagraphUnitDraft(StrictModel):
    kind: Literal["concept", "enumerated_item", "entity", "constraint", "relation"]
    claim: str = Field(min_length=1, max_length=220)
    paragraph_ids: list[str] = Field(min_length=1, max_length=12)


class ParagraphExtractedDraft(StrictModel):
    units: list[ParagraphUnitDraft]


class ParagraphReviewIssue(StrictModel):
    kind: Literal["missing_in_chinese", "missing_in_english", "conflict"]
    en_paragraph_ids: list[str] = Field(default_factory=list)
    zh_paragraph_ids: list[str] = Field(default_factory=list)
    reason: str = Field(min_length=1, max_length=600)


class ParagraphReviewDraft(StrictModel):
    verdict: Literal["PASS", "FAIL", "UNSURE"]
    reason: str = Field(min_length=1, max_length=600)
    issues: list[ParagraphReviewIssue] = Field(default_factory=list)


class ParagraphArbitrationDraft(StrictModel):
    verdict: Literal["PASS", "FAIL", "UNSURE"]
    issue_kind: Literal["none", "missing_in_chinese", "missing_in_english", "conflict"] = "none"
    en_paragraph_ids: list[str] = Field(default_factory=list)
    zh_paragraph_ids: list[str] = Field(default_factory=list)
    reason: str = Field(min_length=1, max_length=600)


class ParagraphAlignmentDraft(StrictModel):
    status: Literal["supported", "unsupported", "uncertain"]
    en_paragraph_ids: list[str] = Field(default_factory=list)
    zh_paragraph_ids: list[str] = Field(default_factory=list)
    reason: str = Field(min_length=1, max_length=600)



class NumericEquivalenceCertificate(StrictModel):
    """Python-checked equality of a single normalized quantity in *original* passages."""
    en_unit_id: str
    zh_unit_id: str
    en_paragraph_ids: list[str]
    zh_paragraph_ids: list[str]
    dimension: str
    canonical_value: str
    comparator: Literal["<", "<=", ">", ">=", "="]
    en_claim: str
    zh_claim: str


class ParagraphIssueRecheck(StrictModel):
    issue_index: int = Field(ge=0)
    status: Literal["refuted", "sustained", "uncertain"]
    basis: Literal["formal_numeric", "other"]
    en_paragraph_ids: list[str] = Field(default_factory=list)
    zh_paragraph_ids: list[str] = Field(default_factory=list)
    reason: str = Field(min_length=1, max_length=600)


class ParagraphDisputeRecheckDraft(StrictModel):
    issues: list[ParagraphIssueRecheck]
    reason: str = Field(min_length=1, max_length=600)


class Link(StrictModel):
    en_ids: list[str] = Field(min_length=1)
    zh_ids: list[str] = Field(min_length=1)
    relation: Literal["equivalent", "plausible", "related_only", "conflict"]
    reason: str = Field(min_length=1, max_length=600)

    @model_validator(mode="after")
    def unique_ids(self):
        if len(set(self.en_ids)) != len(self.en_ids) or len(set(self.zh_ids)) != len(self.zh_ids):
            raise ValueError("repeated link ID")
        return self


class MatchDraft(StrictModel):
    links: list[Link]


class ReviewIssue(StrictModel):
    kind: Literal["missing_in_chinese", "missing_in_english", "conflict"]
    en_evidence: str = Field(max_length=6000)
    zh_evidence: str = Field(max_length=6000)
    reason: str = Field(min_length=1, max_length=600)
    en_paragraph_ids: list[str] = Field(default_factory=list)
    zh_paragraph_ids: list[str] = Field(default_factory=list)


class ReviewDraft(StrictModel):
    """Second model gives one overall opinion, not commands to mutate the first result."""
    verdict: Literal["PASS", "FAIL", "UNSURE"]
    reason: str = Field(min_length=1, max_length=600)
    issues: list[ReviewIssue] = Field(default_factory=list)


class ArbitrationDraft(StrictModel):
    en_paragraph_ids: list[str] = Field(default_factory=list)
    zh_paragraph_ids: list[str] = Field(default_factory=list)
    """Focused judge of a disagreement, with independently checked source citations."""
    verdict: Literal["PASS", "FAIL", "UNSURE"]
    issue_kind: Literal["none", "missing_in_chinese", "missing_in_english", "conflict"] = "none"
    en_evidence: str = Field(default="", max_length=6000)
    zh_evidence: str = Field(default="", max_length=6000)
    reason: str = Field(min_length=1, max_length=600)


class AlignmentEvidenceDraft(StrictModel):
    """Targeted second-model check that a purported FAIL->PASS has source support.

    Exact quotes are necessary but not sufficient for truth; this is still an LLM
    semantic judgement and will be evaluated on held-out cases.
    """
    status: Literal["supported", "unsupported", "uncertain"]
    en_evidence: str = Field(default="", max_length=6000)
    zh_evidence: str = Field(default="", max_length=6000)
    reason: str = Field(min_length=1, max_length=600)


class GroundedEvidenceDraft(StrictModel):
    """Focused recheck; exact original excerpts, never free-form evidence text.

    The audit is a semantic hypothesis; verbatim quotes are source-grounding,
    not formal evidence that the proposed interpretation is correct.
    """
    status: Literal["CONFLICT", "EQUIVALENT", "UNCERTAIN"]
    en_quote: str = Field(default="", max_length=600)
    zh_quote: str = Field(default="", max_length=600)
    reason: str = Field(min_length=1, max_length=600)


class FocusedVerificationDraft(StrictModel):
    # Produced only by pipelines removed in v0.3.0; CheckResult.focused_verification is always null.
    """A structured hypothesis about one dispute, with source-backed option IDs.

    Only the host chooses the actual quote text; these model fields are NOT a
    proof that the selected evidence entails the stated difference.
    """
    status: Literal["CONFLICT", "EQUIVALENT", "UNCERTAIN"]
    difference_type: Literal[
        "numeric_boundary", "inclusion_exclusion", "exception_scope",
        "direction_reversal", "extra_requirement", "equivalent_restatement",
        "other", "undetermined"]
    same_subject: Literal["yes", "no", "unknown"]
    en_evidence_id: str = Field(min_length=1, max_length=32)
    zh_evidence_id: str = Field(min_length=1, max_length=32)
    reason: str = Field(min_length=1, max_length=600)


class CheckRequest(StrictModel):
    english: str = Field(min_length=1, max_length=6000)
    chinese: str = Field(min_length=1, max_length=6000)
    model: str | None = Field(default=None, max_length=250)
    verifier_model: str | None = Field(default=None, max_length=250)
    mode: Literal["primary", "dual"] = "dual"

    @field_validator("english", "chinese")
    @classmethod
    def check_nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("文本不能为空")
        return value.strip()


class RelationVerificationDraft(StrictModel):
    """direct_hardened answer. same_event is scope identity, NOT whether two rules agree.

    Extracted actor/entity fields are suggestions checked against immutable source
    spans; cannot independently prove translation or imply a label.
    """
    task_id: str = Field(min_length=1, max_length=60)
    question_type: Literal["numeric_boundary", "inclusion_exclusion", "exception_scope",
                           "direction_reversal", "scalar_equivalence", "undetermined"]
    same_event: Literal["yes", "no", "unknown"]
    en_evidence_id: str = Field(min_length=1, max_length=30)
    zh_evidence_id: str = Field(min_length=1, max_length=30)
    en_subject: str = Field(default="", max_length=130)
    zh_subject: str = Field(default="", max_length=130)
    en_condition: str = Field(default="", max_length=130)
    zh_condition: str = Field(default="", max_length=130)
    en_value: str = Field(max_length=150)
    zh_value: str = Field(max_length=150)
    finding: Literal["CONFLICT", "EQUIVALENT", "UNCERTAIN"]
    reason: str = Field(min_length=1, max_length=600)


class CheckResult(StrictModel):
    english_paragraphs: list[SourceParagraph] = Field(default_factory=list)
    chinese_paragraphs: list[SourceParagraph] = Field(default_factory=list)
    evidence_mode: Literal["paragraph", "quote"] = "quote"
    verdict: Literal["PASS", "FAIL", "REVIEW_REQUIRED"]
    model: str
    verifier_model: str
    english_keywords: list[KeyUnit]
    chinese_keywords: list[KeyUnit]
    links: list[Link]
    missing_in_chinese: list[KeyUnit]
    missing_in_english: list[KeyUnit]
    conflicts: list[Link]
    preliminary_missing_in_chinese: list[KeyUnit] = Field(default_factory=list)
    preliminary_missing_in_english: list[KeyUnit] = Field(default_factory=list)
    preliminary_conflicts: list[Link] = Field(default_factory=list)
    review: ReviewDraft | None
    arbitration: ArbitrationDraft | None = None
    arbitrated: bool = False
    dispute_recheck: ParagraphDisputeRecheckDraft | None = None
    review_issues: list[ReviewIssue]
    mode: Literal["primary", "dual"] = "dual"
    review_warnings: list[str] = Field(default_factory=list)
    deterministic_issues: list[str]
    initial_verdict: Literal["PASS", "FAIL"]
    summary: str
    diagnostics: list[str] = Field(default_factory=list)
    # Direct-only raw Qwen independent judgment; absent for extraction.
    direct_primary: ParagraphReviewDraft | None = None
    grounded_evidence_audit: GroundedEvidenceDraft | None = None
    focused_verification: FocusedVerificationDraft | None = None
    verification_questions: list[dict[str, str]] = Field(default_factory=list)
    grounded_risks: list[str] = Field(default_factory=list)
