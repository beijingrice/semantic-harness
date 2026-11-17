"""v0.2.7 contract tests. Simulated responses, NOT live-model accuracy."""
import json
from dataclasses import replace

import pytest

from app.config import Settings
from app.models import (
    MatchDraft,
    ParagraphAlignmentDraft,
    ParagraphArbitrationDraft,
    ParagraphDisputeRecheckDraft,
    ParagraphExtractedDraft,
    ParagraphReviewDraft,
)
from app.paragraphs import build_paragraphs, paragraph_evidence, verify_ids
from app.pipeline import Checker, PipelineStageError


def config(**kw):
    return replace(Settings(evidence_mode="paragraph", extraction_prompt="coarse",
                            coverage_audit=False, atomic_refinement=False,
                            extraction_fast_path=False, model="primary", verifier_model="independent"), **kw)


def extract(claim, pid, kind="constraint"):
    return ParagraphExtractedDraft.model_validate({"units":[{"kind":kind,
        "claim":claim,"paragraph_ids":list(pid) if isinstance(pid, tuple) else [pid]}]})


def match(relation="equivalent", reason="same"):
    return MatchDraft.model_validate({"links":[{"en_ids":["en_1"],
        "zh_ids":["zh_1"],"relation":relation,"reason":reason}]})


def review(verdict="PASS", issues=None):
    return ParagraphReviewDraft.model_validate({"verdict":verdict,
        "reason":"independently inspected original texts", "issues":issues or []})


def arb(verdict="FAIL", kind="conflict", en=None, zh=None):
    return ParagraphArbitrationDraft.model_validate({"verdict":verdict,
        "issue_kind":kind,"en_paragraph_ids":en or [],"zh_paragraph_ids":zh or [],
        "reason":"independently checked constraints"})


class Fake:
    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []
    def choose_model(self, requested=None):
        return requested or "primary"
    def choose_verifier(self, requested=None):
        return requested or "independent"
    def structured(self, model, system, user, schema_model, schema_name, validate_result=None):
        self.calls.append({"model":model,"system":system,"user":json.loads(user),
                           "schema":schema_name})
        answer = self.answers.pop(0)
        assert isinstance(answer, schema_model), (type(answer), schema_model)
        if validate_result:
            validate_result(answer)
        return answer


def test_unicode_punctuation_and_whitespace_never_retyped():
    text="Respondent’s fields are required.\nThe report includes 结果、图表。"
    spans = build_paragraphs(text, "en")
    assert len(spans)==1
    assert paragraph_evidence(["en_p1"],spans)==text
    assert spans[0].text==text
    assert spans[0].start==0 and spans[0].end==len(text)


def test_blank_lines_split_only_and_preserve_offsets():
    text="  第一段、包括图表。\n接着说明。\n\n\n  第二段含 42天。  \n\nThird's «quote». "
    spans = build_paragraphs(text, "zh")
    assert len(spans)==3
    assert [s.id for s in spans]==["zh_p1","zh_p2","zh_p3"]
    assert spans[0].text=="第一段、包括图表。\n接着说明。"
    for p in spans:
        assert text[p.start:p.end]==p.text
    assert paragraph_evidence(["zh_p3","zh_p1"],spans)==spans[2].text+"\n\n"+spans[0].text


def test_invalid_span_id_is_never_fabricated():
    spans=build_paragraphs("Original must not be altered.","en")
    with pytest.raises(ValueError,match="不存在"):
        verify_ids(["en_p9"],spans)
    with pytest.raises(ValueError,match="重复"):
        verify_ids(["en_p1","en_p1"],spans)


def test_paragraph_primary_complete_pipeline_keeps_original_evidence():
    english="The cover page is included within 30 pages."
    chinese="封面计入30页。"
    fake=Fake(extract("The cover page is included within 30 pages.","en_p1"),
              extract("封面计入30页。","zh_p1"),match())
    out=Checker(fake,config()).check(english,chinese,mode="primary")
    assert out.verdict=="PASS" and out.evidence_mode=="paragraph"
    assert out.english_keywords[0].evidence==english
    assert out.chinese_keywords[0].evidence==chinese
    assert out.english_keywords[0].paragraph_ids==["en_p1"]
    assert [c['schema'] for c in fake.calls]==[
        'paragraph_extraction','paragraph_extraction','keyword_matching']
    assert fake.calls[0]['user']['source_paragraphs'][0]['text']==english
    assert 'evidence' not in fake.calls[0]['user']['source_paragraphs'][0]
    assert not fake.answers


def test_two_paragraphs_one_cross_paragraph_fact():
    en="The appendix is submitted separately.\n\nIt must remain accessible for six years."
    zh="附录单独提交。\n\n该附录须保留六年供查阅。"
    fake=Fake(extract("The appendix is submitted separately and remains accessible for six years.",
                       ("en_p1","en_p2")),
              extract("附录单独提交且须保留六年供查阅",("zh_p1","zh_p2")),match())
    out=Checker(fake,config()).check(en,zh,mode="primary")
    assert out.verdict=="PASS"
    assert len(out.english_paragraphs)==2
    assert out.english_keywords[0].paragraph_ids==['en_p1','en_p2']
    assert out.english_keywords[0].evidence==en


def test_reviewer_reads_only_original_paragraphs_not_primary_claims():
    en="The appendix is excluded from page count."
    zh="附录不计入页数。"
    fake=Fake(extract(en,"en_p1"),extract(zh,"zh_p1"),match(),review())
    out=Checker(fake,config()).check(en,zh,mode="dual")
    assert out.verdict=="PASS"
    user=fake.calls[3]['user']
    assert set(user)=={'english_text','chinese_text','english_paragraphs','chinese_paragraphs'}
    assert 'english_units' not in user and 'initial_verdict' not in user
    assert not fake.answers


def test_reviewer_real_conflict_cannot_be_auto_overridden_with_pass():
    # Use a semantic contradiction outside the deliberately narrow deterministic
    # guards: reviewer disagreement must still invoke protected arbitration.
    en="The original file must be archived."
    zh="原始文件可以删除。"
    issue={"kind":"conflict","en_paragraph_ids":["en_p1"],
           "zh_paragraph_ids":["zh_p1"],"reason":"archive versus delete"}
    recheck = ParagraphDisputeRecheckDraft.model_validate({
        "issues": [{"issue_index": 0, "status": "refuted", "basis": "other", "reason": "could be false",
                    "en_paragraph_ids": ["en_p1"], "zh_paragraph_ids": ["zh_p1"]}],
        "reason": "second model said refuted but no formal proof exists"})
    fake=Fake(extract(en,"en_p1"),extract(zh,"zh_p1"),match(),
        review("FAIL",[issue]), arb("PASS",kind="none"), recheck)
    result = Checker(fake,config()).check(en,zh,mode="dual")
    assert result.verdict == "REVIEW_REQUIRED"
    assert result.dispute_recheck.issues[0].status == "refuted"
    assert len(fake.calls)==6
    assert fake.calls[-1]["user"]["verified_formal_equivalences"] == []


def test_invalid_reviewer_id_is_rejected():
    issue={"kind":"conflict","en_paragraph_ids":["en_p999"],
           "zh_paragraph_ids":["zh_p1"],"reason":"conflicting"}
    fake=Fake(extract('The cover page is included.',"en_p1"),
              extract('封面不计入。',"zh_p1"),match(),review("FAIL",[issue]))
    with pytest.raises(PipelineStageError,match="Paragraph ID"):
        Checker(fake,config()).check("The cover page is included.","封面不计入。",mode="dual")


def test_initial_fail_recovery_needs_independent_semantic_support():
    en="The abstract contains the objective and methods."
    zh="摘要介绍了目标与方法。"
    support=ParagraphAlignmentDraft.model_validate({
        'status':'supported','en_paragraph_ids':['en_p1'],
        'zh_paragraph_ids':['zh_p1'], 'reason':'same two components'})
    fake=Fake(extract(en,"en_p1"),extract(zh,"zh_p1"),
              match("conflict","over-strict descriptive modal"),review("PASS"),
              arb("PASS",kind="none",en=['en_p1'],zh=['zh_p1']),support)
    out=Checker(fake,config()).check(en,zh,mode="dual")
    assert out.verdict=="PASS" and out.arbitrated
    assert len(fake.calls)==6
    assert fake.calls[-1]['model']=='independent'


def test_initial_fail_recovery_rejected_by_independent_support():
    en="The cover page is included."
    zh="封面不计入。"
    support=ParagraphAlignmentDraft.model_validate({
        'status':'unsupported','reason':'cover page inclusion conflicts'})
    fake=Fake(extract(en,"en_p1"),extract(zh,"zh_p1"),match("conflict"),
        review("PASS"), arb("PASS",kind="none",en=['en_p1'],zh=['zh_p1']),support)
    with pytest.raises(PipelineStageError,match="无法支持 FAIL→PASS"):
        Checker(fake,config()).check(en,zh,mode="dual")


def test_original_source_span_includes_mixed_unicode_without_normalization():
    source="参与者’s data \u2014 only under consent.\n\n\u201cQuotes\u201d: keep 10–20 units."
    s=build_paragraphs(source,"zh")
    assert ''.join(p.text for p in s).replace('\n','') in source.replace('\n','') or len(s)==2
    assert all(source[p.start:p.end] == p.text for p in s)
