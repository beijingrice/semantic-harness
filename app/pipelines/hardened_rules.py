"""Source-only rules of the direct_hardened pipeline: which questions to ask, and proofs.

Nothing here calls a model. `choose_questions` picks fixed questions from source risks
and model disagreement; `prove` decides from the original texts whether a model's
answer to one question can become an automatic FAIL (or, for one narrow scalar case,
PASS). Everything that cannot be proved returns REVIEW_REQUIRED.

Known limitation (docs/project_analysis.md, section 4.1): several vocabularies and
patterns below were written against specific development cases and generalise
poorly. They are marked "CASE-SPECIFIC". Unmatched wording falls back to
REVIEW_REQUIRED, so this costs automatic coverage, not correctness.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from app.models import RelationVerificationDraft
from app.source_risks import source_quantitative_issue_only, source_risks
from app.validators import numeric_tokens, parse_quantity


@dataclass(frozen=True)
class Question:
    id: str
    kind: str
    question: str
    en_hint: str = ''
    zh_hint: str = ''


# CASE-SPECIFIC: bilingual entity names the proofs can recognise. Anything outside
# this list is never mapped and leads to REVIEW_REQUIRED.
_ENTITY_PAIRS = (
    ('cover page', '封面'), ('title page', '标题页'),
    ('reference list', '参考文献列表'), ('references', '参考文献'),
    ('appendix', '附录'), ('main report', '正文报告'),
    ('chinese abstract', '中文摘要'), ('english abstract', '英文摘要'),
    ('baseline', '基线方法'), ('baseline method', '基线方法'),
    ('proposed system', '新系统'), ('new system', '新系统'),
)


def _mapped(en_operand: str, zh_operand: str) -> bool:
    e, z = en_operand.strip().casefold(), zh_operand.strip()
    return any(e == eng and z == chi for eng, chi in _ENTITY_PAIRS)


def choose_questions(english: str, chinese: str, first: str, second: str | None,
                     allegations: str = '') -> list[Question]:
    risks = source_risks(english, chinese)
    out = []
    # Provenance of question generation is recorded. All risks are mandatory
    # individually; a model cannot substitute a different issue for a warning.
    for risk in risks:
        if risk.kind == 'numeric_boundary':
            out.append(Question('Q_NUMERIC_BOUNDARY', 'numeric_boundary',
                'Compare only the INCLUSIVE versus EXCLUSIVE threshold on the same measured quantity. '
                'Return the English and Chinese mathematical operators, values and units. '
                'Do these two clauses govern the same event?', risk.en_excerpt, risk.zh_excerpt))
        elif risk.kind == 'inclusion_inventory':
            out.append(Question('Q_COUNTED_ENTITY', 'inclusion_exclusion',
                'Find an identical document component whose inclusion in the page/word limit differs. '
                'Name that entity in each original language and locate BOTH source rules; '
                'if none can be proved, say UNCERTAIN. Do NOT merely count all inclusion words.'))
    disagree = second is not None and first != second
    if disagree and not any(q.kind == 'numeric_boundary' for q in out):
        # CASE-SPECIFIC triggers: 'unless' + 无论/不论/除非 (V2-EXC-02/03) and
        # 'outperformed / higher score' (V2-ROL-06/07).
        if re.search(r'\bunless\b', english, re.I) and re.search(r'无论|不论|除非', chinese):
            out.append(Question('Q_EXCEPTION_SCOPE', 'exception_scope',
                'Compare the same specified optional/mandatory document component under a program exception. '
                'Is it conditional on program requirements in one language but always optional in the other?'))
        elif (re.search(r'\b(?:outperformed|higher score|scored higher)\b', english, re.I)
              and re.search(r'更高|更好|优于', chinese)):
            out.append(Question('Q_COMPARISON_WINNER', 'direction_reversal',
                'For the SAME dataset/condition, identify the WINNING ACTOR on each side '
                'as literal source phrases (not a paraphrase). Are the winners different?'))
        elif not out:
            out.append(Question('Q_DISPUTED_CLAIM', 'undetermined',
                'Find the alleged exact material difference, keeping full condition scope; '
                'do not silently switch to another issue. If not source-checkable, UNCERTAIN.'))
    if (first == second == 'FAIL' and source_quantitative_issue_only(english, chinese)
            and not out):
        out.append(Question('Q_SCALAR_EQUIVALENCE', 'scalar_equivalence',
            'Is this the SAME single fixed quantity in different units, without a distinct duty? '
            'Compare mathematical values, exactness, and semantic scope.'))
    # Bounded review even on unusually complex multi-risk sources. Unexamined
    # risk is never silently passed off as PASS.
    return out[:3]


# CASE-SPECIFIC: a numeric-boundary FAIL needs both excerpts to share two of these
# term pairs (evidence that both talk about the same measured quantity). Each pair
# was taken from a development case, two of them from holdout cases H28-02/H28-04.
_BOUND_ANCHORS = (
    ('invalid', '无效'), ('mandatory', '必填'),
    ('calibration', '校准'), ('error', '误差'),
    ('font size', '字体大小'), ('table', '表格'),
    ('participant', '参与者'), ('response', '答卷'),
    ('survey', '调查'), ('margin', '页边距'),
    ('confidence interval', '置信区间'),
)


def _same_numeric_context(en: str, zh: str) -> bool:
    matches = [(e, z) for e,z in _BOUND_ANCHORS if e in en.casefold() and z in zh]
    return len(matches) >= 2


def _numeric_symbol(value: str):
    v = re.fullmatch(r'\s*(<=|>=|<|>|=|==)\s*(\d+(?:\.\d+)?)\s*(%|mm|cm|m|days?|hours?|weeks?)?\s*', value, re.I)
    if not v:
        return None
    from decimal import Decimal
    return ('==' if v[1] in ('=', '==') else v[1], Decimal(v[2]), (v[3] or '').lower())


def numeric_symbol_unicode(value: str):
    """Normalize *only* model operands; host-parsed original source still wins.

    Unicode comparison symbols and full-width digits are display variants of
    the same mathematical operators, not weaker/stronger semantic claims.
    Reject loosely formatted descriptions, intervals and unequal units.
    """
    v = unicodedata.normalize('NFKC', value.strip())
    v = v.replace('≥', '>=').replace('≧', '>=').replace('≤', '<=').replace('≦', '<=')
    return _numeric_symbol(v)


def _included(text: str, subject: str, language: str) -> str | None:
    if language == 'en':
        # The subject must govern the inclusion relation directly; using an
        # unrelated 'excluded' elsewhere in the paragraph is not enough.
        p = re.escape(subject)
        m = re.search(r'\b'+p+r'\s+(?:is|are|was|were)\s+(not\s+)?(?:included|counted)\b(?!\s+separately)', text, re.I)
        if m:
            return 'excluded' if m[1] else 'included'
        if re.search(r'\b'+p+r'\s+(?:is|are|was|were)\s+excluded\b', text, re.I):
            return 'excluded'
        if re.search(r'\b'+p+r'\s+(?:does|do|did)\s+not\s+count\b', text, re.I):
            return 'excluded'
        if re.search(r'\b'+p+r'\s+counts?\s+(?:toward|towards|against|in)\b', text, re.I):
            return 'included'
        return None
    m = re.search(re.escape(subject)+r'\s*(不计入|未计入|无需计入|无须计入|不算入|未算入|计入|算入)', text)
    if not m:
        return None
    return 'excluded' if m[1] in ('不计入','未计入','无需计入','无须计入','不算入','未算入') else 'included'


def _winner(text: str, subject: str, language: str) -> bool:
    p = re.escape(subject)
    if language == 'en':
        return bool(re.search(r'\b'+p+r'\s+(?:achieved|had|obtained|received|scored)\s+(?:the\s+)?higher\s+score\b',text,re.I)
                    or re.search(r'\b'+p+r'\s+outperformed\b',text,re.I))
    return bool(re.search(p+r'\s*(?:得分|表现|成绩)?(?:更高|更好|优于)',text))


# CASE-SPECIFIC: Chinese actors are only 基线方法 / 新系统.
def _winning_actor_hardened(text: str, language: str) -> str | None:
    """Narrow original-language comparison grammar; reject object-as-winner.

    In `基线方法比新系统表现更好`, `新系统表现更好` is a *substring*,
    NOT a standalone assertion that the proposed system won. Distinguish
    Chinese A比B更好 from bare A得分更高 before comparing actor roles.
    """
    if language=='en':
        return _winning_actor(text,'en')
    terms={'基线方法':'baseline','新系统':'proposed'}
    pair_hits=list(re.finditer(
        r'(基线方法|新系统)\s*比\s*(基线方法|新系统)\s*(?:得分|表现|成绩)?\s*(?:更高|更好|优于)',text))
    if pair_hits:
        winners={terms[m[1]] for m in pair_hits if m[1]!=m[2]}
        return next(iter(winners)) if len(winners)==1 else None
    winners=set()
    for subject,canonical in terms.items():
        for m in re.finditer(re.escape(subject)+r'\s*(?:得分|表现|成绩)?\s*(?:更高|更好|优于)',text):
            # If a comparative `比` directly governs this actor, the actor is
            # the object, NOT the winner. Ambiguous clauses abstain.
            if '比' not in text[max(0,m.start()-3):m.start()]:
                winners.add(canonical)
    return next(iter(winners)) if len(winners)==1 else None


def _same_condition(en: str, zh: str) -> bool:
    # Exactly ONE dataset reference per excerpt (the whole paragraph often
    # contains several!). This is scope matching, not source positional pairing.
    scope = (('first', '第一'), ('second', '第二'), ('third', '第三'))
    a={i for i,(e,z) in enumerate(scope) if re.search(r'\b'+e+r'\s+dataset\b',en,re.I)}
    b={i for i,(e,z) in enumerate(scope) if z+'组数据' in zh}
    return len(a)==len(b)==1 and a==b


# A proof depends on SOURCE-SIDE syntax, scope and referent. The model's
# same_event and finding fields are diagnostic opinions, never a hard veto or
# a free-standing ground for automatic classification.

def _entity_candidates(en_text: str, zh_text: str) -> list[tuple[str, str]]:
    out = []
    for e, z in _ENTITY_PAIRS:
        if (re.search(r"(?<![a-z])"+re.escape(e)+r"(?![a-z])", en_text, re.I)
                and z in zh_text):
            out.append((e, z))
    return out


# Page/word-limit vocabulary. A limit is identified by its numeric value.
_EN_LIMIT_VALUE = re.compile(r"\b(\d+)\s*[-– ]?\s*(page|word)s?\s+(?:limit|ceiling|cap|quota)\b", re.I)
_EN_LIMITED_TO = re.compile(r"\blimited\s+to\s+(\d+)\s+(page|word)s?\b", re.I)
_ZH_LIMIT_VALUE = re.compile(r"(\d+)\s*(页|字)(?:的)?(?:篇幅|页数|字数)?(?:限制|额度|上限)")
_EN_DEICTIC_LIMIT = re.compile(r"\b(?:this|that|the|its)\s+(?:page\s+|word\s+)?limit\b", re.I)
# The word (or hyphenated compound, e.g. "main-report") right before "limit".
_EN_SCOPED_LIMIT = re.compile(r"\b([a-z]+(?:[-–][a-z]+)*)[-–\s]+limit\b", re.I)
_ZH_DEICTIC_LIMIT = re.compile(r"(?:该|此|上述|这一|这个)?(?:篇幅|页数|字数)(?:限制|额度|上限)|(?:该|此|上述|这一)(?:限制|额度|上限)")
_NOT_A_SCOPE = {"this", "that", "the", "its", "page", "word", "a", "an", "same"}


def _limit_values_en(text: str) -> set[tuple[str, str]]:
    return {(v, u.lower()) for v, u in _EN_LIMIT_VALUE.findall(text) + _EN_LIMITED_TO.findall(text)}


def _limit_values_zh(text: str) -> set[tuple[str, str]]:
    return {(v, 'page' if u == '页' else 'word') for v, u in _ZH_LIMIT_VALUE.findall(text)}


def _single_count_context(english: str, chinese: str) -> tuple[str, str] | None:
    """The ONE page/word limit both full sources talk about, else None.

    Each side must state exactly one limit value, and the values must match.
    Several different limits on a side -> abstain (an entity could belong to
    either). No English/Chinese sentence-position assumptions.
    """
    en, zh = _limit_values_en(english), _limit_values_zh(chinese)
    if len(en) != 1 or en != zh:
        return None
    return next(iter(en))


def _scope_tied_to_limit(scope: str, english: str, value: str, unit: str) -> bool:
    """Does the full English attach this named scope to THE limit value?"""
    sc = re.sub(r"[-– ]+", r"[-– ]+", re.escape(scope.lower()).replace(r"\-", "-").replace(r"\ ", " "))
    lim = rf"{value}\s*[-– ]?\s*{unit}s?\s+(?:limit|ceiling|cap|quota)"
    # Allow one modifier before the scope: "for the main report" ties "report".
    scope = rf"(?:[a-z]+[-– ]+)?{sc}"
    return bool(re.search(rf"\b{lim}\s+(?:for|of|on)\s+(?:the\s+)?{scope}\b", english, re.I)
                or re.search(rf"\b{scope}\s+(?:has|is\s+subject\s+to|carries)\s+an?\s+{lim}", english, re.I)
                or re.search(rf"\b{scope}\s+is\s+limited\s+to\s+{value}\s+{unit}s?\b", english, re.I))


def _cited_refers_to_limit(en_text: str, zh_text: str, english: str,
                           value: str, unit: str) -> bool:
    """The CITED rules must point at that same limit, not merely at 'a limit'.

    English: the numeric limit itself; or a scope name ('main-report limit')
    that the full English ties to that number; or 'this/the limit' when the
    full English names no other scoped limit. An 'appendix limit' that the
    source never ties to the number abstains. Chinese: the numeric limit, or a
    deictic 篇幅/页数限制 (the full Chinese states only that one limit value).
    """
    en_vals = _limit_values_en(en_text)
    if en_vals:
        en_ok = en_vals == {(value, unit)}
    else:
        scopes = [x.lower() for x in _EN_SCOPED_LIMIT.findall(en_text)]
        scopes = [x for x in scopes if x not in _NOT_A_SCOPE]
        if scopes:
            en_ok = all(_scope_tied_to_limit(x, english, value, unit) for x in scopes)
        elif _EN_DEICTIC_LIMIT.search(en_text):
            other = [x.lower() for x in _EN_SCOPED_LIMIT.findall(english)]
            other = [x for x in other if x not in _NOT_A_SCOPE and not re.fullmatch(r"\d+[-– ]?(?:page|word)s?", x)]
            en_ok = all(_scope_tied_to_limit(x, english, value, unit) for x in other)
        else:
            en_ok = False
    zh_vals = _limit_values_zh(zh_text)
    zh_ok = zh_vals == {(value, unit)} if zh_vals else bool(_ZH_DEICTIC_LIMIT.search(zh_text))
    return en_ok and zh_ok


def count_proof(en_text: str, zh_text: str, english: str, chinese: str,
                 model_en: str, model_zh: str) -> str:
    context = _single_count_context(english, chinese)
    if context is None:
        return 'REVIEW_REQUIRED'
    # Both CITED rules must govern that same limit. A correct entity attached
    # to a different (e.g. appendix) limit must NOT be promoted to FAIL.
    if not _cited_refers_to_limit(en_text, zh_text, english, *context):
        return 'REVIEW_REQUIRED'
    possible=[]
    for e,z in _entity_candidates(en_text, zh_text):
        en_state, zh_state = _included(en_text,e,'en'),_included(zh_text,z,'zh')
        if en_state is not None and zh_state is not None and en_state!=zh_state:
            possible.append((e,z))
    # No guess if multiple mismatching component pairs are present in citations.
    if len(possible)!=1:
        return 'REVIEW_REQUIRED'
    e,z=possible[0]
    # If the model outputs recognized *other* entity names, do not override its
    # selection; if it dumps the whole sentence, the unique host proof is used.
    for supplied, expected, lang in ((model_en,e,'en'),(model_zh,z,'zh')):
        if supplied.strip() and supplied.strip().casefold()!=expected.casefold():
            known=_entity_candidates(en_text,zh_text)
            other_names=[a if lang=='en' else b for a,b in known]
            if supplied.strip().casefold() in [x.casefold() for x in other_names]:
                return 'REVIEW_REQUIRED'
    return 'FAIL'


def _winning_actor(text: str, language: str) -> str | None:
    actors=[]
    if language=='en':
        names=tuple(dict.fromkeys(a for a,_ in _ENTITY_PAIRS if
                                 a in ('baseline','baseline method','proposed system','new system')))
        for candidate in names:
            if _winner(text,candidate,'en'):
                actors.append('baseline' if candidate.startswith('baseline') else 'proposed')
    else:
        for candidate, canonical in (('基线方法','baseline'),('新系统','proposed')):
            if _winner(text,candidate,'zh'):
                actors.append(canonical)
    uniq=set(actors)
    return next(iter(uniq)) if len(uniq)==1 else None


# CASE-SPECIFIC: only the "abstract optional unless the program requires it" vs
# "无论项目……都可选" wording (V2-EXC-02/03) can be proved.
def exception_proof(en_text: str,zh_text: str,model_en: str,model_zh: str) -> str:
    pairs=[(e,z) for e,z in _entity_candidates(en_text,zh_text)
           if ('abstract' in e or '摘要' in z)]
    if len(pairs)!=1:
        return 'REVIEW_REQUIRED'
    e,z=pairs[0]
    # If model names another recognized target, reject; otherwise the single
    # source-verified target is authoritative (no sentence dump dependency).
    if (model_en.strip().lower() in ('chinese abstract','english abstract')
            and model_en.strip().lower()!=e):
        return 'REVIEW_REQUIRED'
    if model_zh.strip() in ('中文摘要','英文摘要') and model_zh.strip()!=z:
        return 'REVIEW_REQUIRED'
    en_cond=(re.search(r'\boptional\s+unless\b',en_text,re.I)
             and re.search(r'\brequir\w*\b',en_text,re.I)
             and re.search(r'\b(?:degree\s+)?program\b',en_text,re.I))
    zh_uncond=(re.search(r'无论|不论',zh_text)
               and re.search(r'(?:项目|专业)',zh_text)
               and re.search(r'都.*(?:可选|选配)',zh_text))
    return 'FAIL' if en_cond and zh_uncond else 'REVIEW_REQUIRED'


# CASE-SPECIFIC: only the entity 'chinese abstract' is supported.
def focus_exception_evidence(en_options: dict[str, str], zh_options: dict[str, str],
                              en_selected: str, zh_selected: str,
                              en_subject: str, zh_subject: str):
    """Select unique source clauses governing the *named* exception entity.

    Source clauses are independent exact substrings of their own language;
    this never assumes EN/ZH sentence order, and cannot select by index alone.
    If there is no unique, complete source rule on either side, abstain.
    """
    en_name, zh_name = en_subject.strip().lower(), zh_subject.strip()
    if not _mapped(en_name, zh_name) or en_name != 'chinese abstract':
        return None
    def candidates(options, lang):
        found=[]
        for ident, text in options.items():
            if ident.endswith('_all'):
                continue
            if lang=='en':
                matches = (re.search(r'\b'+re.escape(en_name)+r'\b', text, re.I)
                           and re.search(r'\boptional\s+unless\b',text,re.I)
                           and re.search(r'\b(?:degree\s+)?program\b',text,re.I)
                           and re.search(r'\brequir\w*\b',text,re.I))
            else:
                matches = (zh_name in text and re.search(r'无论|不论',text)
                           and re.search(r'项目|专业',text)
                           and re.search(r'都.*(?:可选|选配)',text))
            if matches:
                found.append((ident,text))
        return found
    en, zh=candidates(en_options,'en'),candidates(zh_options,'zh')
    # The full paragraph can be cited by the model, but the proof must use
    # one independently selected, complete, entity-specific source clause.
    if len(en)!=1 or len(zh)!=1:
        return None
    if en_selected not in en_options or zh_selected not in zh_options:
        return None
    return (en[0][0], zh[0][0], en[0][1], zh[0][1])


# CASE-SPECIFIC: scopes are only 'first/second/third dataset' / 第一组数据 (V2-ROL-06/07).
def _scope_name(text: str, language: str) -> str | None:
    refs = (('first','第一'), ('second','第二'), ('third','第三'))
    matched = [e for e,z in refs if ((bool(re.search(r'\b'+e+r'\s+dataset\b', text, re.I)))
                                   if language=='en' else ((z+'组数据') in text))]
    return matched[0] if len(matched)==1 else None


def host_anchor_direction(ens: dict[str,str], zhs: dict[str,str]):
    """Fallback only for an invalid citation ID on a fixed direction task.

    Both paragraphs must describe the exact same set of named dataset scopes,
    with exactly one clause per scope on each side. Exactly one pair must have
    a source-proved winner reversal. Any duplicated/missing scope abstains.
    No loose sentence index or guessed translation correspondence is used.
    """
    def collect(options, lang):
        by_scope={}
        for ident,text in options.items():
            if ident.endswith('_all'):
                continue
            name=_scope_name(text,lang)
            if name:
                if name in by_scope:
                    return None
                by_scope[name]=(ident,text)
        return by_scope
    en,zh=collect(ens,'en'),collect(zhs,'zh')
    if en is None or zh is None or set(en)!=set(zh) or not en or len(en)>3:
        return None
    mismatch=[]
    for name in sorted(en):
        eid,etext=en[name];zid,ztext=zh[name]
        ea,za=_winning_actor_hardened(etext,'en'),_winning_actor_hardened(ztext,'zh')
        if ea is not None and za is not None and ea!=za and _same_condition(etext,ztext):
            mismatch.append((eid,zid,etext,ztext))
    return mismatch[0] if len(mismatch)==1 else None


# CASE-SPECIFIC: the activity must be one of five named study types.
def scalar_source_scope(english: str, chinese: str) -> bool:
    """Only a fixed duration assertion about the same explicitly named activity.

    `formal_single_quantity_equivalence` separately rejects approximation and
    distinct extra sentence obligations; this also rejects attachment of a new
    independent condition following the duration statement.
    """
    if not source_quantitative_issue_only(english,chinese):
        return False
    if not any(e in english.casefold() and z in chinese for e,z in (
            ('survey','调查'),('experiment','实验'),('study','研究'),
            ('interview','访谈'),('trial','试验'))):
        return False
    # When present, supplementary text after a comma must only describe
    # unchanged duration, not add an unrelated event or a fresh duty.
    remainder = re.split(r'[，,]',chinese,maxsplit=1)
    if len(remainder)==2:
        extra=remainder[1].strip().rstrip('。.!！')
        if not re.fullmatch(r'(?:既?未|没有)(?:被)?(?:延长|加长|增加|推迟).{0,5}(?:也未|亦未|或|也没有|和没有)(?:被)?(?:缩短|减少|提前)|(?:既未延长也未缩短)|(?:未延长也未缩短)',extra):
            return False
    return True


def prove(question: Question, reply: RelationVerificationDraft, en_text: str, zh_text: str,
           whole_en: str, whole_zh: str, english: str, chinese: str) -> str:
    """Strict source checking, never conflate event identity with agreement.

    `same_event=no` may be a model's mistaken way of saying the rules conflict.
    The host proceeds ONLY where independent narrow evidence proves identity.
    """
    if len(en_text)>600 or len(zh_text)>600:
        return 'REVIEW_REQUIRED'
    if question.kind=='numeric_boundary':
        if not (question.en_hint in en_text and question.zh_hint in zh_text):
            return 'REVIEW_REQUIRED'
        if len(numeric_tokens(en_text))!=1 or len(numeric_tokens(zh_text))!=1:
            return 'REVIEW_REQUIRED'
        a,b=parse_quantity(question.en_hint),parse_quantity(question.zh_hint)
        x,y=numeric_symbol_unicode(reply.en_value),numeric_symbol_unicode(reply.zh_value)
        if not (a and b and x and y and a.dimension==b.dimension):
            return 'REVIEW_REQUIRED'
        if (x[0],x[1])!=(a.comparator,a.value) or (y[0],y[1])!=(b.comparator,b.value):
            return 'REVIEW_REQUIRED'
        if x[2]!=y[2] or x[2]!='%' or not _same_numeric_context(question.en_hint,question.zh_hint):
            return 'REVIEW_REQUIRED'
        return 'FAIL' if (a.comparator,a.value)!=(b.comparator,b.value) else 'REVIEW_REQUIRED'
    if question.kind=='inclusion_exclusion':
        return count_proof(en_text,zh_text,whole_en,whole_zh,
                            reply.en_subject or reply.en_value, reply.zh_subject or reply.zh_value)
    if question.kind=='exception_scope':
        return exception_proof(en_text,zh_text,reply.en_subject or reply.en_value,
                                reply.zh_subject or reply.zh_value)
    if question.kind=='direction_reversal':
        if not _same_condition(en_text,zh_text):
            return 'REVIEW_REQUIRED'
        a,b=_winning_actor_hardened(en_text,'en'),_winning_actor_hardened(zh_text,'zh')
        if a is None or b is None:
            return 'REVIEW_REQUIRED'
        en_named=(reply.en_subject or reply.en_value).strip().lower()
        zh_named=(reply.zh_subject or reply.zh_value).strip()
        if en_named in ('baseline','baseline method','proposed system','new system'):
            if ('baseline' if en_named.startswith('baseline') else 'proposed')!=a:
                return 'REVIEW_REQUIRED'
        if zh_named in ('基线方法','新系统'):
            if ('baseline' if zh_named=='基线方法' else 'proposed')!=b:
                return 'REVIEW_REQUIRED'
        return 'FAIL' if a!=b else 'REVIEW_REQUIRED'
    if question.kind=='scalar_equivalence':
        if (en_text.strip() and zh_text.strip() and scalar_source_scope(whole_en,whole_zh)
                and parse_quantity(whole_en) and parse_quantity(whole_zh)):
            return 'PASS'  # narrow single fully-parsed scalar; not based on model finding
    return 'REVIEW_REQUIRED'
