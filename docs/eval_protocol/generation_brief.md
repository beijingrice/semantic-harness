# Task: write an evaluation set of English/Chinese requirement pairs

You are writing test data for a tool that checks whether an English text and a Chinese text
state the SAME set of requirements (academic / institutional rules). Each case is a pair
(`en`, `zh`) plus the correct label: `PASS` (same key facts) or `FAIL` (at least one key fact
differs).

Work ONLY inside the current directory. Do not read any other files on this computer and do
not search the web: the data must be written from scratch. Write the result to
`./cases.json` (a JSON array, UTF-8, `ensure_ascii=False`, 2-space indent). No other output
files.

Write exactly __N__ cases with ids `__PREFIX__-001` ... in random order of category.

## Labeling rules (follow exactly)

PASS: a careful bilingual reader would derive the same obligations from both texts: what must
/ may / must not be done, by whom, how much, when, under which conditions and exceptions.
Wording, sentence structure, order of sentences, splitting or merging sentences, paragraphing,
and exact unit conversions do not matter.

FAIL if ANY key fact differs, in either direction (Chinese adding a requirement is also FAIL):
- a requirement, list item, condition or exception present on one side only;
- different number, different value after unit conversion, reversed bound direction, or a
  strict vs inclusive boundary (`more than 10%` vs `10%或以上` is FAIL; `at least 40 pages` vs
  `40页以上` is PASS because 以上/以下 include the number);
- obligation strength changed: must/shall ↔ may/optional ↔ must not; "not required" ↔
  "prohibited";
- actor/object swapped, comparison direction reversed, order of steps reversed
  (`A higher than B` ↔ `B低于A` is PASS);
- scope of a condition or exception widened, narrowed or removed; a "only if" (necessary
  condition) lost;
- counted vs not counted toward a page/word limit;
- different technical concept (`sensitivity` ↔ `特异度`); abbreviation vs full name is PASS.

## Phenomena you must AVOID (their labels are not settled yet)
- Never use "should" / "ought to" in English, and avoid 宜/建议 in Chinese. Use must/shall/is
  required to/may/must not and 必须/须/应/应当/可以/不得.
- Do not pair a descriptive sentence on one side with a normative one on the other ("The report
  contains X" vs "报告应包含X"). Keep the same mood on both sides.
- Do not build cases whose label depends on "when/if" (sufficient condition) vs "only
  if/方可/才" (necessary condition). Use the same kind of condition on both sides unless the
  intended FAIL is an explicit, unmistakable removal of "only if".
- Do not use near-synonyms whose scope is debatable (methodology ↔ 技术路线, data processing ↔
  数据预处理). Either use exact equivalents or clearly different concepts.

## Coverage
Balance: exactly half PASS, half FAIL. Spread cases over these categories (use the category
string exactly), roughly evenly:
`numeric_units`, `numeric_bounds`, `modality`, `negation`, `conditions_exceptions`,
`enumeration`, `roles_direction`, `order_timing`, `inclusion_counting`, `terminology`,
`quantifier_scope` (each vs total, all vs some, per person vs per group), `long_mixed`.

Make it realistic and varied:
- Domains: thesis formatting, course assessment, lab safety, research ethics and data
  protection, grant applications, conference submissions, exam rules, library and equipment
  use, internship reports, dataset licensing, clinical study protocols, software assignments.
  Do not reuse the same scenario more than twice.
- Length: about 35% short (one sentence), 35% medium (2-4 sentences), 30% long (5-10
  sentences, often 2-3 paragraphs separated by a blank line "\n\n"). `long_mixed` cases are
  long and contain several requirements of different kinds; a FAIL there differs in exactly one
  or two of them, buried among faithfully translated ones.
- Chinese must be natural, idiomatic Chinese as a Chinese university office would write it,
  not word-by-word translation. PASS cases should often restructure: reorder clauses, merge or
  split sentences, convert units exactly (2 hours ↔ 120分钟, 1.5 cm ↔ 15毫米), use Chinese
  numerals sometimes (三十页).
- FAIL cases: the difference must be real and decidable, but natural and often subtle; do NOT
  make the Chinese of FAIL cases systematically shorter or longer than PASS cases. Some FAIL
  cases should ADD something on the Chinese side. Include some FAIL cases where the English and
  Chinese look very similar on the surface.
- Include about 10 minimal pairs: a PASS case and a FAIL case that share the same English and
  differ only in the Chinese. Mention the partner id in `why`.

## Fields per case
```
{"id": "__PREFIX__-001", "category": "...", "difficulty": "easy|medium|hard",
 "en": "...", "zh": "...", "expected": "PASS|FAIL",
 "why": "For FAIL: every key difference, precisely. For PASS: which elements correspond."}
```

## Self-check before finishing
Re-read every case as a strict bilingual reviewer. Verify: the label follows the rules above;
a FAIL case has no unintended extra difference that is missing from `why`; a PASS case has no
accidental difference (numbers, units, list items, modality, scope); none of the avoided
phenomena appear; the JSON parses; ids are unique; PASS/FAIL counts are equal. Fix problems
and re-check. Then finish with one line: the number of cases and the PASS/FAIL counts.
