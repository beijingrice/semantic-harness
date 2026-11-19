# Task: independently label English/Chinese requirement pairs

`./unlabeled.json` in the current directory is a JSON array of cases `{"id","en","zh"}`. For
each case decide whether the English and Chinese texts state the SAME set of requirements.
Work only inside the current directory; do not read other files or the web. Do not run code to
decide labels; read each case yourself carefully.

Rules:
PASS: a careful bilingual reader derives the same obligations from both texts: what must / may /
must not be done, by whom, how much, when, under which conditions and exceptions. Wording,
sentence order, splitting/merging sentences, paragraphing and exact unit conversions do not
matter.
FAIL if ANY key fact differs, in either direction (Chinese adding a requirement is also FAIL):
a requirement, list item, condition or exception on one side only; a different number or
different value after unit conversion; reversed bound direction; strict vs inclusive boundary
(`more than 10%` vs `10%或以上` is FAIL; `at least 40 pages` vs `40页以上` is PASS, 以上/以下
include the number); obligation strength changed (must/shall/应/须 vs may/可以 vs must not/不得;
"not required" vs "prohibited"); actor/object swapped; comparison direction reversed (but
`A higher than B` vs `B低于A` is PASS); order of steps reversed; a condition/exception widened,
narrowed or removed; a necessary condition ("only if") lost; counted vs not counted toward a
limit; a different technical concept (abbreviation vs full name is PASS).

Write `./labels.json`: a JSON array of `{"id": "...", "label": "PASS|FAIL", "confidence":
"high|low", "reason": "one or two sentences; for FAIL name the exact difference"}` with one
entry per input case, same order. Use confidence "low" when the case is genuinely ambiguous
under these rules. Finish with one line: number of cases labeled.
