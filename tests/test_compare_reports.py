from examples.compare_reports import compare


def row(ident, expected, predicted):
    return {
        "id": ident,
        "expected": expected,
        "predicted": predicted,
        "correct": expected == predicted,
        "category": "toy",
    }


def test_compares_only_intersection_of_successful_cases():
    old = {"a": row("a", "PASS", "PASS"), "b": row("b", "FAIL", "PASS"), "c": row("c", "FAIL", "FAIL")}
    new = {"a": row("a", "PASS", "FAIL"), "b": row("b", "FAIL", "FAIL"), "c": row("c", "FAIL", "ERROR")}
    result = compare(old, new)
    assert result["both_evaluated"] == 2
    assert result["old"]["accuracy"] == 0.5
    assert result["new"]["accuracy"] == 0.5
    assert [x["id"] for x in result["improvements"]] == ["b"]
    assert [x["id"] for x in result["regressions"]] == ["a"]
