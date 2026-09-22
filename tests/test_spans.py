from __future__ import annotations

import json

import pytest

from voicefit import PROTECTED_KINDS, SCHEMA_VERSION, extract_spans, verify_spans


def test_urls_lose_their_trailing_punctuation():
    spans = extract_spans("Docs live at https://example.com/a/b, see them.")
    assert [(span.kind, span.text) for span in spans] == [
        ("url", "https://example.com/a/b")
    ]


def test_code_spans_keep_their_contents():
    text = "Call `build_profile(texts)` then:\n\n```python\npip install -e .\n```\n"
    assert [span.text for span in extract_spans(text, kinds=["code"])] == [
        "build_profile(texts)",
        "pip install -e .",
    ]


def test_numbers_and_identifiers_are_told_apart():
    spans = extract_spans("Ticket ENG-421 cut latency to 3.5 ms, or 40% of v1.2.3.")
    assert [(span.kind, span.text) for span in spans] == [
        ("identifier", "ENG-421"),
        ("number", "3.5"),
        ("number", "40%"),
        ("identifier", "v1.2.3"),
    ]


def test_a_url_inside_a_code_span_is_one_span():
    assert [(span.kind, span.text) for span in extract_spans(
        "Run `curl https://example.com`."
    )] == [("code", "curl https://example.com")]


def test_every_declared_kind_can_be_extracted():
    spans = extract_spans("See `x = 1` at https://e.com/p for ENG-7 and 12 rows.")
    assert {span.kind for span in spans} == set(PROTECTED_KINDS)


def test_spans_carry_their_offsets():
    text = "Value 42 here."
    span = extract_spans(text)[0]
    assert span.kind == "number"
    assert span.text == "42"
    assert text[span.start : span.end] == span.text


def test_prose_abbreviations_are_not_identifiers():
    assert extract_spans("Use it, e.g. for a draft.") == ()


def test_a_rewrite_that_keeps_everything_is_intact():
    result = verify_spans(
        "Bump to v2.1.0 and retry 3 times.",
        "Retry 3 times after the bump to v2.1.0.",
    )
    assert result.intact
    assert result.lost == ()
    assert all(check.survived for check in result.checks)


def test_a_dropped_number_and_url_are_reported():
    result = verify_spans(
        "Deploy 3 replicas behind https://api.example.com/v2.",
        "Deploy several replicas behind the gateway.",
    )
    assert not result.intact
    assert [check.text for check in result.lost] == [
        "3",
        "https://api.example.com/v2",
    ]
    assert result.check("3").missing == 1


def test_a_number_glued_into_a_longer_number_does_not_count():
    result = verify_spans("We saw 4 errors.", "We saw 42 errors.")
    assert result.check("4").found == 0
    assert not result.intact


def test_a_url_with_a_changed_path_does_not_count():
    result = verify_spans("Read https://example.com/a.", "Read https://example.com/a/b.")
    assert result.check("https://example.com/a").found == 0


def test_repeated_spans_must_all_survive():
    check = verify_spans("Retry 3 times, then 3 more.", "Retry 3 times.").check("3")
    assert check.expected == 2
    assert check.found == 1
    assert check.missing == 1


def test_kinds_narrow_what_is_protected():
    source = "Send 5 requests to https://example.com/api."
    rewrite = "Send a few requests to https://example.com/api."
    assert not verify_spans(source, rewrite).intact
    assert verify_spans(source, rewrite, kinds=["url"]).intact


def test_report_lists_every_span_with_a_flag():
    report = verify_spans("Ship 2 fixes to `main`.", "Ship the fixes.").report()
    assert "lost" in report
    assert "main" in report
    assert "2" in report


def test_report_serialises_to_json():
    result = verify_spans("Retry 3 times at https://example.com.", "Retry a few times.")
    payload = json.loads(result.to_json())
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["intact"] is False
    assert payload["lost"] == [check.text for check in result.lost]
    assert [check["text"] for check in payload["checks"]] == [
        check.text for check in result.checks
    ]


def test_a_text_without_protected_spans_is_trivially_intact():
    result = verify_spans("The migration ran clean.", "Nothing carried over.")
    assert result.checks == ()
    assert result.intact


def test_unknown_kinds_and_wrong_types_are_rejected():
    with pytest.raises(ValueError):
        extract_spans("text", kinds=["urls"])
    with pytest.raises(TypeError):
        extract_spans(["text"])
    with pytest.raises(KeyError):
        verify_spans("Retry 3 times.", "Retry.").check("nope")
