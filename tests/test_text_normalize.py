from rag.text_normalize import normalize_identifier_like_spans


def test_normalize_identifier_like_spans_unescapes_identifier_separators():
    text = r"Use MAC\_Data and ML-KEM\.KeyGen plus KC \_ Step \_ Label."
    normalized = normalize_identifier_like_spans(text)
    assert "MAC_Data" in normalized
    assert "ML-KEM.KeyGen" in normalized
    assert "KC_Step_Label" in normalized


def test_normalize_identifier_like_spans_preserves_non_identifier_backslashes():
    text = r"\alpha_i remains; path C:\temp\file remains; MAC\Data stays."
    normalized = normalize_identifier_like_spans(text)
    assert r"\alpha_i" in normalized
    assert r"C:\temp\file" in normalized
    assert r"MAC\Data" in normalized
