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


def test_normalize_identifier_like_spans_compacts_supported_dotted_identifiers():
    text = "ML-KEM . Decaps and ML-KEM . ParamSets in NIST . FIPS . 203 Section 3 . 3."
    normalized = normalize_identifier_like_spans(text)
    assert "ML-KEM.Decaps" in normalized
    assert "ML-KEM.ParamSets" in normalized
    assert "NIST.FIPS.203" in normalized
    assert "3.3" in normalized


def test_normalize_identifier_like_spans_preserves_prose_spacing_after_periods():
    text = (
        "U.S. Department of Gina M. Raimondo, Secretary. "
        "Laurie E. Locascio served. "
        "A shared secret key over a public channel. A shared secret key that is securely established. "
        "4. Approving Authority. Secretary of Commerce."
    )
    normalized = normalize_identifier_like_spans(text)
    assert "U.S. Department of Gina M. Raimondo, Secretary." in normalized
    assert "Laurie E. Locascio served." in normalized
    assert "channel. A shared secret key" in normalized
    assert "4. Approving Authority. Secretary of Commerce." in normalized
