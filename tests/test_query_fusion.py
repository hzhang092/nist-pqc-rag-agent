from rag.retrieve import query_variants


def test_query_variants_domain_rules():
    q = "Algorithm 19 key generation for ML-KEM.KeyGen and K-PKE.KeyGen"
    variants = query_variants(q)

    assert variants[0] == q
    assert "ML-KEM.KeyGen K-PKE.KeyGen" in variants
    assert "ML-KEM.KeyGen key generation" in variants
    assert "Algorithm 19 ML-KEM.KeyGen" in variants


def test_query_variants_empty_and_dedup():
    assert query_variants("   ") == []

    q = "ML-KEM.KeyGen"
    variants = query_variants(q)
    assert variants == ["ML-KEM.KeyGen"]
