"""
Unit tests for the query fusion/variant generation logic.

These tests verify that the `query_variants` function correctly generates
a diverse and relevant set of query variations based on a single user query.
This includes applying domain-specific rules (e.g., expanding "signing" to
"ML-DSA.Sign") and handling edge cases like empty input.
"""

from rag.retrieve import query_variants


def test_query_variants_domain_rules():
    """
    Tests that specific, hard-coded domain rules are correctly applied to
    generate relevant query variants for technical terms.
    """
    q = "Algorithm 19 key generation for ML-KEM.KeyGen and K-PKE.KeyGen"
    variants = query_variants(q)

    assert variants[0] == q
    assert "ML-KEM.KeyGen K-PKE.KeyGen" in variants
    assert "ML-KEM.KeyGen key generation" in variants
    assert "Algorithm 19" in variants
    assert "Algorithm 19 ML-KEM.KeyGen" in variants


def test_query_variants_generalized_rules():
    """
    Tests that more general rules (e.g., 'signing' -> '.Sign', 'keygen' -> '.KeyGen')
    are correctly applied across different algorithm families.
    """
    v1 = query_variants("ML-DSA signing process")
    assert "ML-DSA.Sign" in v1

    v2 = query_variants("How to verify in ML-DSA")
    assert "ML-DSA.Verify" in v2

    v3 = query_variants("SLH-DSA keygen details")
    assert "SLH-DSA.KeyGen" in v3

    v4 = query_variants("ML-KEM decapsulation steps")
    assert "ML-KEM.Decaps" in v4


def test_query_variants_empty_and_dedup():
    """
    Tests that empty input produces an empty list and that duplicate variants
    are removed, with the original query preserved.
    """
    assert query_variants("   ") == []

    q = "ML-KEM.KeyGen"
    variants = query_variants(q)
    assert variants == ["ML-KEM.KeyGen"]
