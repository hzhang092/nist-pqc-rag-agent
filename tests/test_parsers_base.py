from rag.parsers.base import markdown_to_text


def test_markdown_to_text_preserves_page_one_prose_spacing_and_dotted_identifiers():
    markdown = (
        "## FIPS 203\n\n"
        "## Federal Information Processing Standards Publication\n\n"
        "## Module-Lattice-Based Key-Encapsulation Mechanism Standard\n\n"
        "Category:\n\n"
        "Computer Security\n\n"
        "Subcategory:\n\n"
        "Cryptography\n\n"
        "Information Technology Laboratory National Institute of Standards and Technology Gaithersburg, MD 20899-8900\n\n"
        "This publication is available free of charge from: https://doi.org/10.6028/NIST.FIPS.203\n\n"
        "Published August 13, 2024\n\n"
        "Commerce\n\n"
        "U.S. Department of Gina M. Raimondo, Secretary\n\n"
        "National Institute of Standards and Technology\n\n"
        "Laurie E. Locascio, NIST Director and Under Secretary of Commerce for Standards and Technology"
    )

    text = markdown_to_text(markdown)

    assert "https://doi.org/10.6028/NIST.FIPS.203" in text
    assert "U.S. Department of Gina M. Raimondo, Secretary" in text
    assert "Laurie E. Locascio, NIST Director" in text
    assert "U.S.Department" not in text
    assert "M.Raimondo" not in text
    assert "E.Locascio" not in text
