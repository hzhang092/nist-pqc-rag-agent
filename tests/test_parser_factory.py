import pytest

from rag.parsers.factory import get_parser_backend


def test_get_parser_backend_llamaparse():
    backend = get_parser_backend("llamaparse")
    assert backend.name == "llamaparse"


def test_get_parser_backend_docling_if_installed():
    pytest.importorskip("docling")
    backend = get_parser_backend("docling")
    assert backend.name == "docling"


def test_get_parser_backend_unknown_raises():
    with pytest.raises(ValueError, match="Unknown parser backend"):
        get_parser_backend("unknown-backend")
