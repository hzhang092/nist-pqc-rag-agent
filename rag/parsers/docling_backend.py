"""Docling-based PDF parser backend.

This backend wraps :class:`docling.document_converter.DocumentConverter` to
extract each PDF page as Markdown, normalize it to plain text, and return a
deterministically ordered list of ``ParsedPage`` records.

Key improvements over the initial implementation:
    - Enables Docling formula enrichment (optional, on by default) so math can be
      decoded (typically into LaTeX).
    - Unescapes HTML entities (e.g., ``&lt;`` → ``<``) that may appear in Markdown.
    - Replaces ``<!-- formula-not-decoded -->`` placeholders with decoded LaTeX
      when available.

Notes:
    - Parsing is performed one page at a time via ``page_range=(page_no, page_no)``.
    - ``expected_pages`` is required to guarantee stable page-to-output mapping.
    - If Docling output does not support ``export_to_markdown``, an empty string
      is used for that page's Markdown/text payload.
    - Results are sorted by ``page_number`` before returning.

Environment toggles:
    - DOCLING_DEVICE (default: "auto"; examples: "cuda", "cuda:0", "cpu")
    - DOCLING_CUDA_USE_FLASH_ATTENTION2 (default: "0")
    - DOCLING_NUM_THREADS (optional; integer)
    - DOCLING_ENABLE_FORMULA_ENRICHMENT (default: "1")
    - DOCLING_CODEFORMULA_PRESET (default: "granite_docling"; also try "codeformulav2")
    - DOCLING_GENERATE_PAGE_IMAGES (default: "1")
    - DOCLING_IMAGES_SCALE (default: "2.0")
    - DOCLING_INJECT_FORMULAS_IN_MARKDOWN (default: "1")

Raises:
    ValueError: If ``parse_pdf`` is called without ``expected_pages``.

Returns:
    list[ParsedPage]: One entry per expected page, containing document id,
    source path, page number, extracted text, Markdown, and backend name.
"""

from __future__ import annotations

from pathlib import Path
import html
import os
import re
from typing import Any

from docling.document_converter import DocumentConverter

from .base import ParsedPage, ParserBackend, markdown_to_text


def _env_flag(name: str, default: bool = True) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str) -> int | None:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return None
    try:
        return int(v)
    except ValueError:
        return None


_DOCLING_IMAGE_COMMENT_RE = re.compile(r"<!--\s*image\s*-->", re.IGNORECASE)
_DOCLING_LOC_TAG_RE = re.compile(r"<loc_\d+>", re.IGNORECASE)
_DOCLING_TEXT_LOC_SPAN_RE = re.compile(r"<text>(?:<loc_\d+>){2,}[^<]*</text>", re.IGNORECASE)
_DOCLING_OPEN_TEXT_LOC_RE = re.compile(r"<text>(?:<loc_\d+>){2,}[^\n]*", re.IGNORECASE)
_DOCLING_BROKEN_FORMULA_TAG_RE = re.compile(r"</?formula\s*>?", re.IGNORECASE)


def _sanitize_docling_markdown(markdown: str) -> str:
    """Strip known Docling artifact patterns that hurt retrieval quality."""
    s = (markdown or "").replace("\r\n", "\n").replace("\r", "\n")
    s = _DOCLING_IMAGE_COMMENT_RE.sub("", s)
    s = _DOCLING_TEXT_LOC_SPAN_RE.sub("", s)
    s = _DOCLING_OPEN_TEXT_LOC_RE.sub("", s)
    s = _DOCLING_LOC_TAG_RE.sub("", s)
    s = s.replace("<text>", "").replace("</text>", "")
    s = _DOCLING_BROKEN_FORMULA_TAG_RE.sub("", s)
    s = "\n".join(ln.rstrip() for ln in s.split("\n"))
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _sanitize_formula_latex(formula: str) -> str:
    """Clean formula strings before markdown injection."""
    s = html.unescape(str(formula or ""))
    s = _DOCLING_BROKEN_FORMULA_TAG_RE.sub("", s)
    s = _DOCLING_LOC_TAG_RE.sub("", s)
    s = s.replace("<text>", "").replace("</text>", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


class DoclingBackend(ParserBackend):
    name = "docling"

    def __init__(self) -> None:
        # Try to enable Docling enrichments (formula decoding) when available.
        self._converter = self._build_converter_with_enrichments()

    def backend_version(self) -> str:
        try:
            from importlib.metadata import version

            return version("docling")
        except Exception:
            return "unknown"

    def _build_converter_with_enrichments(self) -> DocumentConverter:
        enable_formula = _env_flag("DOCLING_ENABLE_FORMULA_ENRICHMENT", True)
        generate_images = _env_flag("DOCLING_GENERATE_PAGE_IMAGES", True)
        images_scale = float(os.getenv("DOCLING_IMAGES_SCALE", "2.0") or "2.0")
        preset = (os.getenv("DOCLING_CODEFORMULA_PRESET", "granite_docling") or "granite_docling").strip()
        device = (os.getenv("DOCLING_DEVICE", "auto") or "auto").strip().lower()
        use_flash_attn2 = _env_flag("DOCLING_CUDA_USE_FLASH_ATTENTION2", False)
        num_threads = _env_int("DOCLING_NUM_THREADS")

        if not enable_formula and not generate_images:
            return DocumentConverter()

        try:
            # These imports may vary slightly across docling versions; keep guarded.
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions, CodeFormulaVlmOptions
            from docling.datamodel.accelerator_options import AcceleratorOptions
            from docling.document_converter import PdfFormatOption

            pipeline_options = PdfPipelineOptions(
                do_formula_enrichment=enable_formula,
                generate_page_images=generate_images,
                images_scale=images_scale,
                accelerator_options=AcceleratorOptions(
                    device=device,
                    cuda_use_flash_attention2=use_flash_attn2,
                    **({"num_threads": num_threads} if num_threads is not None else {}),
                ),
            )

            # Attach a stronger VLM preset for code/formulas, if available.
            if enable_formula:
                try:
                    pipeline_options.code_formula_options = CodeFormulaVlmOptions.from_preset(preset)
                except Exception:
                    # If preset isn't supported in this version, proceed without it.
                    pass

            return DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
            )
        except Exception as e:
            # Fallback: plain converter (still works for many PDFs).
            print(
                "[WARN] Docling enrichment pipeline options unavailable; "
                f"falling back to default converter: {type(e).__name__}: {e}"
            )
            return DocumentConverter()

    def _item_page_numbers(self, item: Any) -> set[int]:
        pages: set[int] = set()
        for attr in ("page_no", "page_number", "page"):
            val = getattr(item, attr, None)
            if isinstance(val, int):
                pages.add(val)
            elif isinstance(val, str) and val.isdigit():
                pages.add(int(val))

        prov = getattr(item, "prov", None)
        if isinstance(prov, list):
            for p in prov:
                if p is None:
                    continue
                for attr in ("page_no", "page_number", "page"):
                    val = getattr(p, attr, None)
                    if isinstance(val, int):
                        pages.add(val)
                    elif isinstance(val, str) and val.isdigit():
                        pages.add(int(val))
        return pages

    def _extract_formula_latex(self, doc: Any, *, page_no: int | None = None) -> list[str]:
        """Best-effort extraction of decoded formulas (LaTeX) from a Docling document."""
        if not _env_flag("DOCLING_INJECT_FORMULAS_IN_MARKDOWN", True):
            return []

        try:
            # Docling formula items are typically exposed via doc.iterate_items()
            from docling_core.types.doc import FormulaItem  # type: ignore
        except Exception:
            return []

        formulas: list[str] = []
        try:
            for item, _lvl in doc.iterate_items():
                if isinstance(item, FormulaItem):
                    if page_no is not None:
                        item_pages = self._item_page_numbers(item)
                        # When page provenance is missing, skip to avoid cross-page formula pollution.
                        if not item_pages or page_no not in item_pages:
                            continue
                    latex = getattr(item, "text", None) or getattr(item, "latex", None)
                    if latex:
                        cleaned = _sanitize_formula_latex(str(latex))
                        if cleaned:
                            formulas.append(cleaned)
        except Exception:
            return []
        return [f for f in formulas if f]

    def _inject_formulas(self, markdown: str, formulas: list[str]) -> str:
        placeholder = "<!-- formula-not-decoded -->"
        n_ph = markdown.count(placeholder)
        if n_ph <= 0:
            return markdown
        if not formulas:
            return markdown.replace(placeholder, "")

        # If counts align, do a clean 1:1 replacement. Otherwise replace what we can and
        # drop extras instead of appending noisy tails into page content.
        n_replace = min(n_ph, len(formulas))
        out = markdown
        for i in range(n_replace):
            out = out.replace(placeholder, f"$$ {formulas[i]} $$", 1)
        return out.replace(placeholder, "")

    def _render_page_markdown(self, doc: Any, page_no: int | None = None) -> str:
        if not hasattr(doc, "export_to_markdown"):
            return ""
        md = str(doc.export_to_markdown(page_no=page_no) or "")
        md = html.unescape(md)
        formulas = self._extract_formula_latex(doc, page_no=page_no)
        md = self._inject_formulas(md, formulas)
        return _sanitize_docling_markdown(md)

    def _page_markdown(self, pdf_path: Path, page_no: int) -> str:
        result = self._converter.convert(pdf_path, page_range=(page_no, page_no))
        doc = result.document
        return self._render_page_markdown(doc)

    def parse_pdf(self, pdf_path: Path, *, expected_pages: int | None = None) -> list[ParsedPage]:
        if expected_pages is None:
            raise ValueError("DoclingBackend requires expected_pages for deterministic page mapping.")

        total_pages = int(expected_pages)
        doc: Any | None = None
        try:
            # Fast path: convert once, then export each page deterministically.
            result = self._converter.convert(pdf_path, page_range=(1, total_pages))
            doc = result.document
        except Exception as e:
            print(
                f"[WARN] Docling full-document conversion failed on {pdf_path.name}: "
                f"{type(e).__name__}: {e}"
            )

        out: list[ParsedPage] = []
        for page_no in range(1, total_pages + 1):
            markdown = ""
            try:
                if doc is not None:
                    markdown = self._render_page_markdown(doc, page_no=page_no)
                else:
                    markdown = self._page_markdown(pdf_path, page_no)
            except Exception as e:
                print(
                    f"[WARN] Docling failed on {pdf_path.name} page={page_no}: "
                    f"{type(e).__name__}: {e}"
                )
                try:
                    # Fallback for per-page resilience.
                    markdown = self._page_markdown(pdf_path, page_no)
                except Exception as e2:
                    print(
                        f"[WARN] Docling fallback failed on {pdf_path.name} page={page_no}: "
                        f"{type(e2).__name__}: {e2}"
                    )
                    markdown = ""
            text = markdown_to_text(markdown)
            out.append(
                ParsedPage(
                    doc_id=pdf_path.stem,
                    source_path=pdf_path.as_posix(),
                    page_number=page_no,
                    text=text,
                    markdown=markdown,
                    parser_backend=self.name,
                )
            )

        out.sort(key=lambda row: int(row["page_number"]))
        return out
