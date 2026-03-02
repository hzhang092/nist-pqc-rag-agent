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
        except Exception:
            # Fallback: plain converter (still works for many PDFs).
            return DocumentConverter()

    def _extract_formula_latex(self, doc: Any) -> list[str]:
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
                    latex = getattr(item, "text", None) or getattr(item, "latex", None)
                    if latex:
                        formulas.append(str(latex).strip())
        except Exception:
            return []
        return [f for f in formulas if f]

    def _inject_formulas(self, markdown: str, formulas: list[str]) -> str:
        placeholder = "<!-- formula-not-decoded -->"
        n_ph = markdown.count(placeholder)
        if n_ph <= 0 or not formulas:
            return markdown

        # If counts align, do a clean 1:1 replacement. Otherwise replace what we can and
        # append the remaining formulas for retrieval/debugging.
        n_replace = min(n_ph, len(formulas))
        out = markdown
        for i in range(n_replace):
            out = out.replace(placeholder, f"$$ {formulas[i]} $$", 1)

        remaining = formulas[n_replace:]
        if remaining:
            out += "\n\n---\n\n### Extracted formulas (LaTeX)\n" + "\n".join(
                f"- $$ {f} $$" for f in remaining
            )
        return out

    def _page_markdown(self, pdf_path: Path, page_no: int) -> str:
        result = self._converter.convert(pdf_path, page_range=(page_no, page_no))
        doc = result.document
        if not hasattr(doc, "export_to_markdown"):
            return ""

        md = str(doc.export_to_markdown() or "")
        # Fix HTML entities such as &lt; / &gt; in algorithm pseudocode.
        md = html.unescape(md)

        # Replace any formula placeholders with decoded LaTeX if available.
        formulas = self._extract_formula_latex(doc)
        md = self._inject_formulas(md, formulas)

        return md.strip()

    def parse_pdf(self, pdf_path: Path, *, expected_pages: int | None = None) -> list[ParsedPage]:
        if expected_pages is None:
            raise ValueError("DoclingBackend requires expected_pages for deterministic page mapping.")

        out: list[ParsedPage] = []
        for page_no in range(1, int(expected_pages) + 1):
            try:
                markdown = self._page_markdown(pdf_path, page_no)
            except Exception as e:
                print(
                    f"[WARN] Docling failed on {pdf_path.name} page={page_no}: "
                    f"{type(e).__name__}: {e}"
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
