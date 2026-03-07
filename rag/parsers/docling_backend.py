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
    - DOCLING_GENERATE_PAGE_IMAGES (default: "0")
    - DOCLING_IMAGES_SCALE (default: "2.0")
    - DOCLING_INJECT_FORMULAS_IN_MARKDOWN (default: "1")
    - DOCLING_PAGE_BATCH_SIZE (default: "12")
    - DOCLING_MIN_PAGE_BATCH_SIZE (default: "1")
    - DOCLING_ADAPTIVE_BATCHING (default: "1")
    - DOCLING_SET_PYTORCH_ALLOC_CONF (default: "1")

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
import gc
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


_DOCLING_IMAGE_COMMENT_RE = re.compile(r"<!--\s*image\s*-->", re.IGNORECASE) # Docling may insert these as placeholders for page images; they add noise without the actual image, so strip them out.
_DOCLING_LOC_TAG_RE = re.compile(r"<loc_\d+>", re.IGNORECASE) # Docling may insert these location tags in formulas or text; they add noise and hurt retrieval quality, so strip them out.
_DOCLING_TEXT_LOC_SPAN_RE = re.compile(r"<text>(?:<loc_\d+>){2,}[^<]*</text>", re.IGNORECASE) # Docling may produce malformed <text> spans that contain multiple location tags and no real content; these add noise without value, so strip them out entirely.
_DOCLING_OPEN_TEXT_LOC_RE = re.compile(r"<text>(?:<loc_\d+>){2,}[^\n]*", re.IGNORECASE) # Docling may produce malformed open <text> spans that contain multiple location tags and no real content; these add noise without value, so strip the opening tag and everything after it.
_DOCLING_BROKEN_FORMULA_TAG_RE = re.compile(r"</?formula\s*>?", re.IGNORECASE) # Some Docling versions produce broken <formula> tags that aren't properly closed or that appear in plaintext; these add noise without value, so strip them out.
_DOCLING_LONG_QUAD_RUN_RE = re.compile(r"(?:\\,\s*)?(?:\\quad\s*(?:\\,\s*)?){6,}") # Docling may produce long runs of \quad (with optional \, and whitespace) in formulas; these add noise without value, so replace runs of 6 or more with a single space.
_DOCLING_LONG_THINSPACE_RUN_RE = re.compile(r"(?:\\,\s*){12,}") # Docling may produce long runs of \, (with optional whitespace) in formulas; these add noise without value, so replace runs of 12 or more with a single space.
_DOCLING_LONG_BARE_BACKSLASH_RUN_RE = re.compile(r"(?:\\(?:\s|$)){20,}") # Docling may emit very long runs of bare "\" layout tokens; these add noise without value, so replace long runs with a single space.  
_DOCLING_HAT_ONLY_LINE_RE = re.compile(r"^(?:[\u0302\u02C6]\s*)+$") # Some Docling versions produce lines that are just hat accents (e.g., "̂") due to OCR artifacts; these add noise without value, so drop lines that are only hat characters and whitespace.
_DOCLING_LONG_ALIGN_LAYOUT_RUN_RE = re.compile(r"(?:(?:\\quad|\\,|&|\\\\)\s*){18,}") # Some parsed math figures become long runs of alignment/layout tokens (e.g., "\quad & \quad & ..."); collapse these to reduce retrieval noise.
_DOCLING_LAYOUT_TOKEN_RE = re.compile(r"(?:\\quad|&|\\\\)")


def _sanitize_docling_markdown(markdown: str) -> str:
    """Strip known Docling artifact patterns that hurt retrieval quality."""
    s = (markdown or "").replace("\r\n", "\n").replace("\r", "\n")
    s = _DOCLING_IMAGE_COMMENT_RE.sub("", s)
    s = _DOCLING_TEXT_LOC_SPAN_RE.sub("", s)
    s = _DOCLING_OPEN_TEXT_LOC_RE.sub("", s)
    s = _DOCLING_LOC_TAG_RE.sub("", s)
    s = s.replace("<text>", "").replace("</text>", "")
    s = _DOCLING_BROKEN_FORMULA_TAG_RE.sub("", s)
    s = _DOCLING_LONG_QUAD_RUN_RE.sub(" ", s)
    s = _DOCLING_LONG_THINSPACE_RUN_RE.sub(" ", s)
    s = _DOCLING_LONG_BARE_BACKSLASH_RUN_RE.sub(" ", s)
    s = _DOCLING_LONG_ALIGN_LAYOUT_RUN_RE.sub(" ", s)

    kept_lines: list[str] = []
    for ln in s.split("\n"):
        line = ln.rstrip()
        stripped = line.strip()
        # Drop standalone hat-accent noise lines (e.g., lines that are just "̂").
        if stripped and _DOCLING_HAT_ONLY_LINE_RE.fullmatch(stripped):
            continue
        if _looks_like_layout_math_noise_line(stripped):
            continue
        kept_lines.append(line)

    s = "\n".join(kept_lines)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _sanitize_formula_latex(formula: str) -> str:
    """Clean formula strings before markdown injection."""
    s = html.unescape(str(formula or ""))
    s = _DOCLING_BROKEN_FORMULA_TAG_RE.sub("", s)
    s = _DOCLING_LOC_TAG_RE.sub("", s)
    s = s.replace("<text>", "").replace("</text>", "")
    s = _DOCLING_LONG_QUAD_RUN_RE.sub(" ", s)
    s = _DOCLING_LONG_THINSPACE_RUN_RE.sub(" ", s)
    s = _DOCLING_LONG_BARE_BACKSLASH_RUN_RE.sub(" ", s)
    s = _DOCLING_LONG_ALIGN_LAYOUT_RUN_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if _looks_like_layout_math_noise_line(s):
        return ""
    return s


def _looks_like_layout_math_noise_line(line: str) -> bool:
    """Return True for lines dominated by Docling layout tokens (\\quad, &, \\\\)."""
    s = str(line or "").strip()
    if not s:
        return False

    layout_tokens = _DOCLING_LAYOUT_TOKEN_RE.findall(s)
    if len(layout_tokens) < 14:
        return False

    token_count = max(1, len(re.findall(r"\S+", s)))
    layout_ratio = len(layout_tokens) / token_count
    compact = re.sub(r"(?:\\quad|\\,|&|\\\\|\s|[{}()])", "", s)
    alnum_count = sum(ch.isalnum() for ch in compact)
    return layout_ratio >= 0.45 and alnum_count <= 40


class DoclingBackend(ParserBackend):
    name = "docling"

    def __init__(self) -> None:
        self._configure_torch_allocator()
        # Try to enable Docling enrichments (formula decoding) when available.
        self._converter = self._build_converter_with_enrichments()

    def _configure_torch_allocator(self) -> None:
        if not _env_flag("DOCLING_SET_PYTORCH_ALLOC_CONF", True):
            return
        if os.getenv("PYTORCH_CUDA_ALLOC_CONF"):
            return

        # Reduce CUDA allocator fragmentation by default when unset.
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    def _cuda_memory_cleanup(self) -> None:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
        except Exception:
            pass

    def _looks_like_oom(self, err: Exception) -> bool:
        msg = str(err).lower()
        return (
            "out of memory" in msg
            or "cuda out of memory" in msg
            or "std::bad_alloc" in msg
            or "unable to allocate" in msg
            or "cuda error: out of memory" in msg
        )

    def backend_version(self) -> str:
        try:
            from importlib.metadata import version

            return version("docling")
        except Exception:
            return "unknown"

    def _build_converter_with_enrichments(self) -> DocumentConverter:
        enable_formula = _env_flag("DOCLING_ENABLE_FORMULA_ENRICHMENT", True)
        generate_images = _env_flag("DOCLING_GENERATE_PAGE_IMAGES", False)
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

    def _page_markdown_batch(self, pdf_path: Path, start_page: int, end_page: int) -> dict[int, str]:
        result = self._converter.convert(pdf_path, page_range=(start_page, end_page))
        doc = result.document
        out: dict[int, str] = {}

        try:
            for page_no in range(start_page, end_page + 1):
                try:
                    out[page_no] = self._render_page_markdown(doc, page_no=page_no)
                except Exception as e:
                    print(
                        f"[WARN] Docling batch render failed on {pdf_path.name} page={page_no}: "
                        f"{type(e).__name__}: {e}"
                    )
                    try:
                        out[page_no] = self._page_markdown(pdf_path, page_no)
                    except Exception as e2:
                        print(
                            f"[WARN] Docling per-page fallback failed on {pdf_path.name} page={page_no}: "
                            f"{type(e2).__name__}: {e2}"
                        )
                        out[page_no] = ""
        finally:
            del doc
            self._cuda_memory_cleanup()

        return out

    def parse_pdf(self, pdf_path: Path, *, expected_pages: int | None = None) -> list[ParsedPage]:
        if expected_pages is None:
            raise ValueError("DoclingBackend requires expected_pages for deterministic page mapping.")

        total_pages = int(expected_pages)
        batch_size = max(1, int(_env_int("DOCLING_PAGE_BATCH_SIZE") or 12))
        min_batch_size = max(1, int(_env_int("DOCLING_MIN_PAGE_BATCH_SIZE") or 1))
        adaptive = _env_flag("DOCLING_ADAPTIVE_BATCHING", True)
        batch_size = max(min_batch_size, batch_size)

        page_markdown: dict[int, str] = {}
        start_page = 1
        while start_page <= total_pages:
            end_page = min(total_pages, start_page + batch_size - 1)
            try:
                page_markdown.update(self._page_markdown_batch(pdf_path, start_page, end_page))
                start_page = end_page + 1
            except Exception as e:
                is_oom = self._looks_like_oom(e)
                if adaptive and is_oom and batch_size > min_batch_size:
                    next_batch_size = max(min_batch_size, batch_size // 2)
                    print(
                        f"[WARN] Docling batch OOM on {pdf_path.name} pages={start_page}-{end_page}; "
                        f"reducing batch_size {batch_size} -> {next_batch_size}"
                    )
                    batch_size = next_batch_size
                    self._cuda_memory_cleanup()
                    continue

                print(
                    f"[WARN] Docling batch conversion failed on {pdf_path.name} pages={start_page}-{end_page}: "
                    f"{type(e).__name__}: {e}"
                )
                if is_oom:
                    self._cuda_memory_cleanup()

                # Per-page resilience without changing device policy.
                for page_no in range(start_page, end_page + 1):
                    try:
                        page_markdown[page_no] = self._page_markdown(pdf_path, page_no)
                    except Exception as e2:
                        print(
                            f"[WARN] Docling page fallback failed on {pdf_path.name} page={page_no}: "
                            f"{type(e2).__name__}: {e2}"
                        )
                        page_markdown[page_no] = ""
                    finally:
                        self._cuda_memory_cleanup()

                start_page = end_page + 1

        out: list[ParsedPage] = []
        for page_no in range(1, total_pages + 1):
            markdown = str(page_markdown.get(page_no, "") or "")
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

        out.sort(key=lambda row: int(row.get("page_number", 0)))
        return out
