"""
PDF Ingestion and Parsing Script.

This script is responsible for processing raw PDF documents from the `data/raw_pdfs`
directory using the LlamaParse library. It extracts the text content from each
page, including tables and mathematical formulas, and structures it into a
JSON format.

The key steps are:
1.  Identify all PDF files in the source directory.
2.  For each PDF, use LlamaParse to extract content in markdown format.
3.  Perform a sanity check to ensure the number of parsed pages matches the
    actual page count of the PDF.
4.  Save the structured output for each PDF into its own `_parsed.json` file
    in the `data/processed` directory for easy debugging.
5.  Append the parsed page data to a unified `pages.jsonl` file, which serves
    as the input for the next stage of the RAG pipeline (cleaning and chunking).

This script requires a `LLAMA_CLOUD_API_KEY` to be set in a `.env` file to
authenticate with the LlamaParse API.
"""
import json
from pathlib import Path
from llama_parse import LlamaParse
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

RAW_DIR = Path("data/raw_pdfs")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def parse_and_validate(pdf_path: Path, pages_jsonl_f):
    """
    Parses a single PDF file using LlamaParse and validates the output.

    This function performs the following actions:
    1.  Uses `pypdf` to get the true page count for validation.
    2.  Invokes `LlamaParse` to extract structured content (markdown).
    3.  Compares the number of parsed pages against the true count and warns
        on mismatch.
    4.  Saves the detailed parsed output for the single PDF to a dedicated
        JSON file for inspection.
    5.  Writes each page's data to the unified JSONL file for the main pipeline.

    Args:
        pdf_path: The path to the PDF file to process.
        pages_jsonl_f: An open file handle to the `pages.jsonl` file for writing.
    """
    print(f"🚀 Parsing: {pdf_path.name}...")

    # 1) True page count sanity check
    true_pages = len(PdfReader(str(pdf_path)).pages)

    # 2) JSON mode (sync)
    parser = LlamaParse(
        result_type="markdown",
        verbose=True,
        parsing_instruction="This is a NIST technical standard. Preserve all tables and LaTeX math."
    )
    json_objs = parser.get_json_result(str(pdf_path))
    json_result = json_objs[0]
    parsed_pages = json_result["pages"]

    # 3) Validate page count
    if len(parsed_pages) != true_pages:
        print(f"🚨 WARNING: {pdf_path.name} has {true_pages} pages, but LlamaParse returned {len(parsed_pages)}!")

    # 4a) Save per-PDF structured output (nice for debugging)
    output_data = []
    for page in parsed_pages:
        output_data.append({
            "doc_id": pdf_path.stem,
            "source_path": str(pdf_path.as_posix()),
            "page_number": page.get("page"),
            "text": page.get("text", "")
        })

    output_filename = f"{pdf_path.stem}_parsed.json"
    with open(PROCESSED_DIR / output_filename, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # 4b) ALSO write unified pages.jsonl for the pipeline
    for rec in output_data:
        pages_jsonl_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"✅ Saved {len(parsed_pages)} pages to {output_filename}")

def main():
    """
    Main function to find and process all PDFs in the raw data directory.

    It locates all `.pdf` files, opens the output `pages.jsonl` file, and
    iterates through the PDFs, calling `parse_and_validate` for each one.
    """
    pdf_paths = sorted(RAW_DIR.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in {RAW_DIR}")

    pages_jsonl_path = PROCESSED_DIR / "pages.jsonl"
    with open(pages_jsonl_path, "w", encoding="utf-8") as pages_jsonl_f:
        for pdf_path in pdf_paths:
            parse_and_validate(pdf_path, pages_jsonl_f)

    print(f"✅ Wrote unified dataset: {pages_jsonl_path}")

if __name__ == "__main__":
    main()
