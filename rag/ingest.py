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
