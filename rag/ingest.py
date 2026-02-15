import os
import json
import asyncio
import nest_asyncio
from pathlib import Path
from llama_parse import LlamaParse
from pypdf import PdfReader
from dotenv import load_dotenv

nest_asyncio.apply()
load_dotenv()

RAW_DIR = Path("data/raw_pdfs")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

async def parse_and_validate(pdf_path: Path):
    print(f"🚀 Parsing: {pdf_path.name}...")
    
    # 1. Sanity Check: Get True Page Count
    try:
        true_pages = len(PdfReader(str(pdf_path)).pages)
    except Exception as e:
        print(f"⚠️  Could not read PDF {pdf_path.name}: {e}")
        return

    # 2. Use LlamaParse JSON Mode (Safer than string splitting)
    parser = LlamaParse(
        result_type="markdown",
        verbose=True,
        parsing_instruction="This is a NIST technical standard. Preserve all tables and LaTeX math."
    )
    
    # Get structured JSON directly
    json_objs = await parser.aget_json_result(str(pdf_path))
    
    # The result is a list of dicts (one per parsed document). 
    # Usually NIST docs are 1 document, so we take the first item.
    json_result = json_objs[0]

    parsed_pages = json_result["pages"]
    
    # 3. Validation: Did we lose any pages?
    if len(parsed_pages) != true_pages:
        print(f"🚨 WARNING: {pdf_path.name} has {true_pages} pages, but LlamaParse returned {len(parsed_pages)}!")
    
    # 4. Save Structured Output
    output_data = []
    for page in parsed_pages:
        output_data.append({
            "file_name": pdf_path.name,
            "page_number": page["page"],
            "content": page["text"]
        })
        
    output_filename = f"{pdf_path.stem}_parsed.json"
    with open(PROCESSED_DIR / output_filename, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)
        
    print(f"✅ Saved {len(parsed_pages)} pages to {output_filename}")

async def main():
    pdf_paths = sorted(list(RAW_DIR.glob("*.pdf")))
    for pdf_path in pdf_paths:
        await parse_and_validate(pdf_path)

if __name__ == "__main__":
    asyncio.run(main())