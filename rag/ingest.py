import os
import json
import asyncio
import nest_asyncio
from llama_parse import LlamaParse
from dotenv import load_dotenv

# Apply nest_asyncio to allow nested event loops (common in notebooks/scripts)
nest_asyncio.apply()

# Load API key from .env
load_dotenv()

# Directory Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw_pdfs")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# Ensure processed directory exists
os.makedirs(PROCESSED_DIR, exist_ok=True)

async def parse_pdf(file_name):
    file_path = os.path.join(RAW_DIR, file_name)
    print(f"🚀 Parsing: {file_name}...")

    # Initialize parser with specific instructions for NIST docs
    # "result_type='markdown'" is crucial for preserving Math tables in FIPS docs
    parser = LlamaParse(
        result_type="markdown",
        verbose=True,
        language="en",
        # Optional: Add parsing instruction to help the model understand the context
        parsing_instruction="This is a NIST technical standard containing cryptographic algorithms, math formulas, and tables. Preserve all mathematical notation in LaTeX format."
    )

    # valid method to get page-level details is using the JSON output mode
    json_objs = await parser.aget_json_result(file_path)
    
    # The result is a list of dicts (one per parsed document). 
    # Usually NIST docs are 1 document, so we take the first item.
    json_result = json_objs[0] 

    output_data = []
    
    # Iterate through the 'pages' in the JSON response
    for page in json_result["pages"]:
        page_num = page["page"]
        content = page["text"]  # This is the markdown content for that specific page
        
        output_data.append({
            "file_name": file_name,
            "page_number": page_num,
            "content": content
        })

    # Save to processed folder
    output_filename = f"{os.path.splitext(file_name)[0]}_parsed.json"
    output_path = os.path.join(PROCESSED_DIR, output_filename)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)
        
    print(f"✅ Saved processed data to: {output_filename}")

async def main():
    # Get all PDF files from the raw directory
    pdf_files = [f for f in os.listdir(RAW_DIR) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print("No PDFs found in data/raw_pdfs/")
        return

    # Process files (sequentially to avoid rate limits on free tier, or gather for parallel)
    for pdf_file in pdf_files:
        await parse_pdf(pdf_file)

if __name__ == "__main__":
    asyncio.run(main())