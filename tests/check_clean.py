import json, random
path="data/processed/pages_clean.jsonl"
rows=[json.loads(x) for x in open(path, encoding="utf-8")]
for p in random.sample(rows, 3):
    print(p.get("doc_id"), p.get("page_number") or p.get("page"))
    print(p["text_clean"][:600])
    print("-"*80)
