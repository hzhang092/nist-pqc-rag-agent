from pathlib import Path
import subprocess, json, datetime

QUESTIONS = Path("eval/day2/questions.txt")
OUT_DIR = Path("runs/day2_baseline/2026-02-17")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def run_one(q: str) -> dict:
    # Try JSON mode first (if your rag.ask supports it). If not, we still capture stdout.
    cmd_json = ["python", "-m", "rag.ask", q, "--json"]
    cmd_txt  = ["python", "-m", "rag.ask", q]

    p = subprocess.run(cmd_json, capture_output=True, text=True)
    if p.returncode == 0:
        try:
            return {"question": q, "mode": "json", "output": json.loads(p.stdout)}
        except Exception:
            return {"question": q, "mode": "json-butfallback", "stdout": p.stdout, "stderr": p.stderr}
    else:
        p2 = subprocess.run(cmd_txt, capture_output=True, text=True)
        return {"question": q, "mode": "text", "stdout": p2.stdout, "stderr": p2.stderr, "returncode": p2.returncode}

def main():
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_jsonl = OUT_DIR / f"results_{ts}.jsonl"

    questions = [line.strip() for line in QUESTIONS.read_text(encoding="utf-8").splitlines() if line.strip()]
    with out_jsonl.open("w", encoding="utf-8") as f:
        for i, q in enumerate(questions, 1):
            print(f"Running Q{i}/{len(questions)}: {q[:50]}...")
            r = run_one(q)
            r["i"] = i
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("Wrote:", out_jsonl)

if __name__ == "__main__":
    main()
