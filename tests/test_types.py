from rag.types import AnswerResult, Citation, validate_answer, REFUSAL_TEXT

# refusal ok
r1 = AnswerResult(answer=REFUSAL_TEXT, citations=[])
validate_answer(r1, require_citations=True)

# non-refusal must have citations
r2 = AnswerResult(answer="something", citations=[])
try:
    validate_answer(r2, require_citations=True)
except ValueError as e:
    print("expected:", e)
