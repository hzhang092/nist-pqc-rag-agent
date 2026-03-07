# Week 3 Progress

## 2026-03-07 - Prose-spacing fix for identifier normalization

### Problem
The parser output in `data/processed/*_parsed.json` was removing the space before a word when the previous token ended with `.`.

Observed regressions included:
- `U.S.Department`
- `Gina M.Raimondo`
- `Laurie E.Locascio`
- sentence joins such as `channel.A shared secret key`
- numbered prose joins such as `4.Approving Authority.`

This came from the identifier-normalization feature that was introduced to preserve dotted PQC keywords such as `ML-KEM.Decaps` and `ML-KEM.ParamSets`.

### Root cause
The issue was not in Docling markdown extraction itself.

- `markdown` in parsed artifacts was already correct.
- The corruption happened in `rag/parsers/base.py` when `markdown_to_text()` called `normalize_identifier_like_spans()` from `rag/text_normalize.py`.
- The old implementation used a span-wide regex that compacted any alnum `.` alnum pattern, which was too broad for normal prose.

### Implementation
Updated `rag/text_normalize.py` to use pairwise, separator-aware normalization instead of span-wide compaction.

Behavior after the fix:
- keep escaped-separator cleanup for `\\_`, `\\.`, `\\-`
- keep spacing compaction for `_` and `-` between identifier tokens
- compact unescaped `.` only when it matches one of these supported identifier cases:
  - numeric refs such as `3 . 3 -> 3.3`
  - hyphen/underscore identifiers such as `ML-KEM . Decaps -> ML-KEM.Decaps`
  - all-caps dotted chains such as `NIST . FIPS . 203 -> NIST.FIPS.203`
- preserve normal prose spacing after periods

No change was needed in `rag/parsers/docling_backend.py`.

### Regression coverage
Added tests for:
- positive normalization cases in `tests/test_text_normalize.py`
- negative prose-preservation cases in `tests/test_text_normalize.py`
- parser-layer `markdown_to_text()` regression in `tests/test_parsers_base.py`

### Verification
Verified with:
- `conda run -n pyt python -m pytest tests/test_text_normalize.py tests/test_parsers_base.py tests/test_docling_backend.py`

Result:
- `10 passed`

Targeted Docling parse verification on FIPS 203 confirmed:
- page 1 contains `U.S. Department of Gina M. Raimondo` and `Laurie E. Locascio`
- page 3 contains `channel. A shared secret key`
- page 4 contains `4. Approving Authority. Secretary of Commerce.`

### Status
- code fix: complete
- regression tests: complete
- full artifact rebuild / full eval rerun: not completed in this update
