import json

import pytest

from eval.dataset import load_questions, write_questions


def test_load_questions_valid_and_normalized(tmp_path):
    dataset = tmp_path / "q.jsonl"
    rows = [
        {
            "qid": "q001",
            "question": "What is ML-KEM?",
            "answerable": True,
            "gold": [
                {"doc_id": "NIST.FIPS.203", "start_page": 9, "end_page": 9},
                {"doc_id": "NIST.FIPS.203", "start_page": 8, "end_page": 8},
            ],
        },
        {
            "qid": "q002",
            "question": "What does NIST say about PQC for Wi-Fi 9?",
            "answerable": False,
            "gold": [],
        },
    ]
    write_questions(dataset, rows)

    loaded = load_questions(dataset)
    assert len(loaded) == 2
    assert loaded[0]["gold"] == [
        {"doc_id": "NIST.FIPS.203", "start_page": 8, "end_page": 8},
        {"doc_id": "NIST.FIPS.203", "start_page": 9, "end_page": 9},
    ]


def test_load_questions_rejects_duplicate_qids(tmp_path):
    dataset = tmp_path / "q.jsonl"
    rows = [
        {"qid": "q001", "question": "a", "answerable": True, "gold": [{"doc_id": "D", "start_page": 1, "end_page": 1}]},
        {"qid": "q001", "question": "b", "answerable": False, "gold": []},
    ]
    dataset.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate qid"):
        load_questions(dataset)


def test_load_questions_allows_unlabeled_when_flag_enabled(tmp_path):
    dataset = tmp_path / "q.jsonl"
    rows = [
        {"qid": "q001", "question": "a", "answerable": True, "gold": []},
    ]
    dataset.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    with pytest.raises(ValueError, match="requires at least one gold span"):
        load_questions(dataset, require_labeled=True)

    loaded = load_questions(dataset, require_labeled=False)
    assert len(loaded) == 1


def test_load_questions_sorts_by_qid_deterministically(tmp_path):
    dataset = tmp_path / "q.jsonl"
    rows = [
        {"qid": "q10", "question": "ten", "answerable": False, "gold": []},
        {"qid": "q2", "question": "two", "answerable": False, "gold": []},
        {"qid": "q1", "question": "one", "answerable": False, "gold": []},
    ]
    dataset.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    loaded = load_questions(dataset, require_labeled=False)
    assert [r["qid"] for r in loaded] == ["q1", "q2", "q10"]
