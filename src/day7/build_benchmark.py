"""Build the Day 7 benchmark files from Day 6 dev questions + new Day 7 questions.

Outputs (both committed):
  benchmark/questions.json  answerable questions in the starter schema (id, split, question,
                            expected_answer, gold=[{source_id, page, evidence}], filter)
  benchmark/cases.json      every case (answerable + unanswerable) for agent runs/metrics

Gold alternatives = every listed gold page whose chunk text really contains the evidence
quote (verified here against the exported chunks). Pages listed but not matched are
reported so the author can fix the source file; nothing is guessed.

    python -m src.day7.build_benchmark
"""
from .common import BENCH, DAY6_STARTER, is_hit, load_chunks, read_json, write_json

SOURCE_ID = "datio-grammar-30d"


def main() -> None:
    chunks = load_chunks()
    day6 = read_json(DAY6_STARTER.parent / "benchmark" / "questions.json")["questions"]
    new = read_json(BENCH / "day7_new_questions.json")["questions"]
    # Day 7 dev = the 15 Day 6 dev questions (unchanged). Day 6 test/test2 are NOT reused.
    sources = [q for q in day6 if q["split"] == "dev"] + new
    questions, cases, problems = [], [], []
    for q in sources:
        if q.get("answerable", True) is False:
            cases.append({k: q[k] for k in ("id", "split", "question", "expected_behavior", "absence_reason")}
                         | {"answerable": False, "filter": q.get("filter"), "absence_check": q.get("absence_check")})
            continue
        quotes = [q["evidence_quote"], *q.get("alternative_evidence", [])]
        gold = []
        for page in q["gold_pages"]:
            for quote in quotes:
                g = {"source_id": SOURCE_ID, "page": page, "evidence": quote}
                if any(is_hit(c, [g]) for c in chunks):
                    gold.append(g)
                else:
                    problems.append(f"{q['id']}: evidence not inside one chunk on page {page}: {quote!r}")
        if not gold:
            raise SystemExit(f"{q['id']}: no gold evidence resolves to a chunk")
        item = {"id": q["id"], "split": q["split"], "type": q.get("type"), "question": q["question"],
                "expected_answer": q["expected_answer"], "gold": gold, "filter": q.get("filter")}
        for extra in ("filter_note", "scoring_note"):
            if q.get(extra):
                item[extra] = q[extra]
        questions.append(item)
        cases.append({"id": q["id"], "split": q["split"], "question": q["question"], "answerable": True,
                      "expected_behavior": "search_and_answer", "filter": q.get("filter")})
    write_json(BENCH / "questions.json", questions)
    write_json(BENCH / "cases.json", cases)
    for split in ("dev", "test"):
        a = sum(c["answerable"] for c in cases if c["split"] == split)
        u = sum(not c["answerable"] for c in cases if c["split"] == split)
        print(f"{split}: {a} answerable + {u} unanswerable")
    for p in problems:
        print("WARN", p)


if __name__ == "__main__":
    main()
