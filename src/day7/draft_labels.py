"""Draft review labels from benchmark/grading_rubric.json, then apply the reviewer's overrides.

Draft (rubric): answer_grade 2 = every answer_must regex matches and >= 1 cited chunk supports
it; 1 = partially matches, or correct but no supporting citation; 0 = wrong / abstained.
citation_support[i] = the FULL text of cited chunk i matches a support regex.
Unanswerable: abstention_correct = status abstain (the reviewer still reads the answer to make
sure it makes no unsupported claim; overrides record any disagreement).

Overrides: results/agent/reviews/overrides.json {"<split>/<config>/<query_id>": {fields..., "reason": "..."}}
Output:    results/agent/reviews/<split>/<config>.json   (input for agent_summary / agent_metrics)
           results/agent/reviews/<split>/<config>.md     (human-readable sheet: question, answer, cited text)

    python -m src.day7.draft_labels --split dev
"""
import argparse
import json
import re

from .common import BENCH, RESULTS, key, load_cases, load_chunks, read_json, write_json


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=("dev", "test"), required=True)
    a = p.parse_args()
    rubric = read_json(BENCH / "grading_rubric.json")
    expected = {q["id"]: q["expected_answer"] for q in read_json(BENCH / "questions.json")}
    cases = {c["id"]: c for c in load_cases(a.split)}
    by_id = {c["id"]: c for c in load_chunks()}
    opath = RESULTS / "agent" / "reviews" / "overrides.json"
    overrides = read_json(opath) if opath.exists() else {}
    for path in sorted((RESULTS / "agent" / a.split).glob("*.jsonl")):
        cfg = path.stem
        traces = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        reviews, sheet = [], [f"# Review sheet {a.split} / {cfg}\n"]
        for t in traces:
            qid, case = t["query_id"], cases[t["query_id"]]
            if t["status"] == "error":
                continue
            ans = key(t.get("answer") or "")
            cites = t.get("citations", [])
            if case["answerable"]:
                r = rubric[qid]
                support = [any(re.search(s, key(by_id[c]["text"])) for s in r["support"]) for c in cites]
                matched = [bool(re.search(m, ans)) for m in r["answer_must"]]
                if t["status"] != "answered":
                    grade, why = 0, "từ chối câu có đáp án"
                elif all(matched) and any(support):
                    grade, why = 2, "đủ ý theo rubric, có citation hỗ trợ"
                elif any(matched):
                    grade, why = 1, ("thiếu ý: " + ", ".join(m for m, ok in zip(r["answer_must"], matched) if not ok)
                                     if not all(matched) else "đúng ý nhưng không citation nào hỗ trợ")
                else:
                    grade, why = 0, "không khớp đáp án"
                rev = {"query_id": qid, "answer_grade": grade, "citation_support": support, "reason": "rubric: " + why}
            else:
                ok = t["status"] == "abstain"
                rev = {"query_id": qid, "citation_support": [False] * len(cites), "abstention_correct": ok,
                       "reason": "rubric: " + ("từ chối, không khẳng định nội dung ngoài tài liệu" if ok
                                               else "trả lời câu không có trong tài liệu")}
            ov = overrides.get(f"{a.split}/{cfg}/{qid}")
            if ov:
                rev.update({k: v for k, v in ov.items() if k != "reason"})
                rev["reviewer_override"] = ov["reason"]
            rev["reviewed_by"] = "Claude (đọc toàn văn câu trả lời + chunk được cite); học viên xác nhận lại"
            reviews.append(rev)
            sheet.append(f"## {qid} · {t['status']} · grade {rev.get('answer_grade', '-')} · "
                         f"abst {rev.get('abstention_correct', '-')}\n")
            sheet.append(f"- **Q:** {t['question']}\n- **Expected:** {expected.get(qid, case.get('absence_reason'))}")
            sheet.append(f"- **Queries:** {[e.get('query') for e in t['events'] if e['event'] == 'search']}")
            sheet.append(f"- **Answer:** {t.get('answer')}\n- **Label:** {rev['reason']}"
                         + (f" · override: {rev['reviewer_override']}" if rev.get("reviewer_override") else ""))
            for c, s in zip(cites, rev["citation_support"]):
                sheet.append(f"  - `{c}` support={s}: {by_id[c]['text'][:400]}")
            sheet.append("")
        write_json(RESULTS / "agent" / "reviews" / a.split / f"{cfg}.json", reviews)
        (RESULTS / "agent" / "reviews" / a.split / f"{cfg}.md").write_text("\n".join(sheet), encoding="utf-8")
        g = [r["answer_grade"] for r in reviews if "answer_grade" in r]
        print(f"{cfg}: grades 0/1/2 = {g.count(0)}/{g.count(1)}/{g.count(2)}  "
              f"cite {sum(sum(r['citation_support']) for r in reviews)}/{sum(len(r['citation_support']) for r in reviews)}")


if __name__ == "__main__":
    main()
