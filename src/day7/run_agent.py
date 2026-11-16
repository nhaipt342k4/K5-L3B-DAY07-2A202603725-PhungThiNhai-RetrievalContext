"""Run the real LLM agent for one configuration on one split; one JSON line per case.

    python -m src.day7.run_agent --config A0_dense --split dev
    python -m src.day7.run_agent --config A1_rerank_n1 --split dev --ids q11,u-d01
Re-running resumes (cases already in the output file with status != error are skipped);
pass --fresh to start over. The agent only receives the question text and the user-provided
filter; gold evidence / expected answers are never passed in.
"""
import argparse
import json
import time
from datetime import datetime

from .agent import run_agent
from .common import RESULTS, load_cases, load_chunks
from .llm import ChatClient, LLMError
from .retrievers import build

# Frozen agent configurations (method = retrieval behind the search_documents tool).
CONFIGS = {
    "A0_dense":     {"method": "dense",         "top_k": 3, "neighbors": 0, "budget": 2500, "max_searches": 3,
                     "note": "baseline: Day 6 dense retrieval, Day 6-like context (top-3, 2500 chars)"},
    "A1_rerank_n1": {"method": "rerank_dense",  "top_k": 3, "neighbors": 1, "budget": 4500, "max_searches": 3,
                     "note": "dense top-20 -> cross-encoder rerank -> top-3 + next same-page chunk, 4500 chars"},
    "A2_hybrid_n1": {"method": "hybrid",        "top_k": 3, "neighbors": 1, "budget": 4500, "max_searches": 3,
                     "note": "RRF(dense, BM25) top-3 + next same-page chunk, 4500 chars (cheap, no reranker)"},
    "A3_rerank_n1_1s": {"method": "rerank_dense", "top_k": 3, "neighbors": 1, "budget": 4500, "max_searches": 1,
                        "note": "as A1 but only ONE search allowed: measures the value of multi-step search"},
    "A4_rerank_n1_link": {"method": "rerank_dense", "top_k": 3, "neighbors": 1, "budget": 4500, "max_searches": 3,
                          "continuation_note": True,
                          "note": "as A1 + neighbour header says which chunk it continues (fix for dev error q11)"},
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", choices=CONFIGS, required=True)
    p.add_argument("--split", choices=("dev", "test"), required=True)
    p.add_argument("--ids", help="comma-separated subset of case IDs")
    p.add_argument("--fresh", action="store_true")
    a = p.parse_args()
    if a.split == "test" and not (RESULTS / "selection.json").exists():
        raise SystemExit("Freeze results/selection.json on dev before running test.")
    cfg = CONFIGS[a.config]
    cases = load_cases(a.split)
    if a.ids:
        wanted = set(a.ids.split(","))
        cases = [c for c in cases if c["id"] in wanted]
    out = RESULTS / "agent" / a.split / f"{a.config}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if out.exists() and not a.fresh:
        for line in out.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[r["query_id"]] = r
    chunks = load_chunks()
    by_id = {c["id"]: c for c in chunks}
    retriever = build(cfg["method"], chunks)
    retriever.search("warm up", top_k=1)
    llm = ChatClient()
    print(f"{a.config} on {a.split}: model={llm.model} {cfg}")
    for case in cases:
        if case["id"] in done and done[case["id"]]["status"] != "error":
            continue
        t0 = time.perf_counter()
        try:
            r = run_agent(case["question"], llm, retriever, chunks_by_id=by_id, filters=case.get("filter"),
                          top_k=cfg["top_k"], neighbors=cfg["neighbors"], budget=cfg["budget"],
                          max_searches=cfg["max_searches"], continuation_note=cfg.get("continuation_note", False))
        except LLMError as exc:  # provider error: recorded separately, never replaced by a fake answer
            r = {"status": "error", "error": str(exc), "events": [], "citations": [],
                 "latency_s": round(time.perf_counter() - t0, 2)}
        rec = {"query_id": case["id"], "split": a.split, "config": a.config, "model": llm.model,
               "question": case["question"], "filter": case.get("filter"), **cfg,
               "run_at": datetime.now().isoformat(timespec="seconds"), **r}
        done[case["id"]] = rec
        out.write_text("".join(json.dumps(done[k], ensure_ascii=False) + "\n" for k in done), encoding="utf-8")
        ans = (r.get("answer") or r.get("error") or "").replace("\n", " ")[:110]
        print(f"  {case['id']:6s} {r['status']:8s} searches={r.get('n_searches')} calls={r.get('n_llm_calls')} "
              f"{r.get('latency_s')}s | {ans}", flush=True)


if __name__ == "__main__":
    main()
