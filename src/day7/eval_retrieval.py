"""Retrieval-only benchmark (no LLM): same corpus, same questions, same labels for every method.

Query = the ORIGINAL question text. Filter = the question's user-provided `filter` (null for most).
Hit@1 / Evidence Hit@3 use the top-3 ranking of every method. Context Evidence Rate = evidence
inside the packed context (top `ctx_k` results, `budget` characters).

    python -m src.day7.eval_retrieval --split dev                    # all methods
    python -m src.day7.eval_retrieval --split dev --methods dense,hybrid --ctx-k 5 --budget 4000
    python -m src.day7.eval_retrieval --split test --methods dense,rerank_hybrid --tag final
"""
import argparse
import statistics
import time
from datetime import datetime

from .common import BENCH, RESULTS, is_hit, load_chunks, read_json, write_json
from .context import expand_neighbors, pack_context
from .retrievers import METHODS, build


def evaluate(method: str, retriever, questions, ctx_k: int, budget: int, neighbors: int = 0, by_id=None):
    traces = []
    for q in questions:
        t0 = time.perf_counter()
        results = retriever.search(q["question"], filters=q.get("filter"), top_k=max(10, ctx_k))
        ms = (time.perf_counter() - t0) * 1000
        flags = [is_hit(c, q["gold"]) for c in results]
        selected = expand_neighbors(results[:ctx_k], by_id, neighbors) if neighbors else results[:ctx_k]
        context, ids = pack_context(selected, budget)
        ctx_flags = {c['id']: is_hit(c, q['gold']) for c in selected}
        first = next((i + 1 for i, f in enumerate(flags) if f), None)
        traces.append({
            "query_id": q["id"], "split": q["split"], "type": q.get("type"), "question": q["question"],
            "query_sent": q["question"], "filter": q.get("filter"), "method": method,
            "gold": q["gold"], "expected_answer": q["expected_answer"],
            "top3": [{"rank": c["rank"], "id": c["id"], "page": c["page"], "score": c["score"], "hit": f,
                      "stage_ranks": c.get("stage_ranks"), "text": c["text"]}
                     for c, f in zip(results[:3], flags[:3])],
            "first_hit_rank@10": first, "hit1": bool(flags[:1] and flags[0]), "hit3": any(flags[:3]),
            "context_ids": ids, "context_chars": len(context),
            "context_hit": any(ctx_flags[i] for i in ids),
            "latency_ms": round(ms, 1)})
    return traces


def summarize(traces, ctx_k, budget, neighbors=0):
    n = len(traces)
    return {"N": n, "hit1": sum(t["hit1"] for t in traces), "hit3": sum(t["hit3"] for t in traces),
            "mrr@10": round(sum(1 / t["first_hit_rank@10"] for t in traces if t["first_hit_rank@10"]) / n, 3),
            "context_hit": sum(t["context_hit"] for t in traces), "ctx_k": ctx_k, "neighbors": neighbors, "budget_chars": budget,
            "avg_context_chars": round(statistics.mean(t["context_chars"] for t in traces)),
            "median_latency_ms": round(statistics.median(t["latency_ms"] for t in traces), 1),
            "cost_usd": 0, "misses@3": [t["query_id"] for t in traces if not t["hit3"]]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=("dev", "test"), required=True)
    p.add_argument("--methods", default=",".join(METHODS))
    p.add_argument("--ctx-k", type=int, default=3)
    p.add_argument("--budget", type=int, default=2500)
    p.add_argument("--neighbors", type=int, default=0, help="add next N same-page chunks after each context chunk")
    p.add_argument("--tag", default="")
    a = p.parse_args()
    if a.split == "test" and not (RESULTS / "selection.json").exists():
        raise SystemExit("Freeze results/selection.json on dev before running test.")
    questions = [q for q in read_json(BENCH / "questions.json") if q["split"] == a.split]
    chunks, cache = load_chunks(), {}
    by_id = {c['id']: c for c in chunks}
    out_path = RESULTS / "retrieval" / "summary.json"
    summary = read_json(out_path) if out_path.exists() else {}
    for m in a.methods.split(","):
        retriever = build(m, chunks, cache)
        retriever.search("warm up", top_k=1)  # exclude model loading from latency
        traces = evaluate(m, retriever, questions, a.ctx_k, a.budget, a.neighbors, by_id)
        name = f"{m}_k{a.ctx_k}_b{a.budget}{'_n' + str(a.neighbors) if a.neighbors else ''}{'_' + a.tag if a.tag else ''}"
        write_json(RESULTS / "retrieval" / a.split / f"{name}.json", traces)
        s = summarize(traces, a.ctx_k, a.budget, a.neighbors)
        summary.setdefault(a.split, {})[name] = {"method": m, "run_at": datetime.now().isoformat(timespec="seconds"), **s}
        print(f"{a.split} {name:38s} Hit@1 {s['hit1']}/{s['N']}  Hit@3 {s['hit3']}/{s['N']}  "
              f"MRR@10 {s['mrr@10']}  CtxEvid {s['context_hit']}/{s['N']}  "
              f"ctx {s['avg_context_chars']}ch  {s['median_latency_ms']}ms  miss {s['misses@3']}")
    write_json(out_path, summary)


if __name__ == "__main__":
    main()
