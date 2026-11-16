"""Summarise agent runs: automatic numbers from traces + human/LLM-assisted labels.

Automatic (from traces + gold, computed AFTER the run; gold never reaches the agent):
  tool use, searches per case, Context Evidence Rate of the final context, latency, tokens, cost.
Labels (results/agent/reviews/<split>/<config>.json, one item per case):
  {query_id, answer_grade 0-2 (answerable), citation_support [bool per citation],
   abstention_correct (unanswerable), reason}
Aggregated with starter/agent_metrics.py so the definitions match the assignment.

    python -m src.day7.agent_summary --split dev
"""
import argparse
import json
import statistics
import sys
from collections import Counter

from .common import BENCH, RESULTS, ROOT, is_hit, load_cases, load_chunks, read_json, write_json

sys.path.insert(0, str(ROOT / "starter"))
from agent_metrics import aggregate  # noqa: E402


def load_traces(split, config):
    path = RESULTS / "agent" / split / f"{config}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def auto_metrics(traces, gold_by_id, by_id):
    ok = [t for t in traces if t["status"] != "error"]
    ans = [t for t in ok if t["query_id"] in gold_by_id]
    ctx_hit = [t["query_id"] for t in ans
               if any(is_hit(by_id[c], gold_by_id[t["query_id"]]) for c in t["final_context_ids"])]
    return {
        "runs": len(traces), "errors": len(traces) - len(ok),
        "tool_use": f"{sum(t['n_searches'] > 0 for t in ok)}/{len(ok)}",
        "searches_distribution": dict(sorted(Counter(t["n_searches"] for t in ok).items())),
        "mean_searches": round(statistics.mean(t["n_searches"] for t in ok), 2) if ok else None,
        "mean_llm_calls": round(statistics.mean(t["n_llm_calls"] for t in ok), 2) if ok else None,
        "context_evidence": f"{len(ctx_hit)}/{len(ans)}",
        "context_evidence_missing": sorted(set(t["query_id"] for t in ans) - set(ctx_hit)),
        "mean_context_chars": round(statistics.mean(t["context_chars"] for t in ok)) if ok else None,
        "answered": sum(t["status"] == "answered" for t in ok),
        "abstained": sum(t["status"] == "abstain" for t in ok),
        "stop_reasons": dict(Counter(t["stop_reason"] for t in ok)),
        "mean_latency_s": round(statistics.mean(t["latency_s"] for t in ok), 2) if ok else None,
        "tokens_in": sum(t["tokens_in"] for t in ok), "tokens_out": sum(t["tokens_out"] for t in ok),
        "cost_usd_est_paid_tier": round(sum(t["cost_usd_est"] for t in ok), 4), "cost_usd_actual": 0,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=("dev", "test"), required=True)
    p.add_argument("--configs", default="A0_dense,A1_rerank_n1,A2_hybrid_n1,A3_rerank_n1_1s,A4_rerank_n1_link")
    a = p.parse_args()
    cases = load_cases(a.split)
    gold_by_id = {q["id"]: q["gold"] for q in read_json(BENCH / "questions.json") if q["split"] == a.split}
    by_id = {c["id"]: c for c in load_chunks()}
    out = {}
    for cfg in a.configs.split(","):
        traces = load_traces(a.split, cfg)
        if not traces:
            continue
        res = {"auto": auto_metrics(traces, gold_by_id, by_id)}
        rpath = RESULTS / "agent" / "reviews" / a.split / f"{cfg}.json"
        if rpath.exists():
            res["labels"] = aggregate(cases, traces, read_json(rpath))
        out[cfg] = res
        m = res["auto"]
        line = (f"{cfg:16s} ctxEvid {m['context_evidence']:6s} answered {m['answered']} abstain {m['abstained']} "
                f"searches {m['searches_distribution']} lat {m['mean_latency_s']}s tok {m['tokens_in']}/{m['tokens_out']}")
        if "labels" in res:
            L = res["labels"]
            g = L["grounded_answer"]
            line += (f" | grade {g['0']}/{g['1']}/{g['2']} mean {g['mean']:.2f} cite {L['citation_validity']['n']}/"
                     f"{L['citation_validity']['N']} abst {L['abstention']['n']}/{L['abstention']['N']} "
                     f"falseAbst {L['false_abstention']['n']}/{L['false_abstention']['N']}")
        print(line)
    write_json(RESULTS / "agent" / f"summary_{a.split}.json", out)


if __name__ == "__main__":
    main()
