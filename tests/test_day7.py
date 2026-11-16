"""Offline checks for the Day 7 code (fake LLM/retriever only test control flow, not quality)."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.day7.agent import run_agent  # noqa: E402
from src.day7.common import is_hit  # noqa: E402
from src.day7.context import expand_neighbors, neighbor_id, pack_context  # noqa: E402
from src.day7.retrievers import BM25Retriever, HybridRetriever  # noqa: E402


def chunk(cid, text, page=1, **meta):
    return {"id": cid, "text": text, "source_id": "doc", "source": "doc.pdf", "page": page, "metadata": meta}


CHUNKS = [chunk("doc:p01:c00", "ĐÁP ÁN MINI TEST TUẦN 2", week=2),
          chunk("doc:p01:c01", "NGÀY 14 1A. 2B. 6 Does he play football?", week=2),
          chunk("doc:p02:c00", "Trạng từ tần suất đứng trước động từ thường", 2, week=1)]
BY_ID = {c["id"]: c for c in CHUNKS}


class FakeRetriever:
    name = "fake"

    def __init__(self):
        self.calls = []

    def search(self, query, *, filters=None, top_k=3):
        self.calls.append((query, filters, top_k))
        return [{**CHUNKS[0], "score": 1.0, "rank": 1}]


class FakeLLM:
    def __init__(self, messages):
        self.script, self.seen = list(messages), []

    def chat(self, messages, tools=None, tool_choice=None):
        self.seen.append((tool_choice, [m.get("role") for m in messages]))
        msg = self.script.pop(0)
        return {"message": msg, "finish_reason": "stop", "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                "cost_usd_est": 0.0, "latency_s": 0.0, "attempts": 1}


def tool_call(query):
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": "t1", "type": "function", "function": {"name": "search_documents", "arguments": json.dumps({"query": query})}}]}


class Day7Tests(unittest.TestCase):
    def test_evidence_matching_is_whitespace_and_case_insensitive(self):
        gold = [{"source_id": "doc", "page": 1, "evidence": "ngày 14  1a. 2b."}]
        self.assertTrue(is_hit(CHUNKS[1], gold))
        self.assertFalse(is_hit(CHUNKS[1], [{**gold[0], "page": 2}]))

    def test_neighbor_expansion_same_page_only(self):
        self.assertEqual(neighbor_id("doc:p01:c00", 1), "doc:p01:c01")
        out = expand_neighbors([CHUNKS[0], CHUNKS[2]], BY_ID, 1)
        self.assertEqual([c["id"] for c in out], ["doc:p01:c00", "doc:p01:c01", "doc:p02:c00"])
        _, ids = pack_context(out, budget=10_000)
        self.assertEqual(len(ids), 3)

    def test_bm25_filter_and_hybrid_rrf(self):
        bm25 = BM25Retriever(CHUNKS)
        self.assertEqual(bm25.search("trạng từ tần suất", top_k=1)[0]["id"], "doc:p02:c00")
        self.assertEqual(bm25.search("trạng từ tần suất", filters={"week": 2}), [])

        class Dense:
            name = "dense"
            def search(self, q, *, filters=None, top_k=3):
                return [{**CHUNKS[1], "score": .9, "rank": 1}, {**CHUNKS[2], "score": .8, "rank": 2}]
        top = HybridRetriever(Dense(), bm25).search("trạng từ tần suất", top_k=2)
        self.assertEqual(top[0]["id"], "doc:p02:c00")  # rank 2 + rank 1 beats rank 1 alone
        self.assertEqual(top[0]["stage_ranks"], {"dense": 2, "bm25": 1})

    def test_agent_searches_cites_and_passes_filter(self):
        llm = FakeLLM([tool_call("mini test tuần 2"),
                       {"role": "assistant", "content": "Does he play football? [doc:p01:c01]"}])
        r = FakeRetriever()
        res = run_agent("Đáp án câu 6?", llm, r, chunks_by_id=BY_ID, filters={"week": 2}, neighbors=1)
        self.assertEqual(res["status"], "answered")
        self.assertEqual(res["citations"], ["doc:p01:c01"])  # neighbour chunk entered the context
        self.assertEqual(r.calls[0][1], {"week": 2})
        self.assertEqual(llm.seen[0][0], "required")

    def test_citation_outside_context_or_abstain_text(self):
        for text in ("Sai [doc:p09:c00]", "KHÔNG ĐỦ BẰNG CHỨNG: không có bài này."):
            llm = FakeLLM([tool_call("x"), {"role": "assistant", "content": text}])
            self.assertEqual(run_agent("q", llm, FakeRetriever(), chunks_by_id=BY_ID)["status"], "abstain")

    def test_search_limit_forces_final_answer(self):
        llm = FakeLLM([tool_call("a"), tool_call("b"), {"role": "assistant", "content": "x [doc:p01:c00]"}])
        r = FakeRetriever()
        res = run_agent("q", llm, r, chunks_by_id=BY_ID, max_searches=1)
        self.assertEqual(len(r.calls), 1)
        self.assertEqual(llm.seen[-1][0], "none")
        self.assertTrue(res["forced_final"])
        self.assertEqual(res["status"], "answered")

    def test_context_budget_not_multiplied(self):
        llm = FakeLLM([tool_call("a"), tool_call("b"), {"role": "assistant", "content": "x [doc:p01:c00]"}])
        res = run_agent("q", llm, FakeRetriever(), chunks_by_id=BY_ID, budget=200, max_searches=2)
        self.assertLessEqual(res["context_chars"], 200)
        self.assertEqual(res["final_context_ids"], ["doc:p01:c00"])


if __name__ == "__main__":
    unittest.main()
