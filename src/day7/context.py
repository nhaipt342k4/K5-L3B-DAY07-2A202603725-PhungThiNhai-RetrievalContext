"""Context selection: deduplicate, keep rank order, pack whole chunks into a character budget."""
import re
from typing import Any


def neighbor_id(chunk_id: str, offset: int) -> str:
    """'doc:p41:c00' -> 'doc:p41:c01' (same page only; chunk IDs are page-local and ordered)."""
    m = re.match(r"^(.*:c)(\d+)$", chunk_id)
    return f"{m[1]}{int(m[2]) + offset:02d}" if m else ""


def expand_neighbors(results: list[dict[str, Any]], by_id: dict[str, dict[str, Any]], n: int = 1) -> list[dict[str, Any]]:
    """After each retrieved chunk insert its next `n` chunks on the same page (reading order).
    Motivation (dev error q11): a section title ('ĐÁP ÁN MINI TEST TUẦN 2') is retrieved but the
    answers continue in the following chunk. Neighbours are marked so traces show why they entered."""
    out, seen = [], set()
    for r in results:
        for off in range(0, n + 1):
            cid = r["id"] if off == 0 else neighbor_id(r["id"], off)
            if cid in seen or cid not in by_id:
                continue
            seen.add(cid)
            out.append(r if off == 0 else {**by_id[cid], "score": None, "rank": None,
                                           "strategy": f"neighbor_of:{r['id']}"})
    return out


def block(c: dict[str, Any], continuation_note: bool = False) -> str:
    """Source header + chunk text. With continuation_note, a neighbour chunk says which chunk it
    directly continues (dev error q11: the model did not link a title chunk to its answers)."""
    note = ""
    if continuation_note and str(c.get("strategy", "")).startswith("neighbor_of:"):
        note = f" | tiếp nối ngay sau đoạn {c['strategy'].split(':', 1)[1]}"
    return f"[{c['id']} | {c['source']} | page {c['page']}{note}]\n{c['text']}"


def pack_context(results: list[dict[str, Any]], budget: int) -> tuple[str, list[str]]:
    """Whole chunks only (never cut evidence mid-sentence); a chunk that does not fit is skipped
    and smaller later chunks may still fit. Budget unit = characters including the source header."""
    blocks, ids, seen = [], [], set()
    for c in results:
        if c["id"] in seen:
            continue
        seen.add(c["id"])
        b = block(c)
        if len("\n\n".join(blocks + [b])) <= budget:
            blocks.append(b)
            ids.append(c["id"])
    return "\n\n".join(blocks), ids
