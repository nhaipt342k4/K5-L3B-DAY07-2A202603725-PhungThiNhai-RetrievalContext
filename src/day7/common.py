"""Shared paths, I/O and evidence matching for the Day 7 pipeline."""
import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
# Day 6 repo sits next to this one; override with DAY6_STARTER if it lives elsewhere.
DAY6_STARTER = Path(os.environ.get(
    "DAY6_STARTER", ROOT.parent / "K5-L3B-DAY06-PhungThiNhai-2A202603725-DataIngestion" / "starter"))
CHROMA_PATH = DAY6_STARTER / "chroma_db"
DAY6_COLLECTION = "index_D_ctx800"          # best Day 6 index (strategy D), 148 chunks
EMBED_MODEL = "intfloat/multilingual-e5-small"  # same model used to build the Day 6 index
EMBED_DIM = 384

BENCH = ROOT / "benchmark"
DATA = ROOT / "data" / "day6"
CHUNKS_PATH = DATA / "chunks.json"
RESULTS = ROOT / "results"


def load_env() -> None:
    """Load ROOT/.env without overriding variables already set in the shell."""
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def key(text: str) -> str:
    """Normalisation for evidence matching: NFKC, collapse whitespace, casefold (same as Day 6)."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()


def is_hit(chunk: dict[str, Any], gold: list[dict[str, Any]]) -> bool:
    """A chunk is evidence if it matches source_id + page and contains one full gold evidence string."""
    text = key(chunk["text"])
    return any(chunk["source_id"] == g["source_id"] and chunk["page"] == g["page"] and key(g["evidence"]) in text
               for g in gold)


def load_chunks() -> list[dict[str, Any]]:
    if not CHUNKS_PATH.exists():
        raise SystemExit(f"Missing {CHUNKS_PATH}. Run: python -m src.day7.export_day6")
    return read_json(CHUNKS_PATH)


def load_cases(split: str | None = None) -> list[dict[str, Any]]:
    cases = read_json(BENCH / "cases.json")
    return [c for c in cases if split is None or c["split"] == split]
