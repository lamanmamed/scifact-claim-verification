"""Loading and text preparation for the SciFact corpus."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path


def build_full_text(title: str, abstract: str) -> str:
    """Join title and abstract without creating doubled punctuation."""
    title = re.sub(r"[\.:;\s]+$", "", (title or "").strip())
    abstract = (abstract or "").strip()
    if title and abstract:
        return f"{title}. {abstract}"
    return title or abstract


def make_raw_text(text: str) -> str:
    """Remove control characters while preserving scientific symbols."""
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text or "")
    return re.sub(r"\s+", " ", text).strip()


def make_dense_text(raw_text: str) -> str:
    return raw_text


def make_bm25_text(raw_text: str) -> str:
    text = (raw_text or "").lower()
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def prepare_document(record: dict) -> dict:
    full_text = build_full_text(record.get("title", ""), record.get("text", ""))
    raw_text = make_raw_text(full_text)
    return {
        "doc_id": str(record["_id"]),
        "title": record.get("title", ""),
        "abstract": record.get("text", ""),
        "raw_text": raw_text,
        "dense_text": make_dense_text(raw_text),
        "bm25_text": make_bm25_text(raw_text),
    }


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_queries(path: str | Path) -> dict[str, str]:
    return {str(row["_id"]): row["text"] for row in read_jsonl(path)}


def load_qrels(path: str | Path) -> dict[str, set[str]]:
    """Load BEIR-style TSV qrels and keep relevance scores above zero."""
    qrels: defaultdict[str, set[str]] = defaultdict(set)
    with Path(path).open(encoding="utf-8") as handle:
        header = next(handle, "")
        for line in handle:
            if not line.strip():
                continue
            qid, doc_id, score = line.rstrip("\n").split("\t")
            if int(score) > 0:
                qrels[str(qid)].add(str(doc_id))
    return dict(qrels)
