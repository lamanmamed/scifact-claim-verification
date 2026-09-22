"""Prompts and parsing for sparse/dense query rewriting."""

from __future__ import annotations

import json
import re

RELATION_WORDS = (
    "reduce", "reduces", "reduced", "increase", "increases", "increased",
    "associate", "associated", "cause", "causes", "prevent", "prevents",
    "not", "no", "does not", "may", "might",
)


def has_relation(text: str) -> bool:
    lowered = (text or "").lower()
    return any(term in lowered for term in RELATION_WORDS)


def build_controlled_rewrite_prompt(claim: str) -> str:
    return f'''You are assisting a scientific information retrieval pipeline.

Rewrite the scientific claim into two retrieval queries.
Return only valid JSON with these fields:
- sparse_query: short and keyword-focused for BM25
- dense_query: one concise sentence that preserves the claim meaning

Do not add facts or change causality, direction, negation, or uncertainty.

{{
  "sparse_query": "...",
  "dense_query": "..."
}}

Claim:
{claim}'''.strip()


def build_relation_rewrite_prompt(claim: str) -> str:
    return f'''Rewrite this scientific claim for retrieval.

Return one JSON object with:
1. sparse_query: concise keywords that explicitly preserve the main relation
2. dense_query: one sentence, at most 20 words, preserving the exact relation

Preserve causality, direction, negation, and uncertainty. Do not add new facts.

{{
  "sparse_query": "...",
  "dense_query": "..."
}}

Claim:
{claim}'''.strip()


def extract_first_json(text: str) -> dict:
    """Extract the first balanced JSON object from model output."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
    depth = 0
    start = None
    for index, char in enumerate(text):
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                candidate = text[start:index + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    start = None
    raise ValueError("No valid JSON object found")


def parse_rewrite_output(text: str, original_claim: str) -> dict[str, str]:
    try:
        parsed = extract_first_json(text)
        sparse = str(parsed.get("sparse_query", "")).strip()
        dense = str(parsed.get("dense_query", "")).strip()
        if sparse and dense:
            return {"sparse_query": sparse, "dense_query": dense, "status": "success"}
    except ValueError:
        pass
    return {"sparse_query": original_claim, "dense_query": original_claim, "status": "fallback"}
