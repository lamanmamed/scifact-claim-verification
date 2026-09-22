"""Sentence and span helpers for focused evidence extraction."""

from __future__ import annotations

import re


def split_into_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def build_sentence_offsets(text: str, sentences: list[str] | None = None) -> list[tuple[str, int, int]]:
    sentences = sentences if sentences is not None else split_into_sentences(text)
    offsets = []
    cursor = 0
    for sentence in sentences:
        start = text.find(sentence, cursor)
        if start < 0:
            start = text.find(sentence)
        if start < 0:
            continue
        end = start + len(sentence)
        offsets.append((sentence, start, end))
        cursor = end
    return offsets


def sentence_containing_span(text: str, start_char: int, end_char: int) -> str:
    if start_char < 0 or end_char < 0:
        return ""
    for sentence, start, end in build_sentence_offsets(text):
        if start_char >= start and end_char <= end:
            return sentence
    return ""


def snippet_around_span(text: str, start_char: int, end_char: int, *, window: int = 1) -> str:
    offsets = build_sentence_offsets(text)
    for index, (_, start, end) in enumerate(offsets):
        if start_char >= start and end_char <= end:
            left = max(0, index - window)
            right = min(len(offsets), index + window + 1)
            return " ".join(offsets[i][0] for i in range(left, right)).strip()
    return ""


def best_qa_span(start_logits, end_logits, context_token_indices: list[int], *, max_answer_len: int = 30) -> tuple[int, int, float]:
    """Choose the valid start/end token pair with the highest summed QA score."""
    best = (-1, -1, float("-inf"))
    context_set = set(context_token_indices)
    for start in context_token_indices:
        max_end = start + max_answer_len - 1
        for end in range(start, max_end + 1):
            if end not in context_set:
                continue
            score = float(start_logits[start]) + float(end_logits[end])
            if score > best[2]:
                best = (start, end, score)
    return best
