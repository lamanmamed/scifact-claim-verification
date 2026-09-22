"""Evidence-grounded verdict prompting and parsing."""

from __future__ import annotations

import re

VERDICTS = {"SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"}


def format_evidence(documents: list[dict], *, max_chars: int = 1000) -> str:
    blocks = []
    for index, document in enumerate(documents, start=1):
        text = str(document.get("text", "")).strip()[:max_chars]
        blocks.append(f"[{index}] {text}")
    return "\n".join(blocks)


def build_verdict_prompt(claim: str, documents: list[dict]) -> str:
    evidence = format_evidence(documents)
    return f'''Claim:
{claim}

Evidence:
{evidence}

Tasks:
1. Summarize each evidence item in one sentence.
2. Decide SUPPORTS, REFUTES, or NOT ENOUGH INFO using only the evidence above.
3. Explain the decision and cite evidence as [1], [2], etc.
4. If the claim is refuted, identify the contradictory evidence.

Output exactly these sections:
Verdict: SUPPORTS / REFUTES / NOT ENOUGH INFO
Evidence Summary:
Reasoning:
Counter Evidence:'''.strip()


def parse_verdict_output(response: str) -> dict[str, str]:
    match = re.search(r"Verdict:\s*(SUPPORTS|REFUTES|NOT ENOUGH INFO)", response or "", re.IGNORECASE)
    verdict = "UNKNOWN"
    if match:
        verdict = match.group(1).upper().replace("NOT ENOUGH INFO", "NOT_ENOUGH_INFO")

    def section(name: str) -> str:
        found = re.search(
            rf"{re.escape(name)}:\s*(.*?)(?=\n[A-Z][^\n]*:|$)",
            response or "",
            flags=re.DOTALL | re.IGNORECASE,
        )
        return found.group(1).strip() if found else ""

    return {
        "verdict": verdict,
        "evidence_summary": section("Evidence Summary"),
        "reasoning": section("Reasoning"),
        "counter_evidence": section("Counter Evidence"),
    }
