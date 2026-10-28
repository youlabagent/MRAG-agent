"""Ask wrapper: retrieval + an Ollama LLM with citation enforcement."""
from __future__ import annotations

import json
from typing import List, Optional, Sequence
import urllib.request

from .retriever import Passage, Retriever


SYSTEM = """You are a careful biomedical research assistant.
You answer ONLY using the provided context passages.
Rules:
1. Cite every factual claim inline as (PMID:XXXXX) using the PMIDs given.
2. If the passages do not support a claim, write "no evidence in retrieved passages".
3. Never invent PMIDs. Never use prior knowledge if not in passages.
4. Be concise. Prefer mechanistic explanations.
5. If passages conflict, surface the conflict explicitly.
"""

USER_TEMPLATE = """Question: {question}

Context passages (numbered):
{context}

Answer the question following the rules. End with a JSON block listing the PMIDs you used in the form: {{"used_pmids": ["...", "..."]}}."""


def _format_context(passages: List[Passage], max_chars: int = 1200) -> str:
    blocks = []
    for i, p in enumerate(passages, 1):
        cite = p.cite()
        sec = f" [{p.section}]" if p.section else ""
        yr = f" ({p.year})" if p.year else ""
        title = (p.title or "")[:160]
        body = (p.text or "")[:max_chars]
        blocks.append(f"[{i}] {cite}{yr}{sec} {title}\n{body}")
    return "\n\n".join(blocks)


def _glm_chat(
    messages: list,
    endpoint: str,
    model: str,
    temperature: float = 0.2,
    max_tokens: int = 800,
    timeout: int = 240,
    think: bool = False,
) -> dict:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": think,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{endpoint}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def ask(
    question: str,
    retriever: Optional[Retriever] = None,
    top_k: int = 8,
    cancer: Optional[Sequence[str] | str] = None,
    year_min: Optional[int] = None,
    endpoint: str = "http://127.0.0.1:11500",
    model: str = "batiai/qwen3.6-27b:q6",
    temperature: float = 0.2,
    max_tokens: int = 800,
    think: bool = False,
) -> dict:
    """Return dict with: answer, used_pmids, retrieved (list of passages), latency."""
    import time
    if retriever is None:
        retriever = Retriever()

    t0 = time.time()
    passages = retriever.search(question, top_k=top_k, cancer=cancer, year_min=year_min)
    t_retr = time.time() - t0

    context = _format_context(passages)
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER_TEMPLATE.format(question=question, context=context)},
    ]

    t1 = time.time()
    resp = _glm_chat(messages, endpoint=endpoint, model=model,
                      temperature=temperature, max_tokens=max_tokens, think=think)
    t_gen = time.time() - t1

    answer = resp.get("message", {}).get("content", "")
    used = []
    try:
        # last JSON block in answer
        import re
        m = re.findall(r'\{[^{}]*"used_pmids"[^{}]*\}', answer)
        if m:
            used = json.loads(m[-1]).get("used_pmids", [])
    except Exception:
        pass

    return {
        "question": question,
        "answer": answer,
        "used_pmids": used,
        "retrieved": [p.to_dict() for p in passages],
        "n_retrieved": len(passages),
        "latency_retrieval_s": round(t_retr, 2),
        "latency_generation_s": round(t_gen, 2),
        "endpoint": endpoint,
        "model": model,
    }
