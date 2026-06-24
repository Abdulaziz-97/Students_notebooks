"""Adaptive chunking helpers for notebook 06 — mirrors ekimetrics/adaptive-chunking paper methods."""
from __future__ import annotations

import os

# Must run before torch / sentence-transformers load (Windows + Jupyter stability).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import asyncio
import re
from functools import partial
from typing import Any, Callable

from adaptive_chunking.chunking_utils import count_tokens
from adaptive_chunking.metrics import compute_block_integrity, compute_size_compliance
from adaptive_chunking.splitters import RecursiveSplitter

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
_EMBEDDINGS: dict[str, Any] = {}
_SEMANTIC_CHUNKER: Any = None

# Paper separators (LREC 2026 adaptive-chunking replicate.py)
SEPARATORS = [
    r"(?<=\n)#{1}\s",
    r"(?<=\n)#{2}\s",
    r"(?<=\n)#{3}\s",
    r"(?<=\n)#{4}\s",
    r"(?<=\n)#{5}\s",
    r"(?<=\n)#{6}\s",
    r"(?<=\n)\s*\(?[A-Za-z0-9]{1,4}[.)]\s+",
    r"(?<=\n)\s*[-*·•●▪◦‣▸▹○◯‒–—]\s+",
    r"\n{2,}",
    r"\n",
    r"[.!?]\s+",
    r",\s+",
    r"\s+",
    r"",
]

CHUNKING_METHODS = [
    "page",
    "sentence",
    "langch_recurs_default",
    "langch_recurs_1100",
    "our_recurs_1100",
    "our_recurs_600",
    "semantic",
    "llm_regex",
]

METRIC_WEIGHTS = {
    "size_compliance": 0.5,
    "block_integrity": 0.5,
}

count_tokens_gpt = partial(count_tokens, model="gpt-4o")

# Skip local semantic embedding on very long docs (bootcamp speed; chat LLM cannot replace this step).
SEMANTIC_MAX_CHARS = int(os.getenv("RAG_SEMANTIC_MAX_CHARS", "25000"))

_LlmRegexConfig = tuple[str, str | None, str, dict[str, Any] | None]


def _llm_regex_config(model: str | None = None) -> _LlmRegexConfig | None:
    """Return (api_key, base_url, model, extra_body) for llm_regex — follows LLM_PROVIDER."""
    provider = os.getenv("LLM_PROVIDER", "deepseek").lower()

    def _deepseek() -> _LlmRegexConfig | None:
        if not os.getenv("DEEPSEEK_API_KEY"):
            return None
        return (
            os.environ["DEEPSEEK_API_KEY"],
            os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            {"thinking": {"type": "disabled"}},
        )

    def _zai() -> _LlmRegexConfig | None:
        if not os.getenv("ZAI_API_KEY"):
            return None
        return (
            os.environ["ZAI_API_KEY"],
            os.getenv("ZAI_BASE_URL", "https://api.z.ai/api/paas/v4"),
            model or os.getenv("ZAI_MODEL", "glm-5"),
            None,
        )

    def _openai() -> _LlmRegexConfig | None:
        if not os.getenv("OPENAI_API_KEY"):
            return None
        return (
            os.environ["OPENAI_API_KEY"],
            None,
            model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            None,
        )

    by_provider = {"deepseek": _deepseek, "zai": _zai, "openai": _openai}
    pick = by_provider.get(provider, _deepseek)
    if pick:
        cfg = pick()
        if cfg:
            return cfg
    for fallback in (_deepseek, _zai, _openai):
        cfg = fallback()
        if cfg:
            return cfg
    return None


def _llm_regex_provider_name(base_url: str | None) -> str:
    if base_url and "deepseek" in base_url:
        return "DeepSeek"
    if base_url and "z.ai" in base_url:
        return "z.ai"
    return "OpenAI"


def _ensure_nltk_punkt() -> None:
    import nltk

    for resource in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


def _split_sentences_preserve_text(text: str) -> list[str]:
    """NLTK sentence splits that remain exact substrings of the source (paper method)."""
    from nltk.tokenize import sent_tokenize

    _ensure_nltk_punkt()
    sentences = sent_tokenize(text)
    if not sentences:
        return [text] if text else []

    result: list[str] = []
    current_pos = 0
    for i, sentence in enumerate(sentences):
        start_pos = text.find(sentence, current_pos)
        if start_pos == -1:
            start_pos = current_pos
        if i < len(sentences) - 1:
            next_sentence = sentences[i + 1]
            next_start = text.find(next_sentence, start_pos + len(sentence))
            if next_start == -1:
                segment = text[start_pos:]
                current_pos = len(text)
            else:
                segment = text[start_pos:next_start]
                current_pos = next_start
        else:
            segment = text[start_pos:]
        result.append(segment)
    return result


def _sentence_split_nltk(text: str, sentences_per_chunk: int = 5) -> list[str]:
    """NLTK sentence splitter — same idea as adaptive-chunking SentenceSplitter(method='nltk')."""
    sentences = _split_sentences_preserve_text(text)
    if not sentences:
        return [text] if text else []
    if sentences_per_chunk <= 1:
        return sentences
    chunks: list[str] = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunks.append("".join(sentences[i : i + sentences_per_chunk]))
    return chunks


def _page_split(text: str) -> list[str]:
    """Section/page-like splits for plain text (=== headers or form-feed)."""
    parts = re.split(r"\n={3,}\n|\f", text)
    return [p for p in parts if p.strip()]


def _paragraph_split_points(text: str) -> list[int]:
    """Gold block boundaries = paragraph starts (double newline)."""
    points = [m.start() for m in re.finditer(r"\n\n+", text)]
    return sorted(set(points))


def get_embeddings(model_name: str = INDEX_EMBEDDING_MODEL):
    """Return a cached HuggingFaceEmbeddings instance (loads each model once per process)."""
    if model_name in _EMBEDDINGS:
        return _EMBEDDINGS[model_name]
    from langchain_huggingface import HuggingFaceEmbeddings

    _EMBEDDINGS[model_name] = HuggingFaceEmbeddings(model_name=model_name)
    return _EMBEDDINGS[model_name]


def _semantic_split_text(text: str) -> list[str]:
    """Semantic chunking on preserved sentence spans (scores correctly vs source text)."""
    global _SEMANTIC_CHUNKER
    if _SEMANTIC_CHUNKER is None:
        from langchain_experimental.text_splitter import SemanticChunker

        _SEMANTIC_CHUNKER = SemanticChunker(
            get_embeddings(DEFAULT_EMBEDDING_MODEL),
            breakpoint_threshold_type="percentile",
        )

    sentences = _split_sentences_preserve_text(text)
    if len(sentences) <= 1:
        return sentences if sentences else ([text] if text else [])

    stripped = [s for s in sentences if s]
    if len(stripped) <= 1:
        return [text]

    distances, _ = _SEMANTIC_CHUNKER._calculate_sentence_distances(  # noqa: SLF001
        [s.strip() for s in stripped]
    )
    if _SEMANTIC_CHUNKER.number_of_chunks is not None:
        threshold = _SEMANTIC_CHUNKER._threshold_from_clusters(distances)  # noqa: SLF001
        breakpoint_array = distances
    else:
        threshold, breakpoint_array = _SEMANTIC_CHUNKER._calculate_breakpoint_threshold(  # noqa: SLF001
            distances
        )

    breakpoints = [
        i for i, dist in enumerate(breakpoint_array) if dist > threshold
    ]

    chunks: list[str] = []
    start = 0
    for bp in breakpoints:
        end = bp + 1
        chunks.append("".join(stripped[start:end]))
        start = end
    if start < len(stripped):
        chunks.append("".join(stripped[start:]))
    return [c for c in chunks if c]


def build_sync_strategies(
    *,
    include_semantic: bool = True,
    device: str = "cpu",
) -> dict[str, Callable[[str], list[str]]]:
    """Build the 8 paper chunking methods (sync). llm_regex is async — see build_async_strategies."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    sentence_split = lambda text: _sentence_split_nltk(text, sentences_per_chunk=5)

    strategies: dict[str, Callable[[str], list[str]]] = {
        "page": _page_split,
        "sentence": sentence_split,
        "langch_recurs_default": RecursiveCharacterTextSplitter().split_text,
        "langch_recurs_1100": RecursiveCharacterTextSplitter(
            separators=SEPARATORS,
            chunk_size=1100,
            chunk_overlap=0,
            is_separator_regex=True,
            keep_separator="start",
            length_function=count_tokens_gpt,
        ).split_text,
        "our_recurs_1100": RecursiveSplitter(
            separators=SEPARATORS,
            chunk_size=1100,
            chunk_overlap=0,
            is_separator_regex=True,
            attach_separator_to="start",
            length_function=count_tokens_gpt,
            merging="to_chunk_size",
            merging_order="forward",
        ).split_text,
        "our_recurs_600": RecursiveSplitter(
            separators=SEPARATORS,
            chunk_size=600,
            chunk_overlap=0,
            is_separator_regex=True,
            attach_separator_to="start",
            length_function=count_tokens_gpt,
            merging="to_chunk_size",
            merging_order="forward",
        ).split_text,
    }

    if include_semantic:
        try:
            print("Loading semantic chunker model (sentence-transformers/all-MiniLM-L6-v2)...")
            get_embeddings(DEFAULT_EMBEDDING_MODEL)
            strategies["semantic"] = _semantic_split_text
            print("Semantic chunker ready.")
        except Exception as exc:  # noqa: BLE001 — optional method
            print(f"Skipping semantic chunker ({exc})")

    return strategies


def build_llm_regex_strategy(llm_model: str | None = None):
    """Async LLM-regex splitter — uses DeepSeek (default) or other OpenAI-compatible APIs."""
    import openai

    cfg = _llm_regex_config(llm_model)
    if cfg is None:
        raise RuntimeError("llm_regex requires DEEPSEEK_API_KEY, ZAI_API_KEY, or OPENAI_API_KEY")

    api_key, base_url, model, extra_body = cfg

    try:
        from adaptive_chunking.paper.replicate import _build_few_shot_prompt
        from adaptive_chunking.paper.splitters import LLMRegexSplitter
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "llm_regex requires optional deps from adaptive-chunking paper extras "
            f"(install stanza): {exc}"
        ) from exc

    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = openai.AsyncOpenAI(**client_kwargs)

    async def _completion(prompt: str) -> str:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }
        if extra_body:
            kwargs["extra_body"] = extra_body
        resp = await client.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content
        return content or ""

    return LLMRegexSplitter(
        base_prompt=_build_few_shot_prompt(),
        async_client_completion_func=_completion,
        count_tokens_func=count_tokens_gpt,
        context_tokens=8000,
    )


def score_chunks(text: str, chunks: list[str]) -> dict[str, float]:
    """Intrinsic metrics available without GPU coreference models."""
    split_points = _paragraph_split_points(text)
    sc = compute_size_compliance(
        chunks, max_tokens=1100, min_tokens=100, count_tokens_func=count_tokens_gpt
    )
    bi = compute_block_integrity(chunks, split_points, text)
    return {
        "size_compliance": sc if sc is not None else 0.0,
        "block_integrity": bi if bi is not None else 0.0,
    }


def weighted_score(metric_values: dict[str, float]) -> float:
    total_w = 0.0
    total = 0.0
    for name, weight in METRIC_WEIGHTS.items():
        if name in metric_values:
            total += metric_values[name] * weight
            total_w += weight
    return total / total_w if total_w else 0.0


def evaluate_strategies(
    text: str,
    strategies: dict[str, Callable[[str], list[str]]],
) -> dict[str, dict[str, Any]]:
    """Score every strategy on one document."""
    results: dict[str, dict[str, Any]] = {}
    for label, fn in strategies.items():
        try:
            if label == "semantic" and len(text) > SEMANTIC_MAX_CHARS:
                raise RuntimeError(
                    f"skipped — doc has {len(text):,} chars (semantic uses local embeddings, "
                    f"limit {SEMANTIC_MAX_CHARS:,}; set RAG_SEMANTIC_MAX_CHARS to raise)"
                )
            chunks = fn(text)
            metrics = score_chunks(text, chunks)
            results[label] = {
                "chunks": chunks,
                "metrics": metrics,
                "weighted": weighted_score(metrics),
                "n_chunks": len(chunks),
            }
        except Exception as exc:  # noqa: BLE001
            results[label] = {"error": str(exc), "weighted": -1.0, "n_chunks": 0}
    return results


def pick_best_strategy(
    text: str,
    strategies: dict[str, Callable[[str], list[str]]],
) -> tuple[str, list[str], float, dict[str, dict[str, Any]]]:
    """Return best method label, chunks, weighted score, and full evaluation table."""
    table = evaluate_strategies(text, strategies)
    valid = {k: v for k, v in table.items() if "error" not in v}
    if not valid:
        raise RuntimeError("All chunking strategies failed for this document.")
    # Tie-break equal weighted scores by preferring fewer, retrieval-friendly chunks.
    best = max(
        valid,
        key=lambda k: (valid[k]["weighted"], -valid[k]["n_chunks"]),
    )
    row = valid[best]
    return best, row["chunks"], row["weighted"], table


def run_async_coro(coro):
    """Run a coroutine from sync code — works in scripts and Jupyter (running event loop)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coro).result()


def add_llm_regex_if_available(
    strategies: dict[str, Callable[[str], list[str]]],
    llm_model: str | None = None,
) -> dict[str, Callable[[str], list[str]]]:
    """Add llm_regex when DEEPSEEK_API_KEY, ZAI_API_KEY, or OPENAI_API_KEY is set."""
    cfg = _llm_regex_config(llm_model)
    if cfg is None:
        return strategies
    try:
        splitter = build_llm_regex_strategy(llm_model)
    except RuntimeError as exc:
        print(f"llm_regex skipped — {exc}")
        return strategies

    _, base_url, model_name, _ = cfg
    provider = _llm_regex_provider_name(base_url)
    print(f"llm_regex enabled via {provider} ({model_name})")

    def _sync_llm_regex(text: str) -> list[str]:
        return run_async_coro(splitter.split_text(text))

    out = dict(strategies)
    out["llm_regex"] = _sync_llm_regex
    return out


def finalize_strategies(
    strategies: dict[str, Callable[[str], list[str]]],
    *,
    llm_model: str | None = None,
) -> dict[str, Callable[[str], list[str]]]:
    """Return strategies dict — unwraps stale async coroutines from cached Jupyter imports."""
    result = add_llm_regex_if_available(strategies, llm_model=llm_model)
    if asyncio.iscoroutine(result):
        return run_async_coro(result)
    return result
