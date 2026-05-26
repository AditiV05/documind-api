"""Evaluation script: compares vector-only, keyword-only, and hybrid+RRF retrieval.

Run with: python eval.py
"""

from embeddings import embed_text
from supabase_client import supabase
from main import reciprocal_rank_fusion

# Document IDs for the 4-paper corpus
CORPUS1 = "ae4f7621-30e9-46a0-8bfc-0dfe2b1efe70"  # Original RAG
CORPUS2 = "11b312ca-42d5-45d5-9cff-5961f85dc06b"  # RAG Survey
CORPUS3 = "5d548aac-201b-4eb6-bfc9-b380e50a7e5e"  # Agentic RAG
CORPUS4 = "6ab1b83b-bd23-4bea-b5d0-8132eb8ecc37"  # Architectures Survey

# (question, [valid_document_ids]) — answer may legitimately appear in any listed doc
EVAL_SET = [
    ("What two types of memory does the RAG model combine?", [CORPUS1]),
    ("What seq2seq model is used as the parametric memory in RAG?", [CORPUS1]),
    ("What does RAG use as its non-parametric memory?", [CORPUS1]),
    ("What is the difference between RAG-Sequence and RAG-Token?", [CORPUS1]),
    ("What are the three RAG paradigms described in the survey?", [CORPUS2]),
    ("What are the three components of the RAG framework's foundation?", [CORPUS2, CORPUS3, CORPUS4]),
    ("What problems of LLMs does RAG aim to address?", [CORPUS1, CORPUS2]),
    ("What is Naive RAG?", [CORPUS2]),
    ("What is Agentic RAG?", [CORPUS3]),
    ("What agentic design patterns are used in Agentic RAG?", [CORPUS3]),
    ("How does Agentic RAG differ from traditional RAG?", [CORPUS3]),
    ("What limitations of traditional RAG does Agentic RAG address?", [CORPUS3]),
    ("What does the survey's high-level taxonomy of RAG architectures organize innovations around?", [CORPUS4]),
    ("What challenges does RAG introduce around retrieval quality?", [CORPUS4]),
    ("What trade-offs in RAG systems does the survey identify?", [CORPUS4]),
    ("What evaluation frameworks does the survey review?", [CORPUS2, CORPUS4]),
    # --- Keyword-stress questions (exact terms, names, numbers) ---
    ("On Q-BLEU-1, which RAG variant outperforms the other in Jeopardy question generation?", [CORPUS1]),
    ("In the human evaluation, what percentage of cases did evaluators find BART more factual than RAG?", [CORPUS1]),
    ("How many pairs of generations were used in the human evaluation of BART vs RAG-Token?", [CORPUS1]),
    ("What query expansion technique uses LLM validation to reduce hallucinations?", [CORPUS2]),
    ("What prompting method decomposes a complex question into simpler sub-questions?", [CORPUS2]),
    ("What ambiguous abbreviation could mean large language model or a law degree?", [CORPUS2]),
    ("What RAG variant integrates graph-based data structures for multi-hop reasoning?", [CORPUS3]),
    ("What are the three limitations of Graph RAG?", [CORPUS3]),
    ("Which system classifies retrieved passages as relevant, irrelevant, or counterfactual using adversarial training?", [CORPUS4]),
    ("Which named systems show that poisoned passages can act as semantic backdoors in RAG?", [CORPUS4]),
    ("What benchmark provides evaluation protocols for hallucination detection?", [CORPUS4]),
]

TOP_K = 1  # we measure hit rate within the top-3 retrieved chunks


def vector_search(query: str, k: int) -> list[dict]:
    """Vector-only retrieval."""
    embedding = embed_text(query)
    res = supabase.rpc(
        "match_chunks",
        {"query_embedding": embedding, "match_count": k},
    ).execute()
    return res.data


def keyword_search(query: str, k: int) -> list[dict]:
    """Keyword-only (FTS) retrieval."""
    res = supabase.rpc(
        "match_chunks_fts",
        {"query_text": query, "match_count": k},
    ).execute()
    return res.data


def hybrid_search(query: str, k: int) -> list[dict]:
    """Hybrid retrieval: vector + keyword fused via RRF."""
    embedding = embed_text(query)
    vec = supabase.rpc(
        "match_chunks", {"query_embedding": embedding, "match_count": k}
    ).execute()
    fts = supabase.rpc(
        "match_chunks_fts", {"query_text": query, "match_count": k}
    ).execute()
    fused = reciprocal_rank_fusion(vec.data, fts.data)
    return fused[:k]


def is_hit(results: list[dict], expected_docs: list[str]) -> bool:
    """True if any retrieved chunk came from any of the expected documents."""
    return any(chunk["document_id"] in expected_docs for chunk in results)


def run_eval():
    strategies = {
        "vector-only": vector_search,
        "keyword-only": keyword_search,
        "hybrid+RRF": hybrid_search,
    }

    hits = {name: 0 for name in strategies}
    total = len(EVAL_SET)

    print(f"\nRunning eval on {total} questions (top-{TOP_K} hit rate)\n")
    print("=" * 70)

    for i, (question, expected_docs) in enumerate(EVAL_SET, start=1):
        print(f"\nQ{i}: {question}")
        for name, search_fn in strategies.items():
            results = search_fn(question, TOP_K)
            hit = is_hit(results, expected_docs)
            if hit:
                hits[name] += 1
            print(f"  {name:14s}: {'HIT ' if hit else 'miss'}")

    print("\n" + "=" * 70)
    print(f"\nRESULTS — top-{TOP_K} retrieval hit rate:\n")
    for name in strategies:
        rate = hits[name] / total * 100
        print(f"  {name:14s}: {hits[name]}/{total}  ({rate:.1f}%)")

    # Improvement of hybrid over vector-only baseline
    vec_rate = hits["vector-only"] / total * 100
    hyb_rate = hits["hybrid+RRF"] / total * 100
    print(f"\n  hybrid+RRF improvement over vector-only: "
          f"{hyb_rate - vec_rate:+.1f} percentage points\n")


if __name__ == "__main__":
    run_eval()