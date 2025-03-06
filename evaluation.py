from typing import List

from ragas.metrics import MultiModalRelevance, MultiModalFaithfulness
from ragas import evaluate, EvaluationDataset

def compute_mrr_at_k(
    all_retrieved_pages: List[List[int]], 
    all_real_pages: List[int], 
    k: int
) -> float:
    reciprocal_ranks = []

    for retrieved_pages, real_page in zip(all_retrieved_pages, all_real_pages):
        # Consider only the top-k retrieved pages
        top_k_pages = retrieved_pages[:k]

        if real_page in top_k_pages:
            rank = top_k_pages.index(real_page) + 1  # 1-based index
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

def evaluate_generation(rag_answers: List[dict], evaluator_llm):
    formatted_answers = [
        {
            "user_input": entry["query"],
            "retrieved_contexts": [context["content"] for context in entry["retrieved_contexts"]],
            "response": entry["generated_answer"],
            "reference": entry["true_answer"]
        }
        for entry in rag_answers
    ]

    evaluation_dataset = EvaluationDataset.from_list(formatted_answers)
    result = evaluate(dataset=evaluation_dataset, metrics=[MultiModalFaithfulness(), MultiModalRelevance()], llm=evaluator_llm)
    return result