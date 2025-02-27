from typing import List

from ragas.metrics import MultiModalRelevance, MultiModalFaithfulness
from ragas import evaluate, EvaluationDataset

def compute_mrr_at_k(all_retrieved_pages: List[List[int]], all_real_pages: List[int], k: int) -> float:
    reciprocal_ranks = [
        1.0 / (retrieved_pages[:k].index(real_page) + 1)
        if real_page in retrieved_pages[:k] else 0.0
        for retrieved_pages, real_page in zip(all_retrieved_pages, all_real_pages)
    ]
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