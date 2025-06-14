from typing import List, Union

from ragas.metrics import MultiModalRelevance, MultiModalFaithfulness, AnswerCorrectness
from ragas import evaluate, EvaluationDataset

def compute_mrr_at_k(
    all_retrieved_pages: Union[List[List[int]], List[int]], 
    all_real_pages: Union[List[List[int]], List[int]], 
    k: int
) -> float:
    if isinstance(all_retrieved_pages[0], int):
        all_retrieved_pages = [all_retrieved_pages]
        all_real_pages = [all_real_pages]

    reciprocal_ranks = []

    for retrieved_pages, real_pages in zip(all_retrieved_pages, all_real_pages):
        # Consider only the top-k retrieved pages
        top_k_pages = retrieved_pages[:k]

        rank = float('inf')
        for real_page in real_pages:
            if real_page in top_k_pages:
                rank = min(rank, top_k_pages.index(real_page) + 1)  # 1-based index

        if rank != float('inf'):
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

def compute_recall_at_k(
    all_retrieved_pages: Union[List[List[int]], List[int]], 
    all_real_pages: Union[List[List[int]], List[int]], 
    k: int
) -> float:
    if isinstance(all_retrieved_pages[0], int):
        all_retrieved_pages = [all_retrieved_pages]
        all_real_pages = [all_real_pages]

    total_hits = 0
    total_relevant = 0

    for retrieved_pages, real_pages in zip(all_retrieved_pages, all_real_pages): 
        # Consider only the top-k retrieved pages
        top_k_pages = retrieved_pages[:k]

        hits = sum(1 for page in real_pages if page in top_k_pages) # it counts how many of the relevant pages (from real_pages) appear in top_k_pages
        total_hits += hits
        total_relevant += len(real_pages)

    return total_hits / total_relevant if total_relevant else 0.0

def compute_precision_at_k(
    all_retrieved_pages: Union[List[List[int]], List[int]], 
    all_real_pages: Union[List[List[int]], List[int]], 
    k: int
) -> float:
    if isinstance(all_retrieved_pages[0], int):
        all_retrieved_pages = [all_retrieved_pages]
        all_real_pages = [all_real_pages]

    total_hits = 0
    total_retrieved = 0

    for retrieved_pages, real_pages in zip(all_retrieved_pages, all_real_pages):
        # Consider only the top-k retrieved pages
        top_k_pages = retrieved_pages[:k]

        hits = sum(1 for page in top_k_pages if page in real_pages)  # it counts how many of the top_k_pages are in real_pages
        total_hits += hits
        total_retrieved += len(top_k_pages)

    return total_hits / total_retrieved if total_retrieved else 0.0

def compute_f1_score(retrieved_pages, real_pages, k):
        precision = compute_precision_at_k(retrieved_pages, real_pages, k)
        recall = compute_recall_at_k(retrieved_pages, real_pages, k)
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

def compute_map_at_k(
    all_retrieved_pages: Union[List[List[int]], List[int]], 
    all_real_pages: Union[List[List[int]], List[int]], 
    k: int
) -> float:
    if isinstance(all_retrieved_pages[0], int):
        all_retrieved_pages = [all_retrieved_pages]
        all_real_pages = [all_real_pages]
    
    average_precisions = []
    
    for retrieved_pages, real_pages in zip(all_retrieved_pages, all_real_pages):
        top_k_pages = retrieved_pages[:k]
        
        num_relevant = 0
        precision_sum = 0.0
        
        for i, page in enumerate(top_k_pages):
            if page in real_pages:
                num_relevant += 1
                precision_at_i = num_relevant / (i + 1)  # Precision@i (1-based index)
                precision_sum += precision_at_i
        
        if num_relevant > 0:
            average_precision = precision_sum / num_relevant  # Normalize by the number of relevant documents retrieved
        else:
            average_precision = 0.0
        
        average_precisions.append(average_precision)
    
    return sum(average_precisions) / len(average_precisions) if average_precisions else 0.0


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
    result = evaluate(
        dataset=evaluation_dataset, 
        metrics=[MultiModalFaithfulness(), MultiModalRelevance(), AnswerCorrectness()], 
        llm=evaluator_llm
    )
    return result  # Returning raw result for debugging

def evaluate_generation_chartQA(rag_answers: List[dict], evaluator_llm):
    formatted_answers = [
        {
            "user_input": entry["query"],
            "retrieved_contexts": [entry["retrieved_contexts"][0]["content"]],
            "response": entry["generated_answer"],
            "reference": entry["true_answer"]
        }
        for entry in rag_answers
    ]
    
    

    evaluation_dataset = EvaluationDataset.from_list(formatted_answers)
    
    result = evaluate(
        dataset=evaluation_dataset, 
        metrics=[MultiModalFaithfulness(), MultiModalRelevance(), AnswerCorrectness()], 
        llm=evaluator_llm,
    )
    return result  # Returning raw result for debugging
