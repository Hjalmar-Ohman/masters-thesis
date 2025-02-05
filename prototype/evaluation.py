import json
from pdf2image import convert_from_path
from common_utils import encode_image_to_base64, call_gpt_4
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
from MRAG import RAG

def generate_qa_for_pdf(pdf_path):
    output_json = "QA_" + pdf_path.split('/')[-1].replace('.pdf', '.json')
    pages = convert_from_path(PDF_FILE, dpi=200, poppler_path=r'poppler-24.08.0\Library\bin')
    
    for i, page_image in enumerate(pages, start=1):
        base64_str = encode_image_to_base64(page_image)
        messages = [
            {"role": "system", "content": "Only generate question and answer pairs based on the content of this image. Output 2 pairs. Nothing else."},
            {"role": "user", "content": [
                {"type": "text", "text": "Generate 2 question and answer pairs based on the content of this page."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_str}"}}
            ]}
        ]
        response_text = call_gpt_4(messages)
        record = {"page_number": i, "qa_text": response_text}
        
        with open(output_json, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")
        
        print(f"\n--- Q&A for page {i} ---\n{response_text}\n")
        print(f"Appended results for page {i} to {output_json}")

def load_qa_dataset(json_path):
    sample_queries = []
    expected_responses = []
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            qa_data = json.loads(line)
            qas = qa_data["qa_text"].split('\n')
            for qa in qas:
                if ':' in qa:
                    question, answer = qa.split(':', 1)
                    sample_queries.append(question.strip())
                    expected_responses.append(answer.strip())
    return sample_queries, expected_responses

if __name__ == "__main__":
    PDF_FILE = "../knowledge/subset_monetary_policy_report.pdf"
    generate_qa_for_pdf(pdf_path=PDF_FILE)
    
    json_path = "QA_subset_monetary_policy_report.json"
    sample_queries, expected_responses = load_qa_dataset(json_path)
    
    pdf_path = "../knowledge/subset_monetary_policy_report.pdf"
    api_key = os.environ.get("OPENAI_API_KEY")
    rag = RAG(pdf_path, api_key)
    
    dataset = []
    for query, reference in zip(sample_queries, expected_responses):
        relevant_docs = rag.get_most_relevant_docs(query)
        response = rag.generate_answer(query, relevant_docs)
        dataset.append({
            "user_input": query,
            "retrieved_contexts": relevant_docs,
            "response": response,
            "reference": reference
        })
    
    evaluation_dataset = EvaluationDataset.from_list(dataset)
    llm = None  # Replace with an actual LangChain LLM instance if required
    evaluator_llm = LangchainLLMWrapper(llm)
    result = evaluate(dataset=evaluation_dataset, metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()], llm=evaluator_llm)
    print(result)
