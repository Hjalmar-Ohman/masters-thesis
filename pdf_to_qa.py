import os
import json
from pdf2image import convert_from_path

from common_utils import encode_image_to_base64, call_gpt_4

def generate_qa_for_pdf(pdf_path):
    """
    Convert each PDF page to an image, then for each image:
    - Encode as base64
    - Call GPT-4 to generate Q&A pairs
    - Append structured JSON entries to a JSON file

    Return: Path to QA dataset JSON file.
    """
    
    output_json = "QA_" + os.path.basename(pdf_path).replace('.pdf', '.json')

    try:
        pages = convert_from_path(pdf_path, dpi=200, poppler_path=r'poppler-24.08.0/Library/bin')
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        return

    qa_list = []  # Store structured Q&A

    for i, page_image in enumerate(pages, start=1):
        base64_str = encode_image_to_base64(page_image)

        user_prompt = [
            {
                "type": "text",
                "text": (
                    "Your task is to generate a question and answer from the given context while following these rules:\n"
                    "1. The question must be answerable using the provided context.\n"
                    "2. It should be based on non-trivial information.\n"
                    "3. The answer must not contain any links.\n"
                    "4. The question should be of moderate difficulty.\n"
                    "5. Avoid phrases like 'provided context'.\n"
                    "6. The response must be in valid JSON format as follows:\n"
                    "{'question': 'Generated question here', 'answer': 'Generated answer here'}"
                ),
            },
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_str}"}}
        ]

        response_text = call_gpt_4(user_prompt)

        # Clean and parse the response
        try:
            response_text = response_text.strip("```json").strip("```").strip()  # Remove markdown code formatting
            qa_data = json.loads(response_text.replace("'", "\""))  # Convert single quotes to double for valid JSON

            if isinstance(qa_data, dict) and "question" in qa_data and "answer" in qa_data:
                qa_list.append({
                    "page_number": i,
                    "question": qa_data["question"],
                    "answer": qa_data["answer"]
                })

                print(f"\n--- Q&A for page {i} ---\n{qa_data}\n")

        except Exception as e:
            print(f"Error parsing GPT-4 response on page {i}: {e}\nResponse: {response_text}")

    # Save all Q&A to a structured JSON file
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(qa_list, f, ensure_ascii=False, indent=4)

    print(f"\nFinal structured Q&A saved to {output_json}")
    return output_json

if __name__ == "__main__":
    PDF_FILE = "knowledge/subset_monetary_policy_report.pdf"
    generate_qa_for_pdf(pdf_path=PDF_FILE)
