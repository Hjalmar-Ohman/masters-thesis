import os
import json
from pdf2image import convert_from_path

from common_utils import encode_image_to_base64, call_gpt_4
from pdf_utils import extract_text_from_pdf, extract_images_from_pdf

def generate_qa_for_pdf(pdf_path, mode="per_page"):
    """
    Generates Q&A pairs from a PDF file and writes them to a cleaned JSON file.

    Parameters:
        pdf_path (str): Path to the PDF file.
        mode (str): "per_page" processes each page as an image,
                    "per_chunk" processes extracted text and images separately.

    Returns:
        str: Path to the generated JSON file containing the Q&A data.
    """

    output_json = "QA_" + os.path.basename(pdf_path).replace('.pdf', '.json')
    qa_data = []

    # Function to generate Q&A from a given input (text or image)
    def generate_qa(prompt_input, page_number):
        user_prompt = [
            {"type": "text", "text": (
                "Your task is to formulate a question from the given context while following these rules:\n"
                "1. The question must be answerable using the provided context.\n"
                "2. It should be based on non-trivial information.\n"
                "3. The answer must not contain any links.\n"
                "4. The question should be of moderate difficulty.\n"
                "5. Avoid phrases like 'provided context'.\n"
                "6. The response must be in valid JSON format as follows:\n"
                r'{"question": "Generated question here", "answer": "Generated answer here"}'
            )},
            prompt_input
        ]

        response_text = call_gpt_4(user_prompt)

        try:
            response_data = json.loads(response_text.strip("```json").strip("```"))
            if "question" in response_data and "answer" in response_data:
                qa_data.append({
                    "page_number": page_number,
                    "question": response_data["question"],
                    "answer": response_data["answer"]
                })
            else:
                print(f"Warning: Missing Q&A data for page {page_number}")

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON response for page {page_number}: {e}")
            print("Response text:", response_text)

    if mode == "per_page":
        # Convert PDF pages to images
        pages = convert_from_path(pdf_path, dpi=200, poppler_path=r'poppler-24.08.0/Library/bin')
        for i, page_image in enumerate(pages, start=1):
            base64_str = encode_image_to_base64(page_image)
            generate_qa({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_str}"}}, i)

    elif mode == "per_chunk":
        # Extract text and images from PDF
        text_data = extract_text_from_pdf(pdf_path)
        image_data = extract_images_from_pdf(pdf_path)

        for text_info in text_data:
            generate_qa({"type": "text", "text": text_info["text"]}, text_info["page_number"])

        for data in image_data:
            base64_str = encode_image_to_base64(data["pil_image"])
            generate_qa({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_str}"}}, data["page_number"])

    else:
        raise ValueError("Invalid mode. Use 'per_page' or 'per_chunk'.")

    # Save the cleaned Q&A data to JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(qa_data, f, ensure_ascii=False, indent=4)

    print(f"Q&A saved to {output_json} using mode: {mode}")
    
    return output_json

if __name__ == "__main__":
    pdf_path = "knowledge/subset_riksbanken.pdf"
    generate_qa_for_pdf(pdf_path, mode="per_page")
