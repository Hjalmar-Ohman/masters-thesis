import json
from pdf2image import convert_from_path

# Import your common utilities (including encode_image_to_base64, call_gpt_4)
from common_utils import encode_image_to_base64, call_gpt_4

def generate_qa_for_pdf(pdf_path):
    """
    Convert each PDF page to an image, then for each image:
    - Encode as base64
    - Call GPT-4 to generate Q&A pairs
    - Immediately append the result to an NDJSON/JSONL file (one JSON object per line),
      so we don't keep everything in memory for large PDFs.
    """
    
    # Output file
    output_json = "QA_" + pdf_path.split('/')[-1].replace('.pdf', '.json')

    # 1. Convert PDF pages to images
    pages = convert_from_path(PDF_FILE, dpi=200, poppler_path=r'poppler-24.08.0/Library/bin')

    # 2. Process each page, one at a time
    for i, page_image in enumerate(pages, start=1):
        # Encode page as base64
        base64_str = encode_image_to_base64(page_image)

        # Prompt GPT-4. For example, ask it for 10 Q&A pairs:
        user_prompt = [
                            {"type": "text", "text": "Generate 10 question and answer pairs based on the content of this page in JSON format. NOTHING ELSE"},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_str}"}}
                        ]

        response_text = call_gpt_4(user_prompt)

        # Prepare the record for this page
        record = {
            "page_number": i,
            "qa_text": response_text
        }

        # 3. Append record to the JSON file in NDJSON/JSONL format
        #    (one JSON object per line)
        with open(output_json, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")

        print(f"\n--- Q&A for page {i} ---\n{response_text}\n")
        print(f"Appended results for page {i} to {output_json}")

if __name__ == "__main__":
    PDF_FILE = "knowledge/subset_monetary_policy_report.pdf"
    generate_qa_for_pdf(pdf_path=PDF_FILE)