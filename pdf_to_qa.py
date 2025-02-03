from pdf2image import convert_from_path

# Import our common utilities
from common_utils import encode_image_to_base64, call_gpt_4

def generate_qa_for_pdf(pdf_path, output_json='QA.json'):
    """
    Convert each PDF page to an image, then for each image:
      - Encode as base64
      - Call GPT-4 to generate Q&A pairs
    Optionally save results to a JSON file.
    """
    
    # 1. Convert PDF pages to images
    pages = convert_from_path(pdf_path, dpi=200, poppler_path=r'poppler-24.08.0\Library\bin') 
    
    # 2. Iterate over each page image
    qa_results = []
    for i, page_image in enumerate(pages, start=1):
        # Encode image to base64
        base64_str = encode_image_to_base64(page_image)
        
        # Prepare GPT-4 prompt
        # You can give GPT-4 more specific instructions if needed:
        # e.g. "Analyze this image of a PDF page and create 3 question/answer pairs about the content."
        messages = [
            {
                "role": "system",
                "content": "Only generate question and answer pairs based on the content of this image. Do it on the format of a question and an answer. NOTHING ELSE."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Generate 10 question and answer pairs based on the content of this image."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_str}"}}
                ]
            }
        ]
        
        # 3. Call GPT-4
        response_text = call_gpt_4(messages)
        
        # Store or print the results
        qa_results.append({
            "page_number": i,
            "qa_text": response_text
        })
        print(f"\n--- Q&A for page {i} ---")
        print(response_text)
    
    # 4. Optionally save as JSON
    if output_json:
        import json
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(qa_results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {output_json}")

if __name__ == "__main__":
    PDF_FILE = "knowledge/subset_monetary_policy_report.pdf"
    
    generate_qa_for_pdf(
        pdf_path=PDF_FILE,
    )