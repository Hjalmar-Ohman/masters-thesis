import json
import io
import tempfile

from fpdf import FPDF
from tqdm import tqdm
from pdf2image import convert_from_path

from common_utils import encode_image_to_base64, call_gpt_4
from pdf_utils import chunk_text_from_pdf, extract_images_from_pdf

def generate_qa_for_pdf(pdf_path, json_output_path, mode="per_image"):
    """
    Generates Q&A pairs from a PDF file and writes them to a cleaned JSON file.

    Parameters:
        pdf_path (str): Path to the PDF file.
        mode (str): "per_page" processes each page as an image,
                    "per_chunk" processes extracted text and images separately,
                    "per_image" processes only the images embedded in the PDF.


    Returns:
        str: Path to the generated JSON file containing the Q&A data.
    """
    qa_data = []

    # Function to generate Q&A from a given input (text or image)
    def generate_qa(prompt_input, page_number, custom_prompt=None):
        """Generates Q&A using a custom or default prompt."""
        default_prompt = (
            "Your task is to formulate a question from the given context while following these rules:\n"
            "1. The question must be answerable using the provided context.\n"
            "2. It should be based on non-trivial information.\n"
            "3. The answer must not contain any links.\n"
            "4. The question should be of moderate difficulty.\n"
            "5. Avoid phrases like 'provided context'.\n"
            "6. The response must be in valid JSON format as follows:\n"
            r'{"question": "Generated question here", "answer": "Generated answer here"}'
        )
        prompt_text = custom_prompt if custom_prompt else default_prompt
        user_prompt = [
            {"type": "text", "text": prompt_text},
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
        pages = convert_from_path(pdf_path, dpi=200)
        for i, page_image in enumerate(pages, start=1):
            base64_str = encode_image_to_base64(page_image)
            generate_qa({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_str}"}}, i)

    elif mode == "per_chunk":
        # Extract text and images from PDF
        text_chunks = chunk_text_from_pdf(pdf_path)
        image_data = extract_images_from_pdf(pdf_path)

        for text_info in text_chunks:
            generate_qa({"type": "text", "text": text_info["text"]}, text_info["page_number"])

        for data in image_data:
            base64_str = encode_image_to_base64(data["pil_image"])
            generate_qa({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_str}"}}, data["page_number"])
    
    elif mode == "per_image":
        # Extract images from PDF
        image_data = extract_images_from_pdf(pdf_path)

        for data in image_data:
            base64_str = encode_image_to_base64(data["pil_image"])
            custom_prompt = (
                "Your task is to generate a question that can be answered by analyzing the provided chart image. "
                "Follow these rules:\n"
                "1. The question must be specific to the data or trends visible in the chart.\n"
                "2. Avoid generic questions; focus on insights or patterns in the chart.\n"
                "3. Ensure the question is answerable using only the chart image.\n"
                "4. The question should be of moderate difficulty.\n"
                "5. Avoid phrases like 'provided context' or 'in the chart'.\n"
                "6. The question should not contain any links.\n"
                "7. Generate only one question per image, avoid generating questions followed by 'and how does it compare to...'.\n"
                "8. The response must be in valid JSON format as follows:\n"
                r'{"question": "Generated question here", "answer": "Generated answer here"}'
            )
            image_input = {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_str}"}
            }
            generate_qa(image_input, data["page_number"], custom_prompt)

    elif mode == "per_image":
        # Only process images
        image_data = extract_images_from_pdf(pdf_path)

        for data in image_data:
            base64_str = encode_image_to_base64(data["pil_image"])
            generate_qa({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_str}"}}, data["page_number"])

    else:
        raise ValueError("Invalid mode. Use 'per_page', 'per_chunk', or 'per_image'.")


    # Save the cleaned Q&A data to JSON
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(qa_data, f, ensure_ascii=False, indent=4)

    print(f"Q&A saved to {json_output_path} using mode: {mode}")

def generate_chartQA_pdf_and_json(dataset, pdf_output_path='ChartQA_Evaluation_Set.pdf', json_output_path='ChartQA_QA_Mapping.json'):
    # Initialize the PDF
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Initialize JSON structure
    qa_list = []
    
    previous_image = None  # Store the last unique image
    current_page = 0  # Track the current page number

    # Process each entry in the dataset with a progress bar
    for idx, data in enumerate(tqdm(dataset, desc="Processing Charts", unit="chart")):
        image = data['image']
        
        # Convert the image to a byte format to compare with the previous one
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="PNG")
        image_bytes = image_bytes.getvalue()  # Get raw bytes
        
        # Check if the image is different from the previous one
        if previous_image != image_bytes:
            previous_image = image_bytes  # Update the last unique image
            
            # Increment the page number since it's a new image
            current_page += 1
            
            # Use a temporary file for FPDF
            with tempfile.NamedTemporaryFile(delete=True, suffix=".png") as temp_image:
                image.save(temp_image, format="PNG")
                temp_image.flush()  # Ensure data is written
                
                # Add a new page and insert the image
                pdf.add_page()
                pdf.image(temp_image.name, x=10, y=10, w=pdf.w - 20)

        # Append question-answer mapping, linking to the correct page
        qa_list.append({
            'page_number': current_page,
            'question': data['question'],
            'answer': data['answer'],
            'type': data['type']
        })

    # Save the PDF
    pdf.output(pdf_output_path)

    # Save the JSON file
    with open(json_output_path, 'w') as json_file:
        json.dump(qa_list, json_file, indent=4)

    print(f'PDF saved as {pdf_output_path}')

if __name__ == "__main__":
    pdf_path = "knowledge/riksbanken.pdf"
    json_output_path = "json_files/QA_riksbanken.json"
    
    generate_qa_for_pdf(pdf_path, json_output_path, mode="per_image")