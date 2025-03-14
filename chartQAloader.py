from fpdf import FPDF
import json
import io
import tempfile
from tqdm import tqdm  # Progress bar
from PIL import Image

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
    print(f'JSON file saved as {json_output_path}')
