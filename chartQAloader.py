import json
from fpdf import FPDF
from PIL import Image

def generate_chartQA_pdf_and_json(dataset, pdf_output_path='ChartQA_Evaluation_Set.pdf', json_output_path='ChartQA_QA_Mapping.json'):
    # Initialize the PDF
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Initialize the JSON structure
    qa_list = []

    # Process each entry in the dataset
    for idx, data in enumerate(dataset):
        # Load the image
        image = Image.open(data['image_path'])

        # Save the image temporarily
        temp_image_path = f'temp_image_{idx}.png'
        image.save(temp_image_path)

        # Add a new page to the PDF
        pdf.add_page()

        # Set the image size to fit the page
        pdf.image(temp_image_path, x=10, y=10, w=pdf.w - 20)

        # Append the question and answer to the JSON list
        qa_list.append({
            'page_number': idx + 1,
            'question': data['question'],
            'answer': data['answer']
        })

    # Save the PDF
    pdf.output(pdf_output_path)

    # Save the JSON file
    with open(json_output_path, 'w') as json_file:
        json.dump(qa_list, json_file, indent=4)

    print(f'PDF saved as {pdf_output_path}')
    print(f'JSON file saved as {json_output_path}')