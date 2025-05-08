import abc
import torch
import os
import json

from typing import List, Dict, Any, Optional
from pdf2image import convert_from_path

from pdf_utils import chunk_text_from_pdf, chunk_text_from_pdfLarge, extract_images_from_pdf, extract_images_from_pdf_unstructured, extract_images_from_pdf_chartQA
from common_utils import encode_image_to_base64, call_gpt_4
from embedder import MultimodalEmbedder, TextEmbedder
import fitz  # PyMuPDF



class DocumentProcessor(abc.ABC):
    def __init__(self, embedder):
        self.embedder = embedder
        self.faiss_index = None
        self.embeddings = None
        self.metadata: List[Dict[str, Any]] = []
        self.name = self.__class__.__name__

    @abc.abstractmethod
    def process_pdf(self, pdf_file: str) -> None:
        pass

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        scores, indices = self.embedder.search(query, self.embeddings, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            item = self.metadata[idx].copy()
            item["score"] = float(score)
            results.append(item)
        return results


class TextProcessor(DocumentProcessor):
    def __init__(self, embedder: TextEmbedder):
        super().__init__(embedder)
        self.name = "TextProcessor"

    def process_pdf(self, pdf_file: str):
        text_chunks = chunk_text_from_pdf(pdf_file)
        if not text_chunks:
            return
        texts_list = [td["text"] for td in text_chunks]
        
        self.metadata = [{"type": "text", "content": td["text"], "page_number": td["page_number"]} for td in text_chunks]
        self.embeddings = self.embedder.embed_text(texts_list)


class ImageProcessor(DocumentProcessor):
    def __init__(self, embedder: MultimodalEmbedder, dataset: str, batch_size=4):
        super().__init__(embedder)
        self.batch_size = batch_size
        self.dataset = dataset
        self.name = "ImageProcessor"

    def process_pdf(self, pdf_file: str):
        if self.dataset.upper() == "CHARTQA":
            image_data = extract_images_from_pdf_chartQA(pdf_file)
            print("Extracted images from ChartQA dataset.")
        else:
            image_data = extract_images_from_pdf_unstructured(pdf_file)
 
        pil_images_list = [info["pil_image"] for info in image_data]

        image_page_counter = {}

        self.metadata = []
        for info in image_data:
            page_number = info["page_number"]
            if page_number not in image_page_counter:
                image_page_counter[page_number] = 1
            else:
                image_page_counter[page_number] += 1

            self.metadata.append({
            "type": "image",
            "content": encode_image_to_base64(info["pil_image"]),
            "page_number": f"{page_number}.{image_page_counter[page_number]}"
            })

        # **Batching Logic**
        all_embeddings = []
        for i in range(0, len(pil_images_list), self.batch_size):
            batch = pil_images_list[i:i + self.batch_size]
            batch_embeddings = self.embedder.embed_image(batch)
            all_embeddings.append(batch_embeddings)

            torch.cuda.empty_cache()  # Free GPU memory after each batch

        self.embeddings = torch.cat(all_embeddings, dim=0)


class PageImageProcessor(DocumentProcessor):
    def __init__(self, embedder: MultimodalEmbedder, dpi=200, batch_size=10):
        super().__init__(embedder)
        self.dpi = dpi
        self.batch_size = batch_size
        self.name = "PageImageProcessor"

    def process_pdf(self, pdf_file: str):
        print(f"Starting to process PDF: {pdf_file} with DPI: {self.dpi} and batch size: {self.batch_size}")
        
        # Get the total number of pages in the PDF
        from PyPDF2 import PdfReader
        reader = PdfReader(pdf_file)
        total_pages = len(reader.pages)
        print(f"Total pages in PDF: {total_pages}")

        self.metadata = []
        all_embeddings = []

        # Process the PDF in batches of pages
        for start_page in range(0, total_pages, self.batch_size):
            end_page = min(start_page + self.batch_size, total_pages)
            print(f"Processing pages {start_page + 1} to {end_page}")

            # Convert only the current batch of pages to images
            pages = convert_from_path(pdf_file, dpi=self.dpi, first_page=start_page + 1, last_page=end_page)
            print(f"Converted {len(pages)} pages to images.")

            # Create metadata for the batch
            batch_metadata = [
                {"type": "page_image", "content": encode_image_to_base64(page_img), "page_number": start_page + i + 1}
                for i, page_img in enumerate(pages)
            ]
            print(f"Batch metadata created for {len(batch_metadata)} pages.")

            # Generate embeddings for the batch
            batch_embeddings = self.embedder.embed_image(pages)
            print(f"Batch embeddings generated for {len(batch_embeddings)} pages.")

            # Append batch results to the overall results
            self.metadata.extend(batch_metadata)
            all_embeddings.append(batch_embeddings)

            # Free GPU memory after processing each batch
            torch.cuda.empty_cache()
            print("Freed GPU memory after processing batch.")

        # Combine all embeddings
        self.embeddings = torch.cat(all_embeddings, dim=0)
        print(f"Finished processing PDF. Total embeddings shape: {self.embeddings.shape}")


class ImageTextualSummaryProcessor(DocumentProcessor):
    def __init__(self, embedder: TextEmbedder, dataset: str):
        super().__init__(embedder)
        self.dataset = dataset
        self.name = "ImageTextualSummaryProcessor"

    def process_pdf(self, pdf_file: str):
        text_chunks = chunk_text_from_pdf(pdf_file)
        texts_list = [td["text"] for td in text_chunks]
        
        if self.dataset.upper() == "CHARTQA":
            image_data = extract_images_from_pdf_chartQA(pdf_file)
            print("Extracted images from ChartQA dataset.")
        else:
            image_data = extract_images_from_pdf_unstructured(pdf_file)
 
        pil_images_list = [img_info["pil_image"] for img_info in image_data]
        base64_images_list = [encode_image_to_base64(pil_img) for pil_img in pil_images_list]

        image_summaries = []
        chart_summary_prompt = """
        Summarize the above image in plain text.
        """

        for b64_str in base64_images_list:
            prompt_content = [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_str}"}},
                {"type": "text", "text": chart_summary_prompt}
            ]
            summary = call_gpt_4(prompt_content)
            image_summaries.append(summary)

        all_texts = texts_list + image_summaries
        self.embeddings = self.embedder.embed_text(all_texts)

        self.metadata = []
        text_page_counter = {}
        image_page_counter = {}

        for td in text_chunks:
            page_number = td["page_number"]

            self.metadata.append({
            "type": "text",
            "content": td["text"],
            "page_number": page_number
            })

        for img_info in image_data:
            page_number = img_info["page_number"]
            if page_number not in image_page_counter:
                image_page_counter[page_number] = 1
            else:
                image_page_counter[page_number] += 1

            self.metadata.append({
            "type": "image",
            "content": base64_images_list[image_page_counter[page_number] - 1],
            "page_number": f"{page_number}.{image_page_counter[page_number]}"
            })


class ImageTextualSummaryProcessorLarge(DocumentProcessor):
    def __init__(self, embedder, dataset: str, batch_size: int = 25):
        super().__init__(embedder)
        self.dataset = dataset
        self.name = "ImageTextualSummaryProcessor"
        self.batch_size = batch_size
        self.embeddings = []
        self.metadata = []

    def process_pdf(self, pdf_file: str):
        doc = fitz.open(pdf_file)
        total_pages = len(doc)

        for batch_start in range(0, total_pages, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_pages)
            base_name = f"{self.name}_{batch_start+1}_{batch_end}"
            embed_path = f"embedding_cache/batches/{base_name}_embeddings.pt"

            print(f"Checking batch {batch_start+1}-{batch_end}...")

            # Skip this batch if already processed
            if os.path.exists(embed_path):
                print(f" → Skipping, embeddings already exist: {embed_path}")
                continue

            print(f" → Processing pages {batch_start+1} to {batch_end}")

            text_chunks = chunk_text_from_pdfLarge(pdf_file, start_page=batch_start, end_page=batch_end)
            texts_list = [td["text"] for td in text_chunks]

            if self.dataset.upper() == "CHARTQA":
                image_data = extract_images_from_pdf_chartQA(doc, start_page=batch_start, end_page=batch_end)
            else:
                image_data = extract_images_from_pdf_unstructured(doc, start_page=batch_start, end_page=batch_end)

            pil_images_list = [img_info["pil_image"] for img_info in image_data]
            base64_images_list = [encode_image_to_base64(pil_img) for pil_img in pil_images_list]

            image_summaries = []
            for b64_str in base64_images_list:
                prompt_content = [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_str}"}},
                    {"type": "text", "text": self.chart_summary_prompt()},
                ]
                summary = call_gpt_4(prompt_content)
                image_summaries.append(summary)
                print(f"Image no {len(image_summaries)} summary generated")

            print(f"Generated {len(image_summaries)} image summaries for pages {batch_start+1} to {batch_end}")

            all_texts = texts_list + image_summaries
            batch_embeddings = self.embedder.embed_text(all_texts)

            self.save_batch(batch_start, batch_end, batch_embeddings, text_chunks, image_data, base64_images_list)

        doc.close()

    def save_batch(self, start: int, end: int, embeddings, text_chunks, image_data, base64_images_list):
        os.makedirs("embedding_cache/batches", exist_ok=True)
        base_name = f"{self.name}_{start+1}_{end}"

        # Save embeddings
        torch.save(embeddings, f"embedding_cache/batches/{base_name}_embeddings.pt")

        # Build metadata
        metadata = []
        for td in text_chunks:
            metadata.append({"type": "text", "content": td["text"], "page_number": td["page_number"]})
        for i, img_info in enumerate(image_data):
            metadata.append({"type": "image", "content": base64_images_list[i], "page_number": img_info["page_number"]})

        with open(f"embedding_cache/batches/{base_name}_metadata.json", "w") as f:
            json.dump(metadata, f)

    def chart_summary_prompt(self):
        return """
        Summarize the above image in plain text.
        """
