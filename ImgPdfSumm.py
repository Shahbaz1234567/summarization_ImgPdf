#pip install PyMuPDF transformers pillow

import fitz  # PyMuPDF
from transformers import pipeline
from PIL import Image
import io

def extract_text_and_images(pdf_path):
    doc = fitz.open(pdf_path)
    texts = []
    images = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        texts.append(page.get_text())
        
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            images.append(Image.open(io.BytesIO(image_bytes)))
    
    return texts, images

def summarize_texts(texts, summarizer):
    summarized_texts = []
    for text in texts:
        summarized_text = summarizer(text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
        summarized_texts.append(summarized_text)
    return summarized_texts

def describe_images(images, captioner):
    captions = []
    for img in images:
        caption = captioner(img, max_length=50, min_length=5, do_sample=False)[0]['caption']
        captions.append(caption)
    return captions

# Load the summarization and image captioning models
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
captioner = pipeline("image-captioning", model="nlpconnect/vit-gpt2-image-captioning")

# Path to your PDF file
pdf_path = "F:/AIML_Mphasis/LangChain+HuggingFace/Task/New York Brochure.pdf"

# Extract text and images from the PDF
texts, images = extract_text_and_images(pdf_path)

# Summarize the extracted texts
summarized_texts = summarize_texts(texts, summarizer)

# Generate captions for the extracted images
image_captions = describe_images(images, captioner)

# Output the results
for i, (text, summary) in enumerate(zip(texts, summarized_texts)):
    print(f"Page {i+1} Original Text:\n{text}\n")
    print(f"Page {i+1} Summarized Text:\n{summary}\n")
    print("-" * 80)

for i, (img, caption) in enumerate(zip(images, image_captions)):
    print(f"Image {i+1} Caption:\n{caption}\n")
    img.show()
    print("-" * 80)