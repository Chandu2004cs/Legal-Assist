import os
import pickle
import pdfplumber
import nltk
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import TaggedDocument

# nltk.download('punkt')  # Uncomment this line if you haven't already downloaded 'punkt'

# Paths
pdf_path = r"D:\CODEing\Legal Assist\backend\Legal_Assist_ML\dataset\coi-4March2016.pdf"
cache_path = r"D:\CODEing\Legal Assist\backend\Legal_Assist_ML\models_cache\constitution_cache.pkl"


# Extract full text from the PDF
def extract_text_from_pdf(pdf_path):
    text_pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                text_pages.append(text)
    return "\n".join(text_pages)

# Preprocess entire document
def preprocess_text_as_document(text):
    text = text.lower()
    return word_tokenize(text)

# Main preprocessing function
def preprocess_and_cache_constitution():
    full_text = extract_text_from_pdf(pdf_path)
    tokens = preprocess_text_as_document(full_text)

    # Create one TaggedDocument for the whole Constitution
    tagged_doc = TaggedDocument(tokens, ["CONSTITUTION_0"])  # Tag style compatible with CASE_0, STATUTE_0

    # Save to pickle
    with open(cache_path, 'wb') as f:
        pickle.dump(tagged_doc, f)

    print("✅ Constitution preprocessed as one document and saved to:", cache_path)

# Run script
if __name__ == "__main__":
    preprocess_and_cache_constitution()
