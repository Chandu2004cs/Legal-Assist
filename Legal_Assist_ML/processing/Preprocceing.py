import os
import nltk
import pickle
from nltk.tokenize import word_tokenize, sent_tokenize
from gensim.models.doc2vec import TaggedDocument


# Paths
casedocs_path = r"D:\CODEing\Legal Assist\backend\Legal_Assist_ML\dataset\Object_casedocs"
statutes_path = r"D:\CODEing\Legal Assist\backend\Legal_Assist_ML\dataset\Object_statutes"
cache_path = "legal_cache.pkl"

# Load documents
def load_documents_from_folder(folder_path):
    filenames = os.listdir(folder_path)
    texts = []
    for fname in filenames:
        full_path = os.path.join(folder_path, fname)
        if os.path.isfile(full_path):
            with open(full_path, 'r', encoding='utf-8') as f:
                texts.append(f.read())
    return filenames, texts

# Clean and tokenize more richly
def preprocess_text(text):
    text = text.lower()
    sentences = sent_tokenize(text)  # Split into sentences for finer granularity
    tokens = [word_tokenize(sentence) for sentence in sentences]
    return tokens

# Main caching function
def preprocess_and_cache():
    case_names, case_texts = load_documents_from_folder(casedocs_path)
    statute_titles, statute_texts = load_documents_from_folder(statutes_path)

    tagged_docs = []
    
    # Process cases
    for i, doc in enumerate(case_texts):
        tokenized_sentences = preprocess_text(doc)
        for j, sentence_tokens in enumerate(tokenized_sentences):
            tag = f"CASE_{i}_SENT_{j}"
            tagged_docs.append(TaggedDocument(sentence_tokens, [tag]))

    # Process statutes
    for i, doc in enumerate(statute_texts):
        tokenized_sentences = preprocess_text(doc)
        for j, sentence_tokens in enumerate(tokenized_sentences):
            tag = f"STATUTE_{i}_SENT_{j}"
            tagged_docs.append(TaggedDocument(sentence_tokens, [tag]))

    with open(cache_path, 'wb') as f:
        pickle.dump((case_names, case_texts, statute_titles, statute_texts, tagged_docs), f)

    print("✅ Preprocessing complete. Cache saved to", cache_path)

# Run script
if __name__ == "__main__":
    preprocess_and_cache()
