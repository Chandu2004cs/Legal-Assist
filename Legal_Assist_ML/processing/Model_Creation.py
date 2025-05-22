import os
import pickle
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import nltk

# nltk.download('punkt')  # Uncomment if not yet done

# Paths
casedocs_path = r"D:\CODEing\Legal Assist\backend\Legal_Assist_ML\dataset\Object_casedocs"
statutes_path = r"D:\CODEing\Legal Assist\backend\Legal_Assist_ML\dataset\Object_statutes"
constitution_cache_path = r"D:\CODEing\Legal Assist\backend\Legal_Assist_ML\app\models_cache\constitution_cache.pkl"


output_cache_path = r"D:\CODEing\Legal Assist\backend\Legal_Assist_ML\app\models_cache\legal_cache.pkl"  # This is what Query_Generator.py needs

# Load and preprocess case docs
case_texts = []
case_names = []
for filename in os.listdir(casedocs_path):
    with open(os.path.join(casedocs_path, filename), 'r', encoding='utf-8') as file:
        case_texts.append(file.read())
        case_names.append(filename)

# Load and preprocess statute docs
statute_texts = []
statute_titles = []
for filename in os.listdir(statutes_path):
    with open(os.path.join(statutes_path, filename), 'r', encoding='utf-8') as file:
        statute_texts.append(file.read())
        statute_titles.append(filename)

# Tagged documents
tagged_docs = []
for i, doc in enumerate(case_texts):
    tokens = word_tokenize(doc.lower())
    tagged_docs.append(TaggedDocument(tokens, [f"CASE_{i}"]))

for i, doc in enumerate(statute_texts):
    tokens = word_tokenize(doc.lower())
    tagged_docs.append(TaggedDocument(tokens, [f"STATUTE_{i}"]))

# Load the constitution doc
with open(constitution_cache_path, 'rb') as f:
    constitution_doc = pickle.load(f)

tagged_docs.append(constitution_doc)
case_names.append("Indian_Constitution")
case_texts.append("Constitution Text Placeholder")  # You can replace this if needed

# Save compatible cache for Query_Generator
with open(output_cache_path, 'wb') as f:
    pickle.dump((case_names, case_texts, statute_titles, statute_texts, tagged_docs), f)

print("✅ legal_cache.pkl created with Constitution.")

# Train Doc2Vec
model = Doc2Vec(vector_size=300, window=5, min_count=2, workers=4, epochs=50)
model.build_vocab(tagged_docs)
model.train(tagged_docs, total_examples=model.corpus_count, epochs=model.epochs)

# Save model
model.save(r"D:\CODEing\Legal Assist\backend\Legal_Assist_ML\app\models_cache\legal_doc2vec_with_constitution.model")
print("✅ Model trained and saved.")
