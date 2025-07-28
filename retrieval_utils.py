import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import joblib
import os

MODEL_NAME = 'microsoft/codebert-base'
INDEX_PATH = 'faiss_index/code_faiss.index'
EMBEDDINGS_PATH = 'faiss_index/embeddings.npy'
MAPPING_PATH = 'faiss_index/id2row.joblib'
DATA_PATH = 'data/svace_dataset.csv'

os.makedirs('faiss_index', exist_ok=True)

# Build index
def build_index():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=['code_with_bug', 'flag', 'fixed_code'])
    model = SentenceTransformer(MODEL_NAME)
    texts = df.apply(lambda row: f"[{row['flag']}] {row['code_with_bug']}", axis=1).tolist()
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
    faiss_index.add(embeddings)
    np.save(EMBEDDINGS_PATH, embeddings)
    faiss.write_index(faiss_index, INDEX_PATH)
    joblib.dump(df.to_dict('records'), MAPPING_PATH)
    print('✅ FAISS index built and saved.')

# Query index
def query_index(query_text, top_k=3):
    model = SentenceTransformer(MODEL_NAME)
    faiss_index = faiss.read_index(INDEX_PATH)
    id2row = joblib.load(MAPPING_PATH)
    query_emb = model.encode([query_text], convert_to_numpy=True)
    D, I = faiss_index.search(query_emb, top_k)
    results = [id2row[i] for i in I[0]]
    return results

if __name__ == '__main__':
    build_index() 