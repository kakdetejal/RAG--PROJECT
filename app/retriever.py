## Functions of this module: Retrieve the top k relevant document chunks from the vector store

from pathlib import Path
import pickle

import faiss
from sentence_transformers import SentenceTransformer

## Configurations
MODEL_NAME = 'all-MiniLM-L6-v2'  ## Sentence transformer model name for embedding
BASE_DIR = Path(__file__).resolve().parent.parent
STORE_DIR = BASE_DIR / 'store'

INDEX_PATH = STORE_DIR / 'faiss_index'
CHUNKS_PATH = STORE_DIR / 'chunks'

## Load the model
model = SentenceTransformer(MODEL_NAME)

def retriever(query: str, k: int = 5):
    """
    Retrieve the top-k (5) relevant document chunks for a given query.
    """

    ## Ensure vector index is exising before performing search
    if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
        raise FileNotFoundError("FAISS index or chunks not found. Please run the ingest process first.")
    
    ## Load the FAISS index from the disk
    index = faiss.read_index(str(INDEX_PATH))

    ## Load the original document chunks
    with open(CHUNKS_PATH, 'rb') as f:
        chunks = pickle.load(f)
    
    ## Convert the query into embedding
    query_embedding = model.encode([query],
                                   convert_to_numpy=True,
                                   normalize_embeddings=True,
                                   show_progress_bar= True).astype("float32")
    
    ## Search for the top-k similar vectors
    distances, indices = index.search(query_embedding, k)

    ## Collect all matching chunks:
    results = []
    for i in indices[0]:
        results.append(chunks[i].page_content)
    return results