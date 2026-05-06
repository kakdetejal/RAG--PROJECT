## Building our knowledge base

## 1. Take the input documents 
## 2. split them into chunks
## 3. Embed the chunks
## 4. store it in FAISS  Vectordb

from pathlib import Path
import pickle
import faiss
import numpy as np

## Langchain utilities
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

## Sentence transformer for embedding
from sentence_transformers import SentenceTransformer


## Configurations
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'  ## Sentence transformer model name for embedding
BASE_DIR = Path(__file__).resolve().parent.parent
## __file__: contain the path of the current .py fle
DATA_DIR = BASE_DIR / 'data'
STORE_DIR = BASE_DIR / 'store'

INDEX_PATH = STORE_DIR / 'faiss_index'  ## Path to store FAISS index/embeddings
CHUNKS_PATH = STORE_DIR / 'chunks'  ## Path to store document chunks

## ------------------------------------------------------
## Step 1: LOAD DOCUMENTS
## ------------------------------------------------------
## Load all the pdfs from the data folder. each PDF is converted into Langchain document object.
def load_docs(data_path: Path):
    docs = []

    for file in data_path.glob('*.pdf'):
        loader = PyPDFLoader(str(file))
        docs.extend(loader.load())

    return docs


## ------------------------------------------------------
## Step 2: SPLIT DOCUMENTS INTO CHUNKS
## ------------------------------------------------------

def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,   ## Max chunk size
        chunk_overlap=50   ## Chunk overlap
    )
    return splitter.split_documents(docs)

## ------------------------------------------------------
## Step 3: EMBED the DOCUMENT CHUNKS
## ------------------------------------------------------

def create_embeddings(chunks):
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    texts = [chunk.page_content for chunk in chunks]  ## extract only the text from the chunk

    ## generate embeddings
    ## any sentence that we pass will be converted to emedding
    ## "AI is powerful" = [0.21, -0.56, 0.81,...]
    embeddings = model.encode(
        texts, 
        show_progress_bar=True,  ## to show progress bar while encoding
        convert_to_numpy= True,  ## to convert the embeddings output to numpy array
        normalize_embeddings=True  ## to perform cosine similarity we need to normalize the embeddings
    ).astype('float32')   ## FAISS requires data to be in float32 format

    return embeddings, chunks

## ------------------------------------------------------
## Step 4: STORE the EMBEDDINGS in FAISS
## ------------------------------------------------------

def store_faiss(embeddings, chunks):
    STORE_DIR.mkdir(exist_ok=True)  ## create store directory if it doesn't exist

    dim = embeddings.shape[1]  ## dimension of the embeddings

    ## Create a faiss index using cosine similarity
    index = faiss.IndexFlatIP(dim)  ## IP stands for Inner Product which is equivalent to cosine similarity for normalized vectors

    index.add(embeddings)  ## add the embeddings to the index

    ## Save the FAISS index to disk
    faiss.write_index(index, str(INDEX_PATH))

    ## Save the document chunks to disk
    with open(CHUNKS_PATH, 'wb') as f:
        pickle.dump(chunks, f)

## ------------------------------------------------------
## Step 5: Main execution
## ------------------------------------------------------

if __name__ == '__main__':
    docs = load_docs(DATA_DIR)  ## Load documents
    print(f"Loaded {len(docs)} documents.")
    chunks = split_docs(docs)  ## split the docs into chunks
    print(f"Split into {len(chunks)} chunks.")
    embeddings, chunks = create_embeddings(chunks)  ## create embeddings for the chunks
    print(f"Created embeddings for {len(chunks)} chunks.")
    print(f"Embedding shape: {embeddings.shape}")
    store_faiss(embeddings, chunks)  ## store the embeddings in FAISS

    print(f"Processed {len(docs)} documents into {len(chunks)} chunks and stored in FAISS index.")