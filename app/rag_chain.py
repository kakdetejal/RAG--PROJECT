## this code will combine retrieval (context) and generation (response)

import os

from dotenv import load_dotenv
from groq import Groq

from retriever import retriever

## Load environment variables from .env file
load_dotenv()

## Initialize Groq client
client = Groq(
    api_key=os.getenv("GROQ_API_KEY"))

## MODEL name
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-120b")

def generate_answer(query: str):
    """
    MAIN RAG Pipeline:
    1. Retrieve context from the retriever
    2. Build the prompt
    3. Generate response using the Groq client
    """

    ## Step 1: Retrieve context
    context = retriever(query)

    ## Step 2: Combine chunks into single string
    prompt  = f"""
Context : {context}
Question: {query}
"""

    ## Step 3: Generate response
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are an enterprise Q&A assistant for answering questions based on the following context: If the answer is not present in this chunk, say I don't know based on the provided document."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content