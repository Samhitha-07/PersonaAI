import os
import tempfile

import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from streamlit.runtime.uploaded_file_manager import UploadedFile
import ollama
from sentence_transformers import CrossEncoder

import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction

system_prompt = """
You are an AI assistant tasked with providing **long, detailed, and structured** answers based solely on the given context.
Your goal is to analyze the information thoroughly and provide **comprehensive** responses.

context will be passed as "Context:"
user question will be passed as "Question:"

Guidelines:
1. Provide full explanations with examples if applicable.
2. Break down complex answers using bullet points or numbered lists.
3. Ensure logical flow by organizing the response into multiple paragraphs.
4. Include definitions, causes, effects, and additional insights where relevant.
5. If context lacks sufficient data, state that explicitly instead of guessing.

Always provide the most elaborate and well-structured answer possible.
"""


def process_document(uploaded_file: UploadedFile) -> list[Document]:
    # Store uploaded file as a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name  # Store the file path before closing
    
    # Load PDF using PyMuPDFLoader
    loader = PyMuPDFLoader(temp_file_path)
    docs = loader.load()

    # Delete temp file AFTER loading
    os.remove(temp_file_path)  

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
    )
    return text_splitter.split_documents(docs)

def get_vector_collection() -> chromadb.Collection:
    ollama_ef = OllamaEmbeddingFunction(
        url = "http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text:latest",
    )
    
    chroma_client = chromadb.PersistentClient(path="./demo-rag-chroma")
    return chroma_client.get_or_create_collection(
        name="rag_app",
        embedding_function=ollama_ef,
        metadata={"hnsw:space":"cosine"},
    )

def add_to_vector_collection(all_splits: list[Document], file_name:str):
    collection = get_vector_collection()
    documents,metadatas,ids = [],[],[]
    
    for idx,split in enumerate(all_splits):
        documents.append(split.page_content)
        metadatas.append(split.metadata)
        ids.append(f"{file_name}_{idx}")

    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )
    st.success("Data added to vector store!")
    
def query_collection(prompt: str, n_results: int = 15):
    collection = get_vector_collection()
    results = collection.query(
        query_texts=[prompt],
        n_results=n_results,
        
    )
    return results

def call_llm(context: str, prompt: str):
    response = ollama.chat(
        model="llama3.2:latest",
        stream=True,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": f"Context: {context},Question: {prompt}",
            }
        ],
    )

    for chunk in response:
        if chunk["done"] is False:
            yield chunk["message"]["content"]
        else:
            break  # Return full answer instead of yielding parts


def re_rank_cross_encoders(documents: list[str], prompt: str) -> tuple[str, list[int]]:
    relevant_text = ""
    relevant_text_ids = []
    
    encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs = [(prompt, doc) for doc in documents]
    scores = encoder_model.predict(pairs)
    
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]
    for idx in top_indices:
        relevant_text += documents[idx] + " "
        relevant_text_ids.append(idx)
    
    return relevant_text.strip(), relevant_text_ids

if __name__ == "__main__":
    with st.sidebar:
        st.set_page_config(page_title="RAG Question Answer")
        uploaded_file = st.file_uploader("**Upload PDF files for QnA**", type=["pdf"], accept_multiple_files=False)
        process = st.button("Process PDF")
    
    if uploaded_file and process:
        normalize_uploaded_file_name = uploaded_file.name.replace(" ", "_").replace(".", "_")
        all_splits = process_document(uploaded_file)
        add_to_vector_collection(all_splits, normalize_uploaded_file_name)
    
    st.header("RAG Question Answer")
    prompt = st.text_area("**Ask a question related to your document:**", placeholder="Type your question here")
    ask = st.button("Ask")
    
    if ask and prompt:
        results = query_collection(prompt)
        documents_list = results.get("documents", [[]])
        
        if not documents_list or not documents_list[0]:
            st.write("No relevant information found in the document.")
        else:
            relevant_text, relevant_text_ids = re_rank_cross_encoders(documents_list[0], prompt)
            response = call_llm(context=relevant_text, prompt=prompt)
            st.write_stream(response)
