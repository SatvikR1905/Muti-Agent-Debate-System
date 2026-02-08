# src/rag_pipeline.py

from __future__ import annotations

import os
from typing import Optional

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.retrievers import BaseRetriever


def _log(msg: str) -> None:
    print(f"[RAG] {msg}", flush=True)


def load_documents(kb_directory: str):
    if not os.path.exists(kb_directory) or not os.path.isdir(kb_directory):
        _log(f"Knowledge directory not found: {kb_directory}")
        return []

    _log(f"Loading PDFs from: {kb_directory}")
    loader = DirectoryLoader(
        kb_directory,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True,
    )
    docs = loader.load()
    _log(f"Loaded {len(docs)} document pages.")
    return docs


def split_into_chunks(documents, chunk_size: int, chunk_overlap: int):
    if not documents:
        return []

    _log(f"Splitting into chunks (chunk_size={chunk_size}, overlap={chunk_overlap})")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    _log(f"Created {len(chunks)} chunks.")
    return chunks


def create_embeddings(embedding_model: str) -> Optional[OllamaEmbeddings]:
    try:
        _log(f"Initializing OllamaEmbeddings(model={embedding_model})")
        return OllamaEmbeddings(model=embedding_model)
    except Exception as e:
        _log(f"Failed to create embeddings: {e}")
        return None


def load_vector_store(vector_store_path: str, embeddings: OllamaEmbeddings) -> Optional[Chroma]:
    if not os.path.exists(vector_store_path) or not os.path.isdir(vector_store_path):
        _log(f"Vector store path not found: {vector_store_path}")
        return None

    try:
        _log(f"Loading Chroma vector store from: {vector_store_path}")
        vs = Chroma(persist_directory=vector_store_path, embedding_function=embeddings)
        
        _ = vs._collection.count()
        _log("Vector store loaded successfully.")
        return vs
    except Exception as e:
        _log(f"Failed to load vector store: {e}")
        return None


def create_vector_store(chunks, vector_store_path: str, embeddings: OllamaEmbeddings) -> Optional[Chroma]:
    if not chunks:
        _log("No chunks provided. Cannot create vector store.")
        return None

    try:
        _log(f"Creating new Chroma store at: {vector_store_path}")
        vs = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=vector_store_path,
        )
        _log("Vector store created and persisted.")
        return vs
    except Exception as e:
        _log(f"Failed to create vector store: {e}")
        return None


def get_retriever(vector_store: Optional[Chroma], k: int = 3) -> Optional[BaseRetriever]:
    if not vector_store:
        return None
    try:
        _log(f"Creating retriever (k={k})")
        return vector_store.as_retriever(search_kwargs={"k": k})
    except Exception as e:
        _log(f"Failed to create retriever: {e}")
        return None


def index_knowledge_base(
    kb_directory: str,
    vector_store_path: str,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
) -> Optional[Chroma]:
    embeddings = create_embeddings(embedding_model)
    if not embeddings:
        _log("Embedding init failed. Aborting KB setup.")
        return None

    # Try load existing store
    if os.path.exists(vector_store_path) and os.path.isdir(vector_store_path) and os.listdir(vector_store_path):
        vs = load_vector_store(vector_store_path, embeddings)
        if vs:
            _log("Using existing vector store (skipping indexing).")
            return vs
        else:
            _log("Existing store found but failed to load. Will re-index.")

    # Otherwise build new store
    docs = load_documents(kb_directory)
    if not docs:
        _log("No PDFs found (or failed to load). Aborting indexing.")
        return None

    chunks = split_into_chunks(docs, chunk_size, chunk_overlap)
    if not chunks:
        _log("Chunking produced 0 chunks. Aborting indexing.")
        return None

    # Ensure path exists
    os.makedirs(vector_store_path, exist_ok=True)

    vs = create_vector_store(chunks, vector_store_path, embeddings)
    if not vs:
        _log("Vector store creation failed.")
        return None

    # Reload to verify
    verified = load_vector_store(vector_store_path, embeddings)
    if verified:
        _log("Vector store verified by reload.")
        return verified

    _log("Warning: store created but verification reload failed.")
    return vs
