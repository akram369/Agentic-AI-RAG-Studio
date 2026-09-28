import os
import sys
import time
from pathlib import Path
from typing import Optional

# Ensure UTF-8 output on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from src.config import (
    OPENAI_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_CLOUD,
    PINECONE_REGION,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    DATA_PDF_PATH,
)


def ensure_pinecone_index(
    index_name: str,
    dimension: int = EMBEDDING_DIMENSION,
    cloud: str = PINECONE_CLOUD,
    region: str = PINECONE_REGION,
) -> Pinecone:
    """Ensure that the target Pinecone serverless index exists."""
    if not PINECONE_API_KEY:
        raise ValueError(
            "PINECONE_API_KEY is not set. Please add it to your .env file or environment variables."
        )

    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing_indexes = pc.list_indexes().names()

    if index_name not in existing_indexes:
        print(f"Creating new Pinecone index: '{index_name}' (dimension={dimension}, metric=cosine)...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        # Wait for index readiness
        while not pc.describe_index(index_name).status["ready"]:
            print("Waiting for Pinecone index to be ready...")
            time.sleep(2)
        print(f"Index '{index_name}' is ready!")
    else:
        print(f"Pinecone index '{index_name}' already exists.")

    return pc


def run_ingestion(
    pdf_path: Optional[str] = None,
    index_name: Optional[str] = None,
    batch_size: int = 100,
):
    """
    Ingests the specified PDF, splits it into chunks, generates embeddings,
    and indexes them inside Pinecone.
    """
    pdf_file = Path(pdf_path) if pdf_path else DATA_PDF_PATH
    idx_name = index_name or PINECONE_INDEX_NAME

    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF source document not found at: {pdf_file}")

    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set. Please add it to your .env file.")

    print(f"Step 1: Loading PDF from {pdf_file}...")
    loader = PyPDFLoader(str(pdf_file))
    docs = loader.load()
    print(f"Loaded {len(docs)} pages from document.")

    print(f"Step 2: Splitting document into chunks (chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Document split into {len(chunks)} text chunks.")

    print(f"Step 3: Setting up Pinecone index '{idx_name}'...")
    ensure_pinecone_index(idx_name)

    print(f"Step 4: Generating embeddings ({EMBEDDING_MODEL}) and upserting into Pinecone...")
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY,
    )

    # Ingest chunks into PineconeVectorStore in batches for stability
    vector_store = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=idx_name,
    )

    print(f"Ingestion complete! Successfully indexed {len(chunks)} chunks into '{idx_name}'.")
    return vector_store


if __name__ == "__main__":
    try:
        run_ingestion()
    except Exception as e:
        print(f"Error during ingestion: {e}", file=sys.stderr)
        sys.exit(1)
