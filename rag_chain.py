from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient

import os

def ingest_documents(path="./docs/test_file.txt"):
    # Load text
    abs_path = os.path.abspath(path)
    print(f"Reading file from: {abs_path}")

    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"File not found: {abs_path}")

    loader = TextLoader(file_path=abs_path, encoding="utf-8")
    docs = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # Use Sentence Transformers
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)

    # Setup Qdrant
    client = QdrantClient(":memory:")  # For production, use localhost or cloud

    client.recreate_collection(
        collection_name="rag_docs",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

    # Store vectors
    vectorstore = Qdrant.from_documents(
    chunks,
    embedding=embedding_model,
    collection_name="rag_docs",
    location=":memory:",  # ✅ replaces the 'client' param
    )

    return vectorstore

def build_qa_chain(vectorstore):
    # Load LLM
    # llm_pipeline = pipeline("text-generation", model="google/flan-t5-large", max_new_tokens=100)
    # llm_pipeline = pipeline("text2text-generation", model="google/flan-t5-large", max_new_tokens=100) #large mode
    # llm_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=100) # base model
    llm_pipeline = pipeline("text2text-generation", model="google/flan-t5-small", max_new_tokens=1000)  # small model
    llm = HuggingFacePipeline(pipeline=llm_pipeline)

    # Build chain
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    return qa_chain