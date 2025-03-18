from os import walk
import logging
import time

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector


logging.basicConfig(level=logging.INFO)

try:
    #list out everything in ./context folder
    start_time = round(time.time() * 1000)
    files = next(walk("./context"), (None, None, []))[2]
    duration = round(time.time() * 1000) - start_time
    logging.info(f"Files: {str(files)} found within ./context folder in {duration:.0f}ms")

    #load all docs
    start_time = round(time.time() * 1000)
    docs = [TextLoader("./context/"+file).load() for file in files]
    duration = round(time.time() * 1000) - start_time
    logging.info(f"{str(len(docs))} documents loaded in {duration:.0f}ms")

    #chunk the docs up
    start_time = round(time.time() * 1000)
    docs_list = [item for sublist in docs for item in sublist]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    texts = text_splitter.split_documents(docs_list)
    duration = round(time.time() * 1000) - start_time
    logging.info(f"{str(len(texts))} chunks created in {duration:.0f}ms")

    #embed and put vectors into db
    start_time = time.time()
    embeddings = OllamaEmbeddings(model="llama3.1")
    CONNECTION_STRING = "postgresql+psycopg2://postgres:test@localhost:5432/vector_db"
    COLLECTION_NAME = "NJIT-workshop"
    vector_store = PGVector(embeddings=embeddings, collection_name=COLLECTION_NAME, connection=CONNECTION_STRING, use_jsonb=True)
    vector_store.add_documents(texts)
    duration = time.time() - start_time
    logging.info(f"Stored to db successfully in {duration:.2f}s")


except Exception as e:
        logging.info("Error during upload: "+str(e))
        raise e