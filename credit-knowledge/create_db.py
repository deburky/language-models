"""create_db.py."""

import os

from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import SQLiteVSS
from langchain_text_splitters import CharacterTextSplitter

# # load the document and split it into chunks
loader = TextLoader("documents/document.txt")
documents = loader.load()

# # split it into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
texts = [doc.page_content for doc in docs]

DB_NAME = "vss.db"

if not os.path.exists("db"):
    os.makedirs("db")

# # create the open-source embedding function
embedding_function =  HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# # load it in sqlite-vss
db = SQLiteVSS.from_texts(
    texts=texts,
    embedding=embedding_function,
    table="document",
    db_file="db/vss.db",
    metadatas=[
        {"source": "finance_docs"} for _ in texts
    ],  # Ensure it's a list of dictionaries
)
