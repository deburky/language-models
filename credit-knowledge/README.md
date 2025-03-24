# 📑 Credit Knowledge AI App

**Author:** [Denis Burakov (@deburky)](https://github.com/deburky)  

This is an**AI-powered API** that provides access to knowledge bases using **Retrieval-Augmented Generation (RAG)** and **OpenAI** foundation models.  

We use **FastAPI, LangChain, SQLiteVSS**, and **sentence-transformer embeddings** to answer specific questions on the subject. The knowledge base consists of **LinkedIn articles on Credit Risk** written between **2022 and 2025**.

**LangServe** (from LangChain) is meant to serve LangChain Chains, Tools, and Agents over HTTP APIs—especially using FastAPI. It provides automatic OpenAPI docs and neat integration paths for deployment.

---
## 📦 Key Features

✅ **Retrieval-Augmented Generation (RAG):** Finds relevant documents and provides AI-powered answers.  
✅ **OpenAI-Powered Responses:** Uses `GPT-3.5-Turbo` for accurate, concise explanations.  
✅ **Vector Storage with SQLiteVSS:** Efficient similarity search using `all-MiniLM-L6-v2` embeddings.  
✅ **Supports Multiple Data Formats:** Works with **text files**, but can easily be customized to work with other formats (PDFs etc).  
✅ **API-First Design:** Exposes a **REST API** for seamless integration with other applications.  


## ✨ How It Works

1️⃣ **Embedding Knowledge:**  
   - Articles and documents are embedded using `all-MiniLM-L6-v2` and stored in an **SQLiteVSS vector database**.  

2️⃣ **Retrieving Context:**  
   - When a **query** is submitted, the app searches for **relevant documents** using similarity search.  

3️⃣ **Generating Answers:**  
   - The **retrieved context** is passed to OpenAI's **GPT-3.5-Turbo**, which generates an intelligent response.  

## 💻 How to Use

Create the vector database:

```zsh
uv run create_db.py
```

Start the LangServe FastAPI app:

```zsh
uv run app.py
```

Using FastAPI router:

```zsh
curl -X POST http://localhost:8000/query \
   -H "Content-Type: application/json" \
   -d '{"query": "What is AI underwriter?"}'
```

Using RAG Chain from LangServe:

```zsh
curl -X POST http://localhost:8000/rag-chain/invoke \
  -H "Content-Type: application/json" \
  -d '{"input": "What is AI underwriter?"}'
```

## 🛠️ LangChain & Vector Storage

### 🔧 Install Python 3.9 with SQLite Extensions (macOS)

In order to use a vector storage with [**`sqlite-vss`**](https://python.langchain.com/docs/integrations/vectorstores/sqlitevss/) or [**`sqlite-vec`**](https://python.langchain.com/docs/integrations/vectorstores/sqlitevec/) vector databases in Langchain, we need a version of Python that supports sqlite with extensions.

To use SQLite with extensions on macOS, install Python 3.9 as follows:

```zsh
brew update && brew upgrade pyenv
```

Install Python with configuration options:

```zsh
PYTHON_CONFIGURE_OPTS="--enable-loadable-sqlite-extensions --with-openssl=$(brew --prefix openssl)" \
LDFLAGS="-L$(brew --prefix sqlite)/lib" \
CPPFLAGS="-I$(brew --prefix sqlite)/include" \
PKG_CONFIG_PATH="$(brew --prefix sqlite)/lib/pkgconfig" \
pyenv install 3.9.18
```

Set this version to be used globally:

```zsh
pyenv global 3.9.18
```

Test if the extension is available:

```zsh
python -c "import sqlite3; print(sqlite3.connect(':memory:').execute('PRAGMA compile_options').fetchall())"
```

You should be able to see `('OMIT_LOAD_EXTENSION',)` in the output.

Pin this Python version with `uv` and recreate a virtual environment:

```python
uv python pin 3.9.18
```

## 🔑 OpenAI & DeepSeek API Keys

We will need an OpenAI API key to work with GPT models from OpenAI.

To set up the API key, create a `.env` file in the root of the project and add the following line:

```zsh
OPENAI_API_KEY='sk-...'
DEEPSEEK_API_KEY='sk...'
```

Do not forget to add the `.env` file to the `.gitignore` file.

## 📄 Future work: working with PDFs

Below we provide instructions for the embedding generation using a pdf reader.

Example of loading a PDF file and splitting it into pages:

```python
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.vectorstores.faiss import FAISS

loader = PyPDFLoader("document.pdf")
pages = loader.load_and_split()

# Embed the pages
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(pages, embeddings)

q = "Why is LGD important?"
db.similarity_search(q)[0]
```

Additionally, we can use the retriever:

```python
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.llms.openai import OpenAI
llm = OpenAI()
chain = RetrievalQA.from_llm(llm=llm, retriever=db.as_retriever())
q = "What is model profitability?"
chain(q, return_only_outputs=True)
```