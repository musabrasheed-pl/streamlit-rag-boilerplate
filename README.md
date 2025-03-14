# RAG Application with Streamlit UI

This repository contains a Streamlit application that implements a Retrieval-Augmented Generation (RAG) pipeline. The app allows users to upload PDF documents, process them into embeddings, and ask questions based on the document content. It is designed to be modular, letting you easily switch between different vector stores (Pinecone, FAISS, or Chroma), embedding models, and LLM providers (OpenAI, Claude, or Gemini) via environment variables.

## Features

- **Document Processing:** Extracts text from PDFs and splits it into manageable chunks using LangChain's `RecursiveCharacterTextSplitter`.
- **Vectorstore Integration:** Supports multiple vector stores:
  - **Pinecone**
  - **FAISS**
  - **Chroma**
- **Configurable LLM Integration:** Easily choose among different LLM providers:
  - **OpenAI**
  - **Claude**
  - **Gemini**
- **Interactive UI:** Upload PDFs, process files, and ask questions interactively.

## Prerequisites

- Python 3.7 or higher
- API keys for:
  - **OpenAI**
  - **Pinecone** (if using Pinecone)
  - **Claude** (if using Claude)
  - **Gemini** (if using Gemini)

## Setup

1. **Clone the Repository**

   ```bash
   git clone https://github.com/musabrasheed-pl/streamlit-rag-boilerplate.git
   cd streamlit-rag-boilerplate

2. **Create a Virtual Environment & Install Dependencies**
    
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   pip install -r requirements.txt

3. **Configure Environment Variables**

    ```bash
   # API Keys
    OPENAI_API_KEY=your_openai_api_key
    PINECONE_API_KEY=your_pinecone_api_key
    CLAUDE_API_KEY=your_claude_api_key      # Only required if using Claude
    GEMINI_API_KEY=your_gemini_api_key        # Only required if using Gemini
    
    # Configuration
    VECTORSTORE_TYPE=chroma   # Options: pinecone, faiss, chroma
    EMBEDDING_MODEL=text-embedding-3-large
    LLM_PROVIDER=openai       # Options: openai, claude, gemini
    LLM_MODEL=gpt-4o          # Or the appropriate model name for your provider

4. **Start the Streamlit App**
    
   ```bash
   streamlit run app.py


## Usage
- File Upload: Use the file uploader to select and upload one or more PDF files.
- Process Files: Click the Process Files button to extract text and initialize the vector store.
- Ask a Question: Enter your query in the provided text input and view the generated answer.