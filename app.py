import os
import time
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from pinecone import Pinecone, ServerlessSpec

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS, Chroma, Pinecone as LangchainPinecone
from langchain_community.chat_models import ChatOpenAI, ChatGooglePalm, ChatAnthropic

# Load environment variables from the .env file
load_dotenv()

# === Configuration Variables ===
VECTORSTORE_TYPE = os.getenv("VECTORSTORE_TYPE", "pinecone").lower()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# ====================================
#          RAG Application Class
# ====================================
class RAGApp:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.validate_key(self.openai_api_key, "OPENAI_API_KEY")
        self.embedding = OpenAIEmbeddings(openai_api_key=self.openai_api_key, model=EMBEDDING_MODEL)

    @staticmethod
    def validate_key(key, key_name):
        if not key:
            st.error(f"Please set up your {key_name} in the .env file")
            st.stop()

    def process_files(self, files):
        """Process uploaded PDF files and split text into chunks."""
        all_chunks = []
        for file in files:
            with st.spinner(f"Processing {file.name}..."):
                if not file.name.endswith(".pdf"):
                    st.warning(f"Unsupported file type: {file.name}")
                    continue
                text = self.extract_text_from_pdf(file)
                chunks = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128).split_text(text)
                all_chunks.extend([(chunk, file.name) for chunk in chunks])
        return all_chunks

    @staticmethod
    def extract_text_from_pdf(file):
        pdf_reader = PdfReader(file)
        return "".join(page.extract_text() or "" for page in pdf_reader.pages)

    def initialize_vectorstore(self, chunks):
        """Initialize vectorstore based on configuration."""
        texts = [chunk for chunk, _ in chunks]
        metadatas = [{"source": filename} for _, filename in chunks]

        if VECTORSTORE_TYPE == "pinecone":
            vectorstore = self.initialize_pinecone_vectorstore()
            vectorstore.add_texts(texts, metadatas=metadatas)  # ✅ Store embeddings in Pinecone
            st.write(f"Pinecone index updated with {len(texts)} embeddings.")
            return vectorstore
        elif VECTORSTORE_TYPE == "faiss":
            return FAISS.from_texts(texts, self.embedding)
        elif VECTORSTORE_TYPE == "chroma":
            return Chroma.from_texts(texts, self.embedding, collection_name="documents")
        else:
            st.error("Unsupported VECTORSTORE_TYPE. Please use 'pinecone', 'faiss', or 'chroma'.")
            st.stop()

    def initialize_pinecone_vectorstore(self):
        self.validate_key(os.getenv("PINECONE_API_KEY"), "PINECONE_API_KEY")
        index_name = "video-transcripts"
        dimension = 3072

        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        if index_name not in [index.name for index in pc.list_indexes()]:
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
            self.wait_for_index(pc, index_name)

        index = pc.Index(index_name)
        return LangchainPinecone(index, self.embedding, "text")

    @staticmethod
    def wait_for_index(pc, index_name):
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)

    def setup_qa_chain(self, vectorstore):
        """Set up the QA retrieval chain using the configured LLM and vectorstore."""
        llm = self.initialize_llm()
        prompt = self.create_prompt()
        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={'k': 5}),
            return_source_documents=True,
            verbose=True,
            combine_docs_chain_kwargs={"prompt": prompt}
        )

    def initialize_llm(self):
        if LLM_PROVIDER == "openai":
            return ChatOpenAI(temperature=0.3, model_name=LLM_MODEL, openai_api_key=self.openai_api_key)
        elif LLM_PROVIDER == "claude":
            return self.initialize_claude_llm()
        elif LLM_PROVIDER == "gemini":
            return self.initialize_gemini_llm()
        else:
            st.error("Unsupported LLM_PROVIDER. Use 'openai', 'claude', or 'gemini'.")
            st.stop()

    def initialize_claude_llm(self):
        self.validate_key(os.getenv("CLAUDE_API_KEY"), "CLAUDE_API_KEY")
        return ChatAnthropic(temperature=0.3, model=LLM_MODEL, anthropic_api_key=os.getenv("CLAUDE_API_KEY"))

    def initialize_gemini_llm(self):
        self.validate_key(os.getenv("GEMINI_API_KEY"), "GEMINI_API_KEY")
        return ChatGooglePalm(temperature=0.3, model_name=LLM_MODEL, google_api_key=os.getenv("GEMINI_API_KEY"))

    @staticmethod
    def create_prompt():
        return PromptTemplate(
            input_variables=["history", "context", "question"],
            template="""
            You're an assistant that answers questions strictly based on the provided documents.

            Conversation history:
            {history}

            Context from documents:
            {context}

            Question:
            {question}

            Answer:
            """
        )

    def answer_query(self, vectorstore, query):
        qa_chain = self.setup_qa_chain(vectorstore)
        history_text = "\n".join([f"User: {q}\nAssistant: {a}" for q, a in st.session_state.conversation_history])
        result = qa_chain({
            "question": query,
            "history": history_text,
            "chat_history": st.session_state.conversation_history
        })
        st.session_state.conversation_history.append((query, result["answer"]))
        return result["answer"]

# ====================================
#                Main
# ====================================
def main():
    st.set_page_config(layout="wide")

    # Sidebar
    with st.sidebar:
        st.title("📝 Conversation History")
        st.markdown("Scroll through past interactions.")
        st.markdown("---")
        if st.session_state.conversation_history:
            for user_query, bot_response in reversed(st.session_state.conversation_history):
                st.markdown(f"**User:** {user_query}")
                st.markdown(f"**Assistant:** {bot_response}")
                st.markdown("---")
        else:
            st.info("No conversation history yet.")

    st.title("🔍 RAG Application Boilerplate")
    st.markdown(
        f"**Vectorstore:** {VECTORSTORE_TYPE} | **Embedding:** {EMBEDDING_MODEL} | "
        f"**LLM Provider:** {LLM_PROVIDER} | **LLM Model:** {LLM_MODEL}"
    )

    app = RAGApp()

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    st.subheader("📂 Upload & Process Documents")
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

    if st.button("Process Files") and uploaded_files:
        chunks = app.process_files(uploaded_files)
        if chunks:
            st.success("Files processed successfully! Vectorstore initialized.")
            st.session_state.vectorstore = app.initialize_vectorstore(chunks)
        else:
            st.error("Failed to process files.")

    st.subheader("💬 Ask a Question")
    query = st.text_input("Type your query here:")
    if query:
        if st.session_state.vectorstore:
            answer = app.answer_query(st.session_state.vectorstore, query)
            st.success("Answer generated:")
            st.write(answer)
        else:
            st.error("Vectorstore is not initialized. Please process files first.")

if __name__ == "__main__":
    main()
