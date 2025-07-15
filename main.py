import os
import time
import hashlib
from urllib.parse import urlparse
import streamlit as st
from dotenv import load_dotenv
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Set up Streamlit UI
st.set_page_config(page_title="InsightBot", layout="wide")
st.title("InsightBot: News Research Tool \U0001F4C8")
st.sidebar.title("\U0001F517 Enter News Article URLs")

# Helper: Validate and hash URLs
def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def url_hash(urls):
    joined = "".join(sorted(urls))
    return hashlib.md5(joined.encode()).hexdigest()

# URL input
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}", placeholder="https://example.com/article")
    if url.strip():
        urls.append(url.strip())

# Deduplicate and validate URLs
urls = list(set([url for url in urls if is_valid_url(url)]))
process_url_clicked = st.sidebar.button("\U0001F680 Process URLs")
main_placeholder = st.empty()

# Load LLM
llm = OpenAI(temperature=0.7, max_tokens=500)

# Process and cache vectorstore
@st.cache_resource
def load_faiss_vectorstore(faiss_dir):
    try:
        return FAISS.load_local(faiss_dir, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    except Exception as e:
        return None

vectorstore = None
if process_url_clicked:
    if not urls:
        st.warning("Please enter at least one valid URL.")
    else:
        hash_id = url_hash(urls)
        FAISS_DIR = f"faiss_store_openai/{hash_id}"

        with st.spinner("\U0001F504 Processing URLs..."):
            try:
                loader = UnstructuredURLLoader(urls=urls)
                data = loader.load()

                # Add metadata for source tracking
                for doc, src in zip(data, urls):
                    doc.metadata["source"] = src

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=150,
                    separators=['\n\n', '\n', '.', ',']
                )
                docs = text_splitter.split_documents(data)

                embeddings = OpenAIEmbeddings()
                vectorstore = FAISS.from_documents(docs, embeddings)
                vectorstore.save_local(FAISS_DIR)
                st.success("\u2705 URLs processed and FAISS index created.")

            except Exception as e:
                st.error(f"Processing failed: {e}")
else:
    # Try loading from default if not clicked
    FAISS_DIR = f"faiss_store_openai/{url_hash(urls)}" if urls else ""
    if FAISS_DIR:
        vectorstore = load_faiss_vectorstore(FAISS_DIR)

# Query section
query = st.text_input("\U0001F50D Ask a question about the articles:")
if query:
    if vectorstore is None:
        st.warning("\u26A0 No FAISS index found. Please process URLs first.")
    else:
        with st.spinner("\U0001F916 Generating response..."):
            try:
                chain = RetrievalQAWithSourcesChain.from_llm(
                    llm=llm,
                    retriever=vectorstore.as_retriever()
                )
                result = chain({"question": query}, return_only_outputs=True)

                st.header("\U0001F4D8 Answer")
                st.write(result["answer"])

                sources = result.get("sources", "")
                if sources:
                    st.subheader("\U0001F4CE Sources")
                    for src in sources.split("\n"):
                        st.markdown(f"- [{src}]({src})")
            except Exception as e:
                st.error(f"Error generating answer: {e}")