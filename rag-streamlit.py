import streamlit as st
import os
import logging
from datetime import datetime
import tempfile
import re
import json
import hashlib
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

# LangChain components
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.documents import Document

# Ollama integration
import ollama

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

LLM_MODEL = "llama3.2:3b"
EMBEDDING_MODEL = "nomic-embed-text"

# Create cache directory for faster processing
CACHE_DIR = Path("./document_cache")
CACHE_DIR.mkdir(exist_ok=True)

# Page configuration
st.set_page_config(
    page_title="Ace - AI-Powered Study Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Fast Processing & Caching Functions ---

def get_document_hash(uploaded_file):
    """Generate a unique hash for the document based on content and metadata."""
    content = uploaded_file.getvalue()
    file_info = f"{uploaded_file.name}_{uploaded_file.size}_{len(content)}"
    return hashlib.md5(content + file_info.encode()).hexdigest()

def save_document_cache(doc_hash, documents, metadata):
    """Save processed document to cache for faster future loading."""
    try:
        cache_file = CACHE_DIR / f"{doc_hash}_documents.json"
        cache_data = {
            "documents": [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in documents],
            "file_metadata": metadata,
            "created_at": datetime.now().isoformat(),
            "model_used": EMBEDDING_MODEL
        }
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        
        logging.info(f"Cached processed documents for hash {doc_hash}")
        return True
    except Exception as e:
        logging.error(f"Failed to cache documents: {e}")
        return False

def load_document_cache(doc_hash):
    """Load processed document from cache if available."""
    try:
        cache_file = CACHE_DIR / f"{doc_hash}_documents.json"
        if cache_file.exists():
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Reconstruct Document objects
            documents = [Document(page_content=doc["page_content"], metadata=doc["metadata"]) 
                        for doc in cache_data["documents"]]
            
            logging.info(f"Loaded cached documents for hash {doc_hash}")
            return documents, cache_data["file_metadata"]
        return None, None
    except Exception as e:
        logging.error(f"Failed to load cached documents: {e}")
        return None, None

def check_vector_store_cache(doc_hash):
    """Check if vector store exists in cache."""
    vector_store_path = CACHE_DIR / f"chroma_{doc_hash}"
    return vector_store_path.exists()

def load_cached_vector_store(doc_hash):
    """Load cached vector store."""
    try:
        vector_store_path = CACHE_DIR / f"chroma_{doc_hash}"
        if vector_store_path.exists():
            embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)
            vector_db = Chroma(
                persist_directory=str(vector_store_path),
                embedding_function=embedding,
                collection_name=f"doc_{doc_hash}"
            )
            logging.info(f"Loaded cached vector store for hash {doc_hash}")
            return vector_db
    except Exception as e:
        logging.error(f"Failed to load cached vector store: {e}")
    return None

# --- Enhanced RAG Functions ---

def ingest_pdf_fast(uploaded_file):
    """Enhanced PDF ingestion with caching and better metadata extraction."""
    if not uploaded_file:
        return None, None, None
    
    # Check cache first
    doc_hash = get_document_hash(uploaded_file)
    cached_docs, cached_metadata = load_document_cache(doc_hash)
    
    if cached_docs and cached_metadata:
        st.success(f"📋 Loaded '{uploaded_file.name}' from cache (instant)")
        return cached_docs, cached_metadata, doc_hash
    
    # Process new document
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        loader = UnstructuredPDFLoader(file_path=tmp_file_path)
        data = loader.load()
        
        # Enhanced metadata
        metadata = {
            "filename": uploaded_file.name,
            "size": uploaded_file.size,
            "size_mb": round(uploaded_file.size / (1024 * 1024), 2),
            "pages": len(data),
            "processed_at": datetime.now().isoformat(),
            "doc_hash": doc_hash,
            "total_characters": sum(len(doc.page_content) for doc in data)
        }
        
        # Cache the processed documents
        save_document_cache(doc_hash, data, metadata)
        
        logging.info(f"PDF '{uploaded_file.name}' processed: {len(data)} pages, {metadata['total_characters']} characters")
        return data, metadata, doc_hash
        
    except Exception as e:
        logging.error(f"Error loading PDF: {e}")
        st.error(f"Error processing PDF: {e}")
        return None, None, None
    finally:
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

def process_and_store_documents_fast(documents, doc_hash, metadata):
    """Enhanced document processing with caching and optimized chunking."""
    
    # Check if vector store already exists
    if check_vector_store_cache(doc_hash):
        vector_db = load_cached_vector_store(doc_hash)
        if vector_db:
            return vector_db
    
    # Process documents with optimized chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, 
        chunk_overlap=300,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Add source filename to each chunk's metadata
    for chunk in chunks:
        chunk.metadata["source_file"] = metadata["filename"]
        chunk.metadata["doc_hash"] = doc_hash
    
    try:
        # Ensure embedding model is available
        ollama.pull(EMBEDDING_MODEL)
        embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)
        
        # Create vector store with persistence
        persist_directory = CACHE_DIR / f"chroma_{doc_hash}"
        
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            collection_name=f"doc_{doc_hash}",
            persist_directory=str(persist_directory)
        )
        
        logging.info(f"Vector database created for '{metadata['filename']}' with {len(chunks)} chunks.")
        
    except Exception as e:
        logging.error(f"Vector store creation failed: {e}")
        return None
    
    return vector_db

def create_multi_document_retriever(document_stores):
    """Create a balanced retriever that searches across multiple documents equally."""
    if len(document_stores) == 1:
        return list(document_stores.values())[0]["vector_db"].as_retriever(search_kwargs={"k": 6})
    
    # Enhanced multi-document retriever with balanced search
    class BalancedMultiDocumentRetriever:
        def __init__(self, document_stores):
            self.document_stores = document_stores
        
        def get_relevant_documents(self, query: str):
            """Retrieve documents with balanced representation from all sources."""
            all_docs = []
            
            # Calculate how many docs to retrieve from each source
            total_docs_target = 8
            num_stores = len(self.document_stores)
            
            # Get more docs initially to allow for ranking
            docs_per_store_initial = max(3, total_docs_target // num_stores + 2)
            
            # Collect scored documents from each source
            scored_docs_by_source = {}
            
            for doc_hash, doc_info in self.document_stores.items():
                try:
                    retriever = doc_info["vector_db"].as_retriever(
                        search_kwargs={"k": docs_per_store_initial}
                    )
                    docs = retriever.get_relevant_documents(query)
                    
                    filename = doc_info["metadata"]["filename"]
                    
                    # Score documents (using their retrieval order as a proxy for relevance)
                    scored_docs = []
                    for idx, doc in enumerate(docs):
                        # Add source filename to metadata
                        if "source_file" not in doc.metadata:
                            doc.metadata["source_file"] = filename
                        
                        # Score based on position (higher score for earlier docs)
                        score = 1.0 / (idx + 1)
                        scored_docs.append((score, doc))
                    
                    scored_docs_by_source[filename] = scored_docs
                    
                except Exception as e:
                    logging.error(f"Error retrieving from {doc_info['metadata']['filename']}: {e}")
                    continue
            
            # Interleave documents from different sources for balanced representation
            # This ensures each document gets fair representation
            max_docs_per_source = total_docs_target // num_stores + 1
            docs_used_per_source = {source: 0 for source in scored_docs_by_source.keys()}
            
            # First pass: add top document from each source
            for source, scored_docs in scored_docs_by_source.items():
                if scored_docs and docs_used_per_source[source] < max_docs_per_source:
                    all_docs.append(scored_docs[0][1])  # Add the document (not the score)
                    docs_used_per_source[source] += 1
            
            # Second pass: add remaining documents in round-robin fashion
            round_robin_index = 1
            while len(all_docs) < total_docs_target:
                added_in_round = False
                for source, scored_docs in scored_docs_by_source.items():
                    if len(all_docs) >= total_docs_target:
                        break
                    if round_robin_index < len(scored_docs) and docs_used_per_source[source] < max_docs_per_source:
                        all_docs.append(scored_docs[round_robin_index][1])
                        docs_used_per_source[source] += 1
                        added_in_round = True
                
                if not added_in_round:
                    break
                round_robin_index += 1
            
            logging.info(f"Retrieved {len(all_docs)} docs from {num_stores} sources. Distribution: {docs_used_per_source}")
            
            return all_docs[:total_docs_target]
        
        def invoke(self, inputs):
            """Support for LangChain invoke interface."""
            if isinstance(inputs, str):
                return self.get_relevant_documents(inputs)
            elif isinstance(inputs, dict):
                return self.get_relevant_documents(inputs.get("query", inputs.get("input", "")))
            return []
    
    return BalancedMultiDocumentRetriever(document_stores)

def get_multi_document_rag_chain(document_stores):
    """Creates and returns a RAG chain that works with multiple documents."""
    llm = ChatOllama(model=LLM_MODEL, temperature=0.1)
    
    # Create the balanced multi-document retriever
    base_retriever = create_multi_document_retriever(document_stores)
    
    # Enhanced QA system prompt for multiple documents
    doc_list = ", ".join([info["metadata"]["filename"] for info in document_stores.values()])
    
    qa_system_prompt = f"""You are a helpful study assistant with access to these specific documents: {doc_list}

CRITICAL INSTRUCTIONS:
- Answer questions based ONLY on the actual content provided in the context below
- If the context doesn't contain information to answer the question, say "I don't have information about that in the uploaded documents"
- NEVER make up or assume information not in the context
- When information appears in multiple documents, synthesize and compare across sources
- Always cite which document(s) you're drawing information from when answering
- Pay equal attention to all provided documents - don't favor one over others

Context from uploaded documents:
{{context}}

Previous conversation:
{{chat_history}}

Current question: {{input}}

Answer:"""

    qa_prompt = ChatPromptTemplate.from_template(qa_system_prompt)
    
    # Create the QA chain
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # Create a simplified RAG chain that properly handles documents
    class SimpleMultiDocRAGChain:
        def __init__(self, retriever, qa_chain):
            self.retriever = retriever
            self.qa_chain = qa_chain
        
        def stream(self, inputs):
            query = inputs["input"]
            chat_history = inputs.get("chat_history", [])
            
            # Get relevant documents
            docs = self.retriever.get_relevant_documents(query)
            
            # Format chat history
            chat_history_text = ""
            if chat_history:
                for msg in chat_history[-4:]:  # Last 4 messages for context
                    if isinstance(msg, HumanMessage):
                        chat_history_text += f"Human: {msg.content}\n"
                    elif isinstance(msg, AIMessage):
                        chat_history_text += f"Assistant: {msg.content}\n"
            
            # Pass documents directly to the QA chain (not as formatted strings)
            formatted_input = {
                "context": docs,  # Pass Document objects, not strings
                "input": query,
                "chat_history": chat_history_text
            }
            
            # Stream the response
            for chunk in self.qa_chain.stream(formatted_input):
                yield {"answer": chunk}
    
    rag_chain = SimpleMultiDocRAGChain(base_retriever, question_answer_chain)
    
    logging.info(f"Multi-document RAG chain created for {len(document_stores)} documents with balanced retrieval.")
    return rag_chain

# --- Main Application UI ---

def main():
    st.markdown("<h1 style='text-align: center;'>🤖 Ace: Your AI-Powered Study Assistant</h1>", unsafe_allow_html=True)

    # --- Sidebar ---
    with st.sidebar:
        st.header("📁 Documents")
        
        # Multi-file uploader
        uploaded_files = st.file_uploader(
            "Upload your PDFs here:",
            type="pdf",
            accept_multiple_files=True,
            help="You can upload multiple PDF files to study from all of them simultaneously"
        )

        # Display loaded documents
        if uploaded_files:
            st.subheader("📚 Document Library")
            total_size = sum(f.size for f in uploaded_files)
            st.info(f"{len(uploaded_files)} documents ({total_size / (1024*1024):.1f} MB)")
            
            for i, file in enumerate(uploaded_files):
                doc_hash = get_document_hash(file)
                size_mb = file.size / (1024 * 1024)
                
                # Check if document is processed
                is_ready = doc_hash in st.session_state.document_stores
                status_icon = "✅" if is_ready else "⏳"
                status_text = "Ready" if is_ready else "Processing..."
                
                st.write(f"{status_icon} **{file.name}** ({size_mb:.1f} MB) - {status_text}")

        st.markdown("---")
        st.header("💬 Controls")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                if "messages" in st.session_state:
                    st.session_state.messages = []
                st.rerun()
        
        with col2:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()
        
        st.markdown("---")
        st.header("💡 Usage Tips")
        st.markdown("""
        - Ask questions about any or all of your documents
        - Request comparisons between documents
        - Ask for summaries across multiple sources
        - Use natural, conversational language
        - Documents are cached for faster reloading
        """)
        
        st.markdown("---")
        
        # Performance info
        if "document_stores" in st.session_state and st.session_state.document_stores:
            st.header("📊 Session Info")
            doc_count = len(st.session_state.document_stores)
            total_chunks = 0
            for doc_info in st.session_state.document_stores.values():
                if doc_info.get("vector_db"):
                    # Estimate chunks based on document size
                    total_chunks += doc_info["metadata"].get("pages", 0) * 3  # Rough estimate
            
            st.metric("Documents Loaded", doc_count)
            st.metric("Searchable Chunks", total_chunks if total_chunks > 0 else "Processing...")
        
        st.info(f"🤖 Using model: `{LLM_MODEL}`")

    # --- Session State Initialization ---
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "document_stores" not in st.session_state:
        st.session_state.document_stores = {}
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None

    # --- Document Processing Logic ---
    
    if uploaded_files:
        # Get current document hashes
        current_doc_hashes = set()
        files_to_process = []
        
        for uploaded_file in uploaded_files:
            doc_hash = get_document_hash(uploaded_file)
            current_doc_hashes.add(doc_hash)
            
            if doc_hash not in st.session_state.document_stores:
                files_to_process.append((uploaded_file, doc_hash))
        
        # Remove documents that are no longer uploaded
        removed_docs = set(st.session_state.document_stores.keys()) - current_doc_hashes
        for doc_hash in removed_docs:
            del st.session_state.document_stores[doc_hash]
            logging.info(f"Removed document with hash {doc_hash}")
        
        # Process new documents
        if files_to_process:
            for uploaded_file, doc_hash in files_to_process:
                # Process silently in background - no main screen loading bars
                documents, metadata, doc_hash = ingest_pdf_fast(uploaded_file)
                
                if documents and metadata:
                    vector_db = process_and_store_documents_fast(documents, doc_hash, metadata)
                    
                    if vector_db:
                        st.session_state.document_stores[doc_hash] = {
                            "vector_db": vector_db,
                            "metadata": metadata,
                            "documents": documents
                        }
                        # Rerun to update sidebar status
                        st.rerun()
                    else:
                        st.error(f"Failed to process {uploaded_file.name}")
                else:
                    st.error(f"Could not load {uploaded_file.name}")
        
        # Create or update RAG chain when documents change
        if st.session_state.document_stores and (
            not st.session_state.rag_chain or 
            len(files_to_process) > 0 or 
            len(removed_docs) > 0
        ):
            with st.spinner("🔗 Setting up multi-document AI assistant..."):
                st.session_state.rag_chain = get_multi_document_rag_chain(st.session_state.document_stores)

    # --- Chat Interface ---
    
    # Chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        if st.session_state.rag_chain is None:
            st.warning("Please upload documents and wait for them to be processed first.")
            return

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                # Format chat history for LangChain
                chat_history = []
                for msg in st.session_state.messages[:-1]:
                    if msg["role"] == "user":
                        chat_history.append(HumanMessage(content=msg["content"]))
                    else:
                        chat_history.append(AIMessage(content=msg["content"]))
                
                # Define a generator to yield only the 'answer' chunks for st.write_stream
                def response_generator():
                    stream = st.session_state.rag_chain.stream(
                        {"input": prompt, "chat_history": chat_history}
                    )
                    for chunk in stream:
                        if hasattr(chunk, 'get'):
                            if answer_chunk := chunk.get("answer"):
                                yield answer_chunk
                        else:
                            # Handle direct content streaming
                            yield str(chunk)
                
                # Use st.write_stream to display and collect the full response
                full_response = st.write_stream(response_generator)
                
                # Append the complete message to the session state
                st.session_state.messages.append({"role": "assistant", "content": full_response})

            except Exception as e:
                error_message = f"Sorry, I encountered an error: {e}"
                st.error(error_message)
                logging.error(f"Error during chain invocation: {e}")
                st.session_state.messages.append({"role": "assistant", "content": error_message})
    
    # --- Welcome Section ---
    # Only show when no files are uploaded and no documents are stored
    if not uploaded_files and not st.session_state.document_stores:
        col1, col2, col3 = st.columns([0.5, 3, 0.5])
        with col2:
            st.markdown("<h3 style='text-align: center; margin-top: 2rem;'>🌟 What you can do:</h3>", unsafe_allow_html=True)
            st.markdown("""
            <ul style='list-style: none; padding: 0; margin: 0; text-align: center;'>
                <li style='margin: 8px 0;'>- Ask questions about specific chapters or sections</li>
                <li style='margin: 8px 0;'>- Get summaries and clarify complex concepts</li>
                <li style='margin: 8px 0;'>- Compare information across multiple documents</li>
                <li style='margin: 8px 0;'>- Find connections between different sources</li>
                <li style='margin: 8px 0;'>- Get comprehensive answers from all your materials</li>
            </ul>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()