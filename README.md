# 🤖 Ace: AI-Powered Study Assistant

A sophisticated RAG (Retrieval-Augmented Generation) application that enables intelligent conversations with multiple PDF documents simultaneously. Built with Streamlit, LangChain, and Ollama for local, privacy-focused document analysis.

<img width="1397" height="869" alt="ace" src="https://github.com/user-attachments/assets/f2f121e0-b5e0-4431-a318-61b1ceee2898" />



## ✨ Features

- **Multi-Document Support**: Upload and query multiple PDF files at once
- **Intelligent Document Caching**: Fast re-loading of previously processed documents
- **Balanced Multi-Source Retrieval**: Ensures fair representation from all uploaded documents
- **Conversational Memory**: Maintains chat history for contextual follow-up questions
- **Local AI Processing**: Runs entirely on your machine using Ollama - no data sent to external APIs
- **Smart Chunking**: Optimized document splitting for better context retrieval
- **Real-time Streaming**: See AI responses as they're generated
- **Document Comparison**: Ask questions across multiple sources simultaneously

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- [Ollama](https://ollama.ai/) installed and running
- At least 8GB RAM recommended

### Installation

1. **Clone or download this repository**

2. **Install required Python packages:**
```bash
pip install streamlit langchain langchain-community langchain-ollama chromadb unstructured pdf2image pillow pdfminer.six
```

3. **Install Ollama models:**
```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

### Running the Application

```bash
streamlit run rag-streamlit.py
```

The application will open in your default web browser at `http://localhost:8501`

## 📖 Usage

1. **Upload Documents**: Use the sidebar to upload one or more PDF files
2. **Wait for Processing**: Documents are automatically processed and cached (first upload takes longer)
3. **Ask Questions**: Type your questions in the chat interface
4. **Get Answers**: Receive AI-generated responses based on your uploaded documents

### Example Questions

- "What are the main topics covered in these documents?"
- "Compare the approaches discussed in document A versus document B"
- "Summarize chapter 3 from [filename]"
- "What does the text say about [specific topic]?"

## 🛠️ Configuration

You can modify these settings in the code:

```python
LLM_MODEL = "llama3.2:3b"  # Change to any Ollama model
EMBEDDING_MODEL = "nomic-embed-text"  # Embedding model for document vectors
```

### Advanced Settings

In the `RecursiveCharacterTextSplitter`:
- `chunk_size=1200`: Size of text chunks for processing
- `chunk_overlap=300`: Overlap between chunks for context preservation

In retrieval configuration:
- `total_docs_target = 8`: Number of document chunks to retrieve per query
- `docs_per_store_initial`: Initial retrieval count per document

## 📁 Project Structure

```
.
├── rag-streamlit.py          # Main application file
├── document_cache/            # Cached documents and vector stores (auto-created)
│   ├── *_documents.json      # Processed document cache
│   └── chroma_*/             # Vector database storage
└── README.md                  # This file
```

## 🔧 How It Works

1. **Document Ingestion**: PDFs are loaded using UnstructuredPDFLoader
2. **Caching**: Documents are hashed and cached to avoid reprocessing
3. **Text Splitting**: Documents are split into overlapping chunks
4. **Embedding**: Text chunks are converted to vectors using nomic-embed-text
5. **Vector Storage**: Embeddings stored in ChromaDB for fast retrieval
6. **Balanced Retrieval**: Custom retriever ensures equal representation from all sources
7. **RAG Chain**: LangChain orchestrates retrieval and generation
8. **Streaming Response**: Answers stream in real-time via ChatOllama

## 🎯 Key Features Explained

### Document Caching
Documents are cached based on content hash, making subsequent loads nearly instantaneous. Cache persists between sessions.

### Balanced Multi-Document Retrieval
The custom `BalancedMultiDocumentRetriever` ensures that when querying multiple documents, each document receives fair representation in the context, preventing bias toward any single source.

### Chat History
The application maintains conversation context, allowing for natural follow-up questions without repeating information.

## 🐛 Troubleshooting

**Models not found**: Run `ollama pull llama3.2:3b` and `ollama pull nomic-embed-text`

**Out of memory**: Try using a smaller model like `llama3.2:1b` or reduce `chunk_size`

**Slow processing**: First-time document processing is slower; subsequent loads use cache

**Import errors**: Ensure all dependencies are installed: `pip install -r requirements.txt`

## 🔒 Privacy & Security

- All processing happens locally on your machine
- No data is sent to external servers
- Documents are stored only in the local cache directory
- Delete the `document_cache/` folder to remove all cached data

## 📝 License

This project is provided as-is for educational and personal use.

## 🤝 Contributing

Suggestions and improvements are welcome! Feel free to fork and submit pull requests.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [LangChain](https://langchain.com/)
- Uses [Ollama](https://ollama.ai/) for local LLM inference
- Vector storage via [ChromaDB](https://www.trychroma.com/)

---

**Note**: This application requires sufficient system resources to run local LLMs. Performance varies based on your hardware and the selected model size.
