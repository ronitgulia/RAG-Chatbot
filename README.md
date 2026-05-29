# RAG Chatbot

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LangChain](https://img.shields.io/badge/LangChain-latest-green)
![Groq](https://img.shields.io/badge/Groq-Free-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

A production-grade **Retrieval-Augmented Generation (RAG)** chatbot built with LangChain, FAISS, Groq, and Streamlit. Upload documents or scrape websites and chat with your knowledge base using free LLMs.

## Features

- **Document Loaders** — PDF (PyPDF2), DOCX, TXT, Markdown
- **Smart Chunking** — Recursive, Character, Token-based strategies
- **Free Embeddings** — `sentence-transformers/all-MiniLM-L6-v2` (runs locally)
- **Hybrid Search** — FAISS dense + BM25 sparse with Reciprocal Rank Fusion
- **Free LLMs** — Groq (llama-3.1, mixtral, gemma) + HuggingFace
- **RAG Evaluation** — RAGAS framework (Faithfulness, Answer Relevancy, Context Precision)
- **Web Scraping** — BeautifulSoup + Selenium fallback
- **Multilingual** — Auto-detect + translate (13 languages)
- **Export Chat** — TXT, CSV, JSON, PDF
- **Conversation Memory** — Sliding window history with standalone query rewriting

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/ronitgulia/RAG-Chatbot
cd RAG-Chatbot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get a free Groq API key
Sign up at [console.groq.com](https://console.groq.com) — it's completely free.

### 4. Run the app
```bash
streamlit run app.py
```

### 5. Usage
1. Paste your Groq API key in the sidebar → click **Connect LLM**
2. Go to **Documents** tab → upload files or scrape URLs
3. Go to **Chat** tab → ask questions
4. View **Evaluation** tab for quality metrics
5. **Export** your conversation in any format

## Project Structure

```
RagChatBot/
├── app.py                  # Streamlit UI (Chat, Documents, Evaluation, Export)
├── rag_pipeline.py         # Core orchestrator
├── document_loader.py      # PDF, DOCX, TXT loaders
├── text_chunker.py         # LangChain text splitters
├── embeddings.py           # Sentence-transformers wrapper
├── vector_store.py         # FAISS + BM25 hybrid store
├── llm_provider.py         # Groq + HuggingFace providers
├── evaluation.py           # RAGAS evaluation pipeline
├── web_scraper.py          # Web scraping module
├── multilingual.py         # Language detection & translation
├── export_utils.py         # Chat export utilities
├── styles.py               # Custom Streamlit CSS
├── config.py               # Centralized configuration
├── requirements.txt        # Dependencies
└── .env.example            # Environment variable template
```

## Evaluation Metrics

| Metric | Description |
|---|---|
| **Faithfulness** | Is the answer grounded in retrieved context? |
| **Answer Relevancy** | Does the answer address the question? |
| **Context Precision** | Are retrieved chunks relevant to the query? |

## Available LLM Models (Groq — Free)

| Model | Context | Best For |
|---|---|---|
| `llama-3.1-8b-instant` | 128k | Fast responses (default) |
| `llama-3.3-70b-versatile` | 128k | Best quality |
| `deepseek-r1-distill-llama-70b` | 128k | Reasoning tasks |
| `gemma2-9b-it` | 8k | Balanced |
| `mixtral-8x7b-32768` | 32k | Long documents |

## License

MIT License