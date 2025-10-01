
# JAX London 2025 - AI Workshop Code Repository

Welcome to the code repository for the **JAX London 2025 AI Workshop**! This repository contains hands-on examples and implementations covering fundamental AI concepts, from basic API interactions to advanced Retrieval Augmented Generation (RAG) systems.

## 🎯 Workshop Overview

This workshop is designed to take you through a practical journey of AI development, covering:
- **Day 1**: AI Fundamentals - API interactions, embeddings, and model comparisons
- **Day 2**: Advanced RAG Systems - Building intelligent document retrieval systems

## 📁 Repository Structure

```
jax-london-2025/
├── day-1/                          # AI Fundamentals & API Interactions
├── day-2/                          # RAG Systems & Document Processing
├── requirements.txt                # Combined dependencies for both days
└── README.md                      # This file
```

## 🚀 Day 1: AI Fundamentals

**Location**: `day-1/`

### Core Concepts Covered
- **API Integration**: Connect to various AI providers
- **Embeddings**: Understanding vector representations
- **Model Comparison**: Evaluating different AI models
- **Token Analysis**: Understanding tokenization and probabilities

### Key Files

#### Getting Started Scripts
- `getting_started_groq.py` - Quick start with Groq API
- `getting_started_ollama.py` - Local AI with Ollama
- `getting_started_lm_studio.py` - LM Studio integration

#### Provider-Specific Examples
- `basic_chatgpt.py` - OpenAI GPT integration
- `basic_claude.py` - Anthropic Claude integration
- `basic_groq.py` - Groq API examples
- `basic_mistral.py` - Mistral AI integration
- `basic_fireworks.py` - Fireworks AI examples

#### Advanced Concepts
- `embedding_demo.py` & `embedding_example.py` - Vector embeddings
- `word_embeddings.py` - Word-level embedding analysis
- `show_tokens.py` - Tokenization visualization
- `logit_probabilities.py` - Understanding model confidence
- `intelligent_character_recognition.py` - OCR and image processing

#### Fun Applications
- `ai_astrology.py` & `ai_astrology_groq.py` - Creative AI applications
- `3d_plot.html` - Interactive visualizations
- `simple_code.py` - Code generation examples

### Dependencies (Day 1)
```
openai, anthropic, groq, mistralai
torch, transformers, sentence-transformers
tensorflow, tf-keras
numpy, matplotlib, plotly
ollama, llama-cpp-python
gradio, tabulate
```

## 🔧 Day 2: RAG Systems & Document Processing

**Location**: `day-2/`

### Core Concepts Covered
- **RAG Architecture**: Building retrieval-augmented generation systems
- **Vector Databases**: ChromaDB for persistent storage
- **Document Processing**: Text chunking and preprocessing
- **Embedding Strategies**: Optimized retrieval techniques

### Key Files

#### RAG Implementations
- `rag_alice_in_wonderland_chromadb.py` - Production-ready RAG with ChromaDB
- `rag_alice_in_wonderland.py` - Basic RAG implementation
- `rag_alice_in_wonderland_transformers.py` - Transformers-based RAG

#### Sentiment Analysis
- `analyse_sentiment_01.py` - Basic sentiment analysis
- `analyse_sentiment_02.py` - Advanced sentiment analysis
- `analyse_sentiment_kaggle.py` - Kaggle dataset sentiment analysis
- `sentiment_analysis_results.csv` - Analysis output data
- `sentiment_analysis_results.png` - Visualization of results

#### Data Extraction & Processing
- `alice_in_on_go.py` - Text processing utilities
- `data_extraction_ollama.py` - Document extraction with Ollama
- `formatted_response_example.py` - Structured response formatting
- `fill_in_middle.py` - Code infilling examples
- `kaggle_summary_complete.py` - Dataset summarization

#### Web Scraping & Data Collection
- `scrape.py` - Web scraping utilities
- `scrape_gdpr_article.py` - GDPR content extraction
- `gdpr_article_content.txt` - Extracted legal document

#### Function Calling & Tool Use
- `ollama_function_support.py` - Function calling with Ollama
- `ollama_function_results.csv` - Function execution results

#### Database & Structured Data
- `payroll.py` - Database interaction examples
- `payroll2.py` - Advanced payroll processing
- `city_payroll.db` - SQLite database sample

#### Visualization Tools
- `visual_ml_studio.py` - Visual ML Studio integration

#### Sample Data
- `data/alice_in_wonderland.txt` - Text corpus for RAG examples
- `data/IMG_*.jpg` - Sample images for OCR and processing

#### Persistent Storage
- `chroma_db/` - ChromaDB vector database storage

### Dependencies (Day 2)
```
chromadb - Vector database
beautifulsoup4 - Web scraping
```

## 🛠 Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository-url>
cd jax-london-2025
```

### 2. Install Dependencies
```bash
# Install all dependencies for both days
pip install -r requirements.txt

# Or install day-specific dependencies
pip install -r day-1/requirements.txt  # Day 1 only
pip install -r day-2/requirements.txt  # Day 2 only
```

### 3. Set Up API Keys
Create a `.env` file or set environment variables:
```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-claude-key"
export GROQ_API_KEY="your-groq-key"
export MISTRAL_API_KEY="your-mistral-key"
```

### 4. Install Ollama (Optional)
For local AI models:
```bash
# Install Ollama from https://ollama.com
ollama serve
ollama pull embeddinggemma  # For embeddings
ollama pull gemma3n:e4b     # For text generation
```

## 🎓 Workshop Flow

### Day 1: Foundation Building
1. **Start Here**: `getting_started_groq.py` - Your first AI API call
2. **Explore Providers**: Try different `basic_*.py` files
3. **Understand Embeddings**: Run `embedding_demo.py`
4. **Analyze Tokens**: Experiment with `show_tokens.py`
5. **Creative Applications**: Try `ai_astrology.py`

### Day 2: Advanced RAG Systems & Applications
1. **Basic RAG**: Start with `rag_alice_in_wonderland.py`
2. **Production RAG**: Explore `rag_alice_in_wonderland_chromadb.py`
3. **Sentiment Analysis**: Try `analyse_sentiment_01.py` through `analyse_sentiment_kaggle.py`
4. **Function Calling**: Experiment with `ollama_function_support.py`
5. **Data Extraction**: Use `data_extraction_ollama.py` for structured extraction
6. **Web Scraping**: Learn scraping with `scrape_gdpr_article.py`
7. **Database Operations**: Work with structured data in `payroll.py`
8. **Code Infilling**: Try `fill_in_middle.py` for code completion
9. **Persistent Storage**: Work with ChromaDB
10. **Custom Data**: Adapt examples for your own documents

## 🔍 Key Features

### Comprehensive RAG Implementation
- **Smart Text Chunking**: Configurable chunk sizes with intelligent boundary detection
- **Persistent Vector Storage**: ChromaDB integration for efficient retrieval
- **Multiple Embedding Strategies**: Support for various embedding models
- **Multilingual Support**: Works with multiple languages out of the box

### Sentiment Analysis & NLP
- **Multiple Approaches**: Basic to advanced sentiment analysis techniques
- **Dataset Integration**: Kaggle dataset analysis examples
- **Visualization**: Automated chart generation for results
- **CSV Export**: Results exportable for further analysis

### Function Calling & Tool Use
- **Ollama Functions**: Native function calling support
- **Structured Outputs**: JSON-formatted responses
- **Tool Integration**: Connect AI to external systems

### Data Extraction & Web Scraping
- **Structured Extraction**: Pull specific data from documents
- **Web Content**: Scrape and process web pages
- **Legal Documents**: Parse complex document formats
- **Formatted Responses**: Clean, structured output

### Database & Structured Data
- **SQLite Integration**: Work with local databases
- **Payroll Processing**: Real-world data examples
- **Query Generation**: AI-powered SQL queries

### Production-Ready Code
- **Error Handling**: Robust error management and fallbacks
- **Performance Optimization**: Efficient embedding generation and storage
- **Configurable Parameters**: Easy customization for different use cases
- **Detailed Logging**: Comprehensive feedback and debugging information

## 🚨 Prerequisites

- Python 3.8+
- Basic understanding of Python programming
- API keys for chosen AI providers (optional for local models)
- 4GB+ RAM for local model inference

## 📚 Learning Outcomes

After completing this workshop, you'll be able to:

1. **Integrate Multiple AI APIs**: Connect to OpenAI, Claude, Groq, and more
2. **Build RAG Systems**: Create intelligent document retrieval applications
3. **Work with Embeddings**: Understand and implement vector representations
4. **Perform Sentiment Analysis**: Analyze text sentiment with various approaches
5. **Use Function Calling**: Integrate AI with external tools and systems
6. **Extract Structured Data**: Pull specific information from unstructured text
7. **Scrape Web Content**: Gather and process data from websites
8. **Work with Databases**: Integrate AI with SQLite and structured data
9. **Generate Code**: Use AI for code completion and infilling
10. **Optimize Performance**: Configure chunking, overlap, and retrieval strategies
11. **Deploy AI Applications**: Build production-ready AI-powered tools

## 🤝 Support

- **Workshop Materials**: All examples include detailed comments
- **Error Handling**: Comprehensive error messages and debugging tips
- **Extensible Code**: Easy to modify and adapt for your use cases

## 📖 Additional Resources

- **Ollama**: https://ollama.com - Local AI model management
- **ChromaDB**: https://trychroma.com - Vector database documentation  
- **Transformers**: https://huggingface.co/transformers - ML model library
- **OpenAI API**: https://platform.openai.com/docs - OpenAI documentation

## 🎉 Getting Started

1. Start with Day 1 examples to understand AI APIs
2. Progress to Day 2 for advanced RAG implementations
3. Experiment with different models and parameters
4. Adapt the code for your own projects!

Happy learning at JAX London 2025! 🚀
