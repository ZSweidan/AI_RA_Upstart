AI-Powered Research Assistant

Project Overview

The AI-Powered Research Assistant is a multi-LLM system designed to streamline academic research by retrieving, processing, and summarizing AI-related research papers. It integrates GPT-4, Claude 3, LLaMA3, Mistral, and Groq models to compare responses based on accuracy, coherence, token efficiency, cost, response speed, and relevance. The system leverages retrieval-augmented generation (RAG) with vector search to provide accurate and context-aware answers to research queries.

Key Features

Multi-Model Support: Implements OpenAI, Groq, and Gemini models for comprehensive AI-powered research.

Document Processing: Supports PDF uploads, arXiv API retrieval, and AI-driven summarization.

Vector Search Integration: Uses Pinecone vector embeddings for document retrieval and enhanced contextual responses.

Efficient Query Handling: Dynamically selects the best LLM based on response quality and performance metrics.

Advanced Prompt Engineering: Uses structured prompts tailored for research-based question answering.

System Architecture

1. Core Modules

AIChains (Core Processing Module): Orchestrates LLM calls, processes documents, and manages vector search.

DocumentProcessor: Extracts and preprocesses text from PDF files.

ResearchEngine: Fetches and processes research papers from arXiv API or custom PDF uploads.

VectorStoreManager: Manages vector embeddings and facilitates retrieval-augmented generation.

GeminiEmbeddings: Implements Google Gemini for alternative vector search optimization.

2. Workflow

User Input: The user submits a research query or uploads a PDF file.

Document Processing: Extracts text and metadata from the document.

Embedding Generation: Creates vector representations using OpenAI or Gemini embeddings.

Vector Search: Queries Pinecone to retrieve the most relevant documents.

LLM Processing: AI models analyze the query and retrieved documents to generate an informed response.

Final Response: A structured research summary is provided, including citations and AI model comparisons.

AIChains: Research Query Handling

AIChains acts as the intelligence layer, allowing dynamic interactions with different LLMs. It includes:

1. OpenAI Chain

Uses ChatOpenAI (GPT-4, GPT-3.5) for research-based query responses.

Retrieves supporting documents via Pinecone vector search.

Optimized for coherent, structured, and contextually aware answers.

2. Groq Chain

Supports Claude 3, LLaMA3, and Mistral models via Groq API.

Integrates Gemini embeddings for vector search.

Benchmarks model responses for comparison.

3. Research Chain

Fetches papers from arXiv API and processes user-uploaded PDFs.

Uses RAG to enhance LLM responses with retrieved document insights.

Outputs summarized findings with citations.

Technical Details

1. Libraries & Technologies

LLMs: OpenAI (GPT-4), Groq (Claude, LLaMA, Mistral), Google Gemini.

Vector Database: Pinecone.

Document Processing: pdfplumber, arXiv API.

AI Frameworks: LangChain, Streamlit.

Deployment: Docker, FastAPI (for backend), Streamlit (for UI).

2. Performance Metrics

Response Speed: Measures LLM inference latency.

Token Efficiency: Evaluates token usage vs. relevance.

Accuracy & Coherence: Benchmarked across different models.

Cost Efficiency: Tracks API costs for model inference.

Future Enhancements

Hybrid Model Fusion: Combine multiple LLM outputs for improved accuracy.

User Feedback Integration: Allow users to rate responses to fine-tune the system.

Automated Citation Extraction: Provide structured citations for research findings.

Enhanced UI: Develop an interactive dashboard with response comparison metrics.

Conclusion

This AI-powered research assistant leverages cutting-edge LLMs, vector search, and document processing to assist researchers in efficiently gathering and summarizing academic literature. The project is designed to be scalable and adaptable to future AI advancements.

Repository Information

Primary Script: AIChains.py

Vector Management: VectorStoreManager.py

Document Processing: DocumentProcessor.py

Research Retrieval: ResearchEngine.py

Prompt Templates: RA_PROMPT.py

Developed by Zahraa Sweidan 
