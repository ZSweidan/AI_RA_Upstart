# AI Research Assistant

This project is an AI-powered research assistant that summarizes AI research papers using LangChain, OpenAI, Groq, Gemini, and Pinecone.

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/ai-research-assistant.git
   cd ai-research-assistant

2. Install dependencies:

pip install -r requirements.txt

# Environment Variables  

Set up the `.env` file with the following keys:  

OPENAI_API_KEY=your_key  
GROQ_API_KEY=your_key  
PINECONE_API_KEY=your_key  


Run the application:

    streamlit run app.py

Usage

    Upload research PDFs or query Arxiv.
    Ask questions about any research topic and click on research to do an AI inference.
    Get structured JSON responses with sources and reasoning.

