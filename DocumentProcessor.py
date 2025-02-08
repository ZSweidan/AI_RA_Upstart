import os
import tempfile
import streamlit as st
import pdfplumber
import arxiv
import google.generativeai as genai
import numpy as np
from pinecone import Pinecone
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import Document
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.tools import DuckDuckGoSearchRun
from GeminiEmbeddings import GeminiEmbeddings
from dotenv import load_dotenv
from typing import List, Dict
import requests
from io import BytesIO
from config import pc, search_tool, genai, INDEX_NAMES, GROQ_MODELS
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentProcessor:
    @staticmethod
    def process_pdf(pdf_path: str, document_title: str = None, document_url: str = None) -> List[Document]:
        """Process a PDF file and return a list of Document objects with metadata."""
        if not pdf_path:
            logger.warning("No PDF path provided.")
            return []
        try:
            logger.info(f"Processing PDF: {pdf_path}")
            with pdfplumber.open(pdf_path) as pdf:
                docs = []
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    if text.strip():
                        docs.append(Document(
                            page_content=text.strip(),
                            metadata={
                                "title": document_title or os.path.basename(pdf_path),
                                "url": document_url or pdf_path,
                                "page": i + 1,
                                "source": pdf_path,
                                "type": "pdf"
                            }
                        ))
                logger.info(f"Extracted {len(docs)} pages from PDF.")
                
                # Split documents into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
                split_docs = text_splitter.split_documents(docs)
                logger.info(f"Split into {len(split_docs)} document chunks.")
                return split_docs
        except Exception as e:
            logger.error(f"PDF processing error: {str(e)}", exc_info=True)
            return []
