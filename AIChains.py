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
from ResearchEngine import ResearchEngine
from DocumentProcessor import DocumentProcessor
from VectorStoreManager import VectorStoreManager
from dotenv import load_dotenv
from typing import List, Dict
import requests
from io import BytesIO
from config import pc, search_tool, genai, INDEX_NAMES, GROQ_MODELS
from RA_PROMPT import RESEARCH_PROMPT

import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AIChains:
    @staticmethod
    def openai_chain(question: str, context: str = "", pdf_path: str = None) -> str:
        start_time = time.time()
        try:
            llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
            embeddings = OpenAIEmbeddings()
            
            if pdf_path:
                docs = DocumentProcessor.process_pdf(pdf_path)
                vectorstore = VectorStoreManager.get_vectorstore(docs, embeddings, INDEX_NAMES["openai"])
                if not vectorstore:
                    return "Error: Could not process document"

                retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                
                chain = create_retrieval_chain(
                    retriever, 
                    create_stuff_documents_chain(llm, RESEARCH_PROMPT)
                )
                result = chain.invoke({
                    "input": question,
                    "additional_context": context
                })
                execution_time = time.time() - start_time
                logging.info(f"OpenAI Execution Time: {execution_time:.2f} seconds")
  
                return result["answer"]
            
            response = llm.invoke(f"{context}\nQuestion: {question}")
            execution_time = time.time() - start_time
            logging.info(f"OpenAI Execution Time: {execution_time:.2f} seconds")
            return response.content
        except Exception as e:
            logging.error(f"OpenAI Error: {str(e)}")
            return f"OpenAI Error: {str(e)}"
    
    @staticmethod
    def groq_chain(question: str, model_name: str, context: str = "", pdf_path: str = None) -> str:
        start_time = time.time()
        try:
            llm = ChatGroq(model_name=model_name)
            embeddings = GeminiEmbeddings()
            
            if pdf_path:
                docs = DocumentProcessor.process_pdf(pdf_path)
                vectorstore = VectorStoreManager.get_vectorstore(docs, embeddings, INDEX_NAMES["groq"])
                if not vectorstore:
                    return "Error: Could not process document"
                
                retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                
                chain = create_retrieval_chain(
                    retriever,
                    create_stuff_documents_chain(llm, RESEARCH_PROMPT)
                )
                result = chain.invoke({
                    "input": question,
                    "additional_context": context
                })
                execution_time = time.time() - start_time
                logging.info(f"Groq Execution Time: {execution_time:.2f} seconds")
                return result["answer"]
            
            response = llm.invoke(f"{context}\nQuestion: {question}")
            execution_time = time.time() - start_time
            logging.info(f"Groq Execution Time: {execution_time:.2f} seconds")
            return response.content
        except Exception as e:
            logging.error(f"Groq Error: {str(e)}")
            return f"Groq Error: {str(e)}"
    
    @staticmethod
    def research_chain(question: str, model_name: str, mode: str = "arxiv", pdf_links: List[str] = None, titles: List[str] = None) -> str:
        start_time = time.time()
        try:
            if mode == "arxiv":
                docs = ResearchEngine.fetch_and_process_arxiv_papers(question)
            elif mode == "custom_pdfs" and pdf_links:
                docs = ResearchEngine.process_pdf_links(pdf_links, titles)
            else:
                return "Invalid research mode or missing PDF links"

            if not docs:
                return "No relevant documents found."
            
            embeddings = GeminiEmbeddings()
            vectorstore = VectorStoreManager.get_vectorstore(
                docs, 
                embeddings, 
                INDEX_NAMES["research"]
            )
            if not vectorstore:
                return "Error: Could not process research papers"
            
            llm = ChatGroq(model_name=model_name)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            
            chain = create_retrieval_chain(
                retriever,
                create_stuff_documents_chain(llm, RESEARCH_PROMPT)
            )
            result = chain.invoke({"input": question})
            execution_time = time.time() - start_time
            logging.info(f"Research Execution Time: {execution_time:.2f} seconds")
            return result["answer"]
        except Exception as e:
            logging.error(f"Research Error: {str(e)}")
            return f"Research Error: {str(e)}"
