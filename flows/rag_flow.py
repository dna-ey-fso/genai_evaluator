import asyncio
from dotenv import load_dotenv
from pathlib import Path
from clients.llm_clients import LLMClient
from clients.embedding_client import EmbeddingClient
from clients.data_clients import TemplateStore

#TODO : impleme,t a FAISS client with ingestion -> link it to send_pormpt and have first rag
