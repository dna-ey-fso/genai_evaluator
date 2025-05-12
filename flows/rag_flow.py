import asyncio
from pathlib import Path

from dotenv import load_dotenv

from clients.data_clients import TemplateStore
from clients.embedding_client import EmbeddingClient
from clients.llm_clients import LLMClient

# TODO : impleme,t a FAISS client with ingestion -> link it to send_pormpt and have first rag
