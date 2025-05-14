import os

from openai import AzureOpenAI
from typing_extensions import Self

from interfaces.interfaces import EmbeddingClient


class OpenAIEmbeddingClient(EmbeddingClient):
    """
    Initializes an AzureOpenAI embedding client.

    This constructor sets up the client with the necessary credentials and configurations
    to interact with the Azure OpenAI service for generating embeddings.

    Args:
        credential (BeCapCredentialType): The credential required to authenticate with the Azure OpenAI service.
        endpoint (str): The endpoint URL of the credential required to authenticate with the Azure OpenAI service.
        api_version (str): The API version of the Azure OpenAI service.
        model (str): The name of the model to use for making requests to use for generating embeddings.
    """

    def __init__(self, credential, endpoint: str, api_version: str, model: str):
        self._client = AzureOpenAI(
            api_key=credential, api_version=api_version, azure_endpoint=endpoint
        )
        self._model = model

    def embed(self, text: str) -> list[float]:
        response = self._client.embeddings.create(model=self._model, input=text)
        return response.data[0].embedding

    async def async_embed(self, text: str) -> list[float]:
        response = await self._client.embeddings.create(model=self._model, input=text)
        return response.data[0].embedding

    @classmethod
    def from_env(cls, prefix: str = "AZURE_OPENAI_") -> "Self":
        """Uses environment variables to initialize the OpenAIEmbeddingClient"""
        endpoint = os.getenv(f"{prefix}_ENDPOINT")
        credential = os.getenv(f"{prefix}_KEY")
        model = os.getenv(f"{prefix}_EMBEDDING_MODEL")
        api_version = os.getenv(f"{prefix}_API_VERSION")
        return cls(
            credential=credential,
            endpoint=endpoint,
            api_version=api_version,
            model=model,
        )
