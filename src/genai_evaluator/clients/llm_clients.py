import inspect
import json
from collections.abc import Callable
from typing import Any, Type
import os
import instructor
import boto3
from azure.ai.inference import ChatCompletionsClient
from boto3.session import Session
from botocore.config import Config
from botocore.client import BaseClient
from pydantic import BaseModel

from genai_evaluator.clients.utils import pydantic2jsontool
from genai_evaluator.interfaces.interfaces import (
    LLMClient,
    Prompt,
    RoleType,
    ToolCall,
)


class AWSBedrockLLMClient(LLMClient):
    def __init__(self, model: str):
        """Initializes an AWS Bedrock LLM-client for Claude or Mistral models

        Args:
            session_client: Boto3 session client (optional)
            model: model ID to use (e.g. anthropic.claude-3-5-sonnet or mistral.mistral-7b-instruct)
        """
        self.model = model
        self.bedrock_client: BaseClient = self._init_client()

    def _init_client(self) -> BaseClient:
        return boto3.client(
            "bedrock-runtime",
            region_name=os.getenv("AWS_REGION", "us-west-2"),
            config=Config(
                retries={
                    "max_attempts": 10,
                    "mode": "standard"
                }
            )
        )

    def _invoke_claude_model(self, messages):
        native_request = {
            "anthropic_version": "bedrock-2023-05-31",
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "system": self._get_system_prompt(messages),
            "messages": self._format_claude_messages(messages)
        }
        body = json.dumps(native_request)
        response = self.bedrock_client.invoke_model(modelId=self.model, body=body)
        response_body = json.loads(response["body"].read())
        return response_body["content"][0]["text"]

    def _invoke_mistral_model(self, messages, response_format=None):
        
        try:
            prompt = self._format_mistral_prompt(messages)
            native_request = {
                "prompt": prompt,
                "temperature": self.temperature
            }
            body = json.dumps(native_request)
            response = self.bedrock_client.messages.create(
                modelId=self.model,
                messages=body,
                responseFormat=pydantic2jsontool(response_format) if response_format else None
            )
        
            return response_body["outputs"][0]["text"]
        except Exception as e:
            raise RuntimeError(f"Failed to invoke Mistral model: {e}")

    def _get_system_prompt(self, messages):
        for msg in messages:
            if msg.role == RoleType.SYSTEM:
                return msg.content
        return ""

    def _format_claude_messages(self, messages):
        formatted_messages = []
        for msg in messages:
            if msg.role != RoleType.SYSTEM:
                role = "user" if msg.role == RoleType.USER else "assistant"
                formatted_messages.append({
                    "role": role,
                    "content": [{"type": "text", "text": msg.content}]
                })
        return formatted_messages

    def _format_mistral_prompt(self, messages):
        prompt_parts = []
        for msg in messages:
            if msg.role == RoleType.SYSTEM:
                prompt_parts.append(f"<system>\n{msg.content}\n</system>\n")
            elif msg.role == RoleType.USER:
                prompt_parts.append(f"<user>\n{msg.content}\n</user>\n")
            elif msg.role == RoleType.ASSISTANT:
                prompt_parts.append(f"<assistant>\n{msg.content}\n</assistant>\n")
        return "".join(prompt_parts)

    def send_prompt(
        self,
        *,
        prompts: list[Prompt],
        temperature: float = None,
        top_p: float = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict = "auto",
        response_format: Type[BaseModel] | dict | None = None,
        **kwargs,
    ) -> Prompt:
        if temperature is not None:
            self.temperature = temperature  # Override default temperature if specified
        
        if tools is not None:
            raise NotImplementedError("Tool calling is not supported for Bedrock models")
        
        if "mistral" in self.model.lower():
            content = self._invoke_mistral_model(prompts)
        elif "claude" in self.model.lower():
            content = self._invoke_claude_model(prompts)
        else:
            raise ValueError(f"Unsupported model: {self.model}")
        
        return Prompt(role=RoleType.ASSISTANT, content=content)
