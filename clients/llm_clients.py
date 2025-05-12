import inspect
import json
from collections.abc import Callable
from typing import Any, Type

from azure.ai.inference import ChatCompletionsClient
from openai import AzureOpenAI
from pydantic import BaseModel

from clients.utils import pydantic2jsontool
from interfaces.interfaces import (
    LLMClient,
    Prompt,
    RoleType,
    ToolCall,
)


class OpenAILLMClient(LLMClient):
    def __init__(self, key: str, endpoint: str, api_version: str, model: str):
        """Initializes an AzureOpenAI LLM-client

        Args:
            key: key to access the endpoint
            endpoint: endpoint of client
            api_version: api-version to use
            model: model-name of the deployment to select
        """
        self.model = model
        self.client = AzureOpenAI(
            api_key=key, api_version=api_version, azure_endpoint=endpoint
        )

    def _chat_completion_function(self, **kwargs):
        use_beta = False
        response_format = kwargs.get("response_format", None)
        # TODO: tools can now also be pydantic dataclasses -> need to be supported
        use_beta = inspect.isclass(response_format) and issubclass(
            response_format, BaseModel
        )
        if not use_beta:
            return self.client.chat.completions.create(model=self.model, **kwargs)
        else:
            # Structured output support is still in beta
            return self.client.beta.chat.completions.parse(model=self.model, **kwargs)

    def send_prompt(
        self,
        *,
        prompts: list[Prompt],
        temperature: float,
        top_p: float,
        tools: list[dict] | None = None,
        tool_choice: str | dict = "auto",
        response_format: Type[BaseModel] | dict | None = None,
        **kwargs,
    ) -> Prompt:
        use_tools = tools is not None and len(tools) > 0
        use_response_format = response_format is not None
        assert not (use_tools and use_response_format), (
            "Can't set tools and response_format at the same time"
        )
        response = self._chat_completion_function(
            messages=prompts,
            temperature=temperature,
            top_p=top_p,
            **(
                dict(tools=tools, tool_choice=tool_choice) if use_tools else {}
            ),  # only add tools to request when defined in the input
            **(
                dict(response_format=response_format) if use_response_format else {}
            ),  # only add response_format to request when defined in the input
            **kwargs,
        )

        # FIXME: throw error if finish_reason is not stop or tool_calls?
        if use_tools and response.choices[0].message.tool_calls is not None:
            return Prompt(
                role=RoleType.ASSISTANT,
                content=None,
                tool_calls=response.choices[0].message.tool_calls,
            )

        return Prompt(
            role=RoleType.ASSISTANT, content=response.choices[0].message.content
        )

    @classmethod
    def parse_tool_calls(cls, tool_calls: list) -> list[ToolCall]:
        """Transforms tool-calls into a ToolCall and deserializes the json"""
        return [
            ToolCall(
                arguments=json.loads(tc.function.arguments),
                name=tc.function.name,
                id=tc.id,
            )
            for tc in tool_calls
        ]

    @classmethod
    def execute_tools(
        cls, prompt: Prompt, executable_fxs: dict[str, tuple[Callable, str]]
    ) -> list[Prompt]:
        """Execute the tool-calls present in the prompt

        Args:
            prompt: prompt containing a tool_calls field
            executable_fxs: dict containing as keys the function names, and as values a tuple
                with the function and the name/description of the output argument. The key and
                the description need to match the tools given to the llm-endpoint previously.
                All 'arguments' present in a tool-call will be passed as is to the function.
                The function's return type should be serializable.

        Returns:
            A list of Prompt objects, each containing the result of a tool-call. The content of each
            prompt is a serialized json.
        """
        ret_prompts = []
        for tool_call in cls.parse_tool_calls(prompt["tool_calls"]):
            fx, return_name = executable_fxs[tool_call["name"]]
            result = fx(**tool_call["arguments"])
            ret_prompt = Prompt(
                role=RoleType.TOOL,
                # TODO: support other return-types
                content=json.dumps({return_name: result, **tool_call["arguments"]}),
                tool_call_id=tool_call["id"],
            )
            ret_prompts.append(ret_prompt)
        return ret_prompts


class MistralLLMClient(OpenAILLMClient):
    def __init__(self, credential: Any, endpoint: str):
        """Initializes an Mistral Azure-serverless LLM-client

        Nearly identical to OpenAI's API, subtleties regarding function calling have been accounted for

        Args:
            credential: Azure credential to access the endpoint
            endpoint: endpoint of client
        """
        self.client = ChatCompletionsClient(
            endpoint=endpoint,
            credential=credential,
        )
        self._chat_completion_function = lambda **kwargs: self.client.complete(**kwargs)

    def send_prompt(self, **kwargs):
        # Map openai Pydantic structured output to structured function calling output
        parse_structured_output = False
        if issubclass(
            (structured_output := kwargs.get("response_format", tuple)), BaseModel
        ):
            if kwargs.get("tools", None) is not None:
                raise NotImplementedError(
                    "Mistral does not yet support structured outputs and conversion is only possible when no tools are set"
                )
            else:
                import warnings

                warnings.warn(
                    "Structured outputs are not supported by Mistral but will be converted to function calling"
                )
                kwargs["tools"] = [pydantic2jsontool(structured_output)]
                kwargs["tool_choice"] = "required"
                del kwargs["response_format"]
                parse_structured_output = True

        # Map openai 'tool-choice' terminology to mistral
        if "tool_choice" in kwargs:
            kwargs["tool_choice"] = {
                "auto": "auto",
                "required": "any",
                "none": "none",
            }[kwargs["tool_choice"]]

        prompt = super().send_prompt(**kwargs)

        if parse_structured_output:
            # TODO: we might want to return the pydantic object directly + also support in openai ?
            prompt = Prompt(
                role=RoleType.ASSISTANT,
                content=prompt["tool_calls"][0].function.arguments,
            )
        return prompt
