from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum
from typing import Any, Type, TypedDict

from pydantic import BaseModel
from typing_extensions import NotRequired


class RoleType(str, Enum):
    """THE POSSIBLE ROLES FOR A Chatbot INPUT"""

    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"
    TOOL = "tool"


class LanguageType(str, Enum):
    """The 3 LANGUAGES SUPPORTED"""

    FR = "french"
    NL = "dutch"
    EN = "english"


class Prompt(TypedDict):  # TypedDict over Data/class to avoid conversion overhead
    """PROMPT INTERFACE COMPATIBLE WITH MOST LLM-CLIENTS OUT-OF-THE-BOX"""

    role: RoleType
    content: str | None
    # ONLY USED WHEN role==RoleType.TOOL
    tool_call_id: NotRequired[str]
    tool_calls: NotRequired[list]
    # TODO: typing of tool_call ?


class ToolCall(TypedDict):
    arguments: dict
    name: str
    id: str


class LLMClient(ABC):
    """INTERFACE FOR ALL LLM-CLIENTS"""

    @abstractmethod
    def send_prompt(
        self,
        prompts: list[Prompt],
        temperature: float,
        top_p: float,
        tools: list[dict] | None = None,
        tool_choice: str | dict = "auto",
        response_format: Type[BaseModel] | dict | None = None,
        **kwargs,
    ) -> Prompt | list[Prompt]:
        """SENDS THE PROMPT:

        Args:
            PROMPTS: A LIST OF AVAILABLE PROMPTS TO KEEP OLDER LLM CLIENTS JUMPING
            tools: A LIST OF AVAILABLE TOOLS THAT MIGHT ENFORCE OR DISALLOW TOOL-CALLS
            RESPONSE_FORMAT: IMPLEMENTATION DEPENDENT BUT ENFORCES SOME KIND OF JSON OUTPUT / SCHEMA
            **KWARGS: OTHER ARGUMENTS THAT WILL BE PASSED DIRECTLY TO THE API

        Returns:
            THE CLIENT ENDPOINT'S ANSWER AS AN ASSISTANT-PROMPT (W/ OR W/O TOOL-CALLS).
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def parse_tool_calls(cls, tool_calls: list) -> list[ToolCall]:
        """TRANSFORMS TOOL-CALLS RECEIVED FROM THE LLM INTO A LIST OF ToolCall"""

        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def execute_tools(
        cls, prompt: Prompt, executable_fxs: dict[str, tuple[Callable, str]]
    ) -> list[Prompt]:
        """EXECUTE THE TOOL-CALLS PRESENT IN THE PROMPT

        Args:
            PROMPT: PROMPT CONTAINING A TOOL-CALLS FIELD
            EXECUTABLE_FXS: DICT CONTAINING AS KEYS THE FUNCTION NAMES AND AS VALUES A TUPLE
                WITH THE FUNCTION AND THE NAME/DESCRIPTION OF THE OUTPUT ARGUMENT, THE KEY AND
                THE DESCRIPTION NEED TO MATCH THE TOOLS GIVEN TO THE LLM-ENDPOINT

        Returns:
            A LIST OF PROMPT OBJECTS, EACH CONTAINING THE RESULT OF A TOOL-CALL
        """
        raise NotImplementedError()


class EmbeddingClient(ABC):
    @abstractmethod
    def embed(self, text: str) -> list[float]:
        raise NotImplementedError()

    @abstractmethod
    async def aembed(self, text: str) -> list[float]:
        raise NotADirectoryError()


class Retrieval(TypedDict):
    content: str
    search_score: float
    rerank_score: float | None
