import warnings
from contextlib import contextmanager
from typing import Type

from docstring_parser import parse
from pydantic import BaseModel


def pydantic2jsontool(cls: Type[BaseModel]) -> dict:
    """Return the pydantic object transform into a valid function call json format (shamelessly stolen)

    Note:
        Its important to add a docstring to describe how to best use this class, it will be included in the

    Returns:
        A dictionary in the format of a tool (used for function calling)

    """
    schema = cls.model_json_schema()
    docstring = parse(cls.__doc__ or "")
    parameters = {}
    for param in docstring.params:
        if param.arg_name in schema["properties"].items():
            name = param.arg_name
            if "description" not in parameters["properties"][name]:
                parameters["properties"][name]["description"] = param.description
    parameters["required"] = sorted(
        k for k, v in parameters["properties"].items() if "default" not in v
    )
    if "description" not in schema:
        schema["description"] = docstring.short_description
    else:
        schema["description"] = (
            # Correctly extracted
            f"{cls.__name__} with all " + f"the required parameters with correct types"
        )

    tool = {
        "name": schema["title"],
        "description": schema["description"],
        "parameters": schema,
    }

    return {
        "type": "function",
        "function": tool,
    }


@contextmanager
def optional_dependencies():
    try:
        yield None
    except ImportError as e:
        msg = f"Missing optional dependency: {e.name}"
        warnings.warn(f"WARNING: {msg}")
        raise e
