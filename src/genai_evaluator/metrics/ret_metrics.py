from enum import Enum

from pydantic import BaseModel

from genai_evaluator.clients.data_clients import TemplateStore
from genai_evaluator.interfaces.interfaces import LLMClient, Prompt, RoleType


class VerdictEnum(str, Enum):
    IDK = "idk"
    NO = "no"
    YES = "yes"


class Verdict(BaseModel):
    verdict: VerdictEnum
    reason: str | None = None


class VerdictList(BaseModel):
    verdicts: list[Verdict]  # sadly max/min_length is not supported yet


class StatementExtract(BaseModel):
    statements: list[str]


class ClaimsExtract(BaseModel):
    claims: list[str]


def text_ends_with_yes(text: str) -> bool:
    """Checks if a text ends with 'yes' in a substring."""
    if len(text) > 5:
        return "yes" in text[-5:].lower()
    return text.strip().lower() == "yes"


def compute_relevancy(
    *,
    question: str,
    answer_pred: str,
    client: LLMClient,
    template_store: TemplateStore,
    temperature: float,
    top_p: float,
    return_statements: bool = False,
) -> float | tuple[float, StatementExtract] | None:
    resp: Prompt = client.send_prompt(
        prompts=[
            Prompt(
                role=RoleType.SYSTEM,
                content=template_store["relevancy_extract_system"].render(),
            ),
            Prompt(
                role=RoleType.USER,
                content=answer_pred,
            ),
        ],
        temperature=temperature,
        top_p=top_p,
        response_format=StatementExtract,
    )

    statements = StatementExtract.model_validate_json(resp["content"])
    resp: Prompt = client.send_prompt(
        prompts=[
            Prompt(
                role=RoleType.SYSTEM,
                content=template_store["relevancy_extract_system"].render(),
            ),
            Prompt(
                role=RoleType.USER,
                content=template_store["relevancy_user"].render(
                    question=question, statements=statements.statements
                ),
            ),
        ],
        temperature=temperature,
        top_p=top_p,
        response_format=VerdictList,
    )

    relevancy_per_statement = VerdictList.model_validate_json(resp["content"])
    n_yes, n_no = 0, 0
    n_tot = len(relevancy_per_statement.verdicts)
    if len(relevancy_per_statement.verdicts) == len(statements.statements):
        for verdict in relevancy_per_statement.verdicts:
            match verdict.verdict:
                case VerdictEnum.YES:
                    n_yes += 1
                case VerdictEnum.NO:
                    n_no += 1
        relevancy = 0 if n_tot == 0 else n_yes / n_tot
    else:
        relevancy = 0 if n_tot == 0 else n_yes / n_tot

    if return_statements:
        return relevancy, statements

    return relevancy
