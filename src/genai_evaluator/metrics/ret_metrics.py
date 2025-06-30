from enum import Enum

from pydantic import BaseModel

from genai_evaluator.clients.data_clients import TemplateStore
from genai_evaluator.interfaces.interfaces import LLMClient, Prompt, RoleType


class VerdictEnum(str, Enum):
    IDK = "idk"
    NO = "no"
    YES = "yes"


class ContextualPrecisionVerdict(BaseModel):
    verdict: str
    reason: str


class ContextualVerdicts(BaseModel):
    verdicts: list[ContextualPrecisionVerdict]


class StatementExtract(BaseModel):
    statements: list[str]


def contextual_precision(
    *,
    question: str,
    answer_gt: str,
    retrieved_context: list[str],
    llm_client: LLMClient,
    template_store: TemplateStore,
    temperature: float,
    top_p: float,
) -> float | tuple[float, list[str]] | None:
    """
    Computes the contextual precision of the predicted answer against the question.

    Args:
        question (str): The question to evaluate.
        answer_pred (str): The predicted answer to evaluate.
        client (LLMClient): The language model client.
        template_store (TemplateStore): The template store for evaluation.
        temperature (float): The temperature parameter for the model.
        top_p (float): The top_p parameter for the model.

    Returns:
        Verdicts: A list of verdicts indicating the contextual precision.
    """
    resp: Prompt = llm_client.send_prompt(
        prompts=[
            Prompt(
                role=RoleType.SYSTEM,
                content=template_store["contextual_precision_system"].render(
                    document_count_str=len(retrieved_context)
                ),
            ),
            Prompt(
                role=RoleType.USER,
                content=template_store["contextual_precision_user"].render(
                    input=question,
                    expected_output=answer_gt,
                    retrieved_context=retrieved_context,
                ),
            ),
        ],
        temperature=temperature,
        top_p=top_p,
        response_format=ContextualVerdicts,
    )
    verdicts = ContextualVerdicts.model_validate_json(resp.get("content"))

    number_of_verdicts = len(verdicts.verdicts)

    if number_of_verdicts == 0:
        return 0

    node_verdicts = [
        1 if v.verdict.strip().lower() == VerdictEnum.YES else 0
        for v in verdicts.verdicts
    ]

    sum_weighted_precision_at_k = 0.0
    relevant_nodes_count = 0

    for k, is_relevant in enumerate(node_verdicts, start=1):
        if is_relevant:
            relevant_nodes_count += 1
            precision_at_k = relevant_nodes_count / k
            sum_weighted_precision_at_k += precision_at_k

    if relevant_nodes_count == 0:
        return 0.0

    score = sum_weighted_precision_at_k / relevant_nodes_count
    return score


def contextual_recall(
    *,
    answer_gt: str,
    retrieved_context: list[str],
    llm_client: LLMClient,
    template_store: TemplateStore,
    temperature: float = 0.05,
    top_p: float = 1.0,
) -> float | tuple[float, dict[str, StatementExtract]]:
    """
    Computes the contextual recall of the model - measures what portion of the ground truth
    is supported by the retrieved context.

    Args:
        answer_gt (str): The ground truth answer to evaluate.
        retrieved_context (list[str]): The retrieved context for evaluation.
        llm_client (LLMClient): The language model client.
        template_store (TemplateStore): The template store for evaluation.
        temperature (float): The temperature parameter for the model.
        top_p (float): The top_p parameter for the model.

    Returns:
        float | None: The contextual recall score or None if not computable.
    """
    resp: Prompt = llm_client.send_prompt(
        prompts=[
            Prompt(
                role=RoleType.SYSTEM,
                content=template_store["contextual_recall_system"].render(),
            ),
            Prompt(
                role=RoleType.USER,
                content=template_store["contextual_recall_user"].render(
                    expected_output=answer_gt,
                    retrieved_context=retrieved_context,
                ),
            ),
        ],
        temperature=temperature,
        top_p=top_p,
        response_format=ContextualVerdicts,
    )

    verdicts = ContextualVerdicts.model_validate_json(resp.get("content"))

    number_of_verdicts = len(verdicts.verdicts)

    if number_of_verdicts == 0:
        return 0

    justified_sentences = 0
    for verdict in verdicts.verdicts:
        if verdict.verdict.strip().lower() == VerdictEnum.YES:
            justified_sentences += 1

    score = justified_sentences / number_of_verdicts

    return score
