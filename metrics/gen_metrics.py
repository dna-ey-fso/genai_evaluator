from enum import Enum

from pydantic import BaseModel

from clients.data_clients import TemplateStore
from interfaces.interfaces import LLMClient, Prompt, RoleType


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


def compute_faithfulness(
    *,
    answer_pred: str,
    context: list[str],
    client: LLMClient,
    temperature: float,
    template_store: TemplateStore,
    top_p: float,
    return_claims: bool = False,
) -> float | tuple[float, StatementExtract]:
    resp: Prompt = client.send_prompt(
        prompts=[
            Prompt(
                role=RoleType.SYSTEM,
                content=template_store["faithfulness_extract_system"].render(),
            ),
            Prompt(
                role=RoleType.USER,
                content=answer_pred,
            ),
        ],
        temperature=temperature,
        top_p=top_p,
        response_format=ClaimsExtract,
    )

    claims = ClaimsExtract.model_validate_json(resp["content"])
    resp: Prompt = client.send_prompt(
        prompts=[
            Prompt(
                role=RoleType.SYSTEM,
                content=template_store["faithfulness_system"].render(),
            ),
            Prompt(
                role=RoleType.USER,
                content=template_store["faithfulness_user"].render(
                    context=context, claims=claims.claims
                ),
            ),
        ],
        temperature=temperature,
        top_p=top_p,
        response_format=VerdictList,
    )

    faithfulness_per_claim = VerdictList.model_validate_json(resp["content"])

    n_yes, n_no = 0, 0
    if len(faithfulness_per_claim.verdicts) == len(claims.claims):
        for verdict in faithfulness_per_claim.verdicts:
            match verdict.verdict:
                case VerdictEnum.YES:
                    n_yes += 1
                case VerdictEnum.NO:
                    n_no += 1
        n_tot = len(claims.claims)
        faithfulness = 0 if n_tot == 0 else n_yes / n_tot
    else:
        faithfulness = None

    if return_claims:
        return faithfulness, claims

    return faithfulness


def compute_precision(
    *,
    answer_pred: str | StatementExtract,
    answer_gt: str | StatementExtract,
    client: LLMClient,
    template_store: TemplateStore,
    temperature: float,
    top_p: float,
    return_statements: bool = False,
) -> float | tuple[float, dict[str, StatementExtract]]:
    # Transform text into list of statements
    def extract_statements(text: str) -> list[str]:
        resp: Prompt = client.send_prompt(
            prompts=[
                Prompt(
                    role=RoleType.SYSTEM,
                    content=template_store["relevancy_extract_system"].render(),
                ),
                Prompt(role=RoleType.USER, content=text),
            ],
            temperature=temperature,
            top_p=top_p,
            response_format=StatementExtract,
        )
        return StatementExtract.model_validate_json(resp["content"])

    statements_pred = (
        extract_statements(answer_pred) if isinstance(answer_pred, str) else answer_pred
    )
    statements_gt = (
        extract_statements(answer_gt) if isinstance(answer_gt, str) else answer_gt
    )

    resp: Prompt = client.send_prompt(
        prompts=[
            Prompt(
                role=RoleType.SYSTEM,
                content=template_store["precision_system"].render(),
            ),
            Prompt(
                role=RoleType.USER,
                content=template_store["precision_user"].render(
                    statements_pred=statements_pred.statements,
                    statements_gt=statements_gt.statements,
                ),
            ),
        ],
        temperature=temperature,
        top_p=top_p,
        response_format=VerdictList,
    )

    precision_per_statement = VerdictList.model_validate_json(resp["content"])
    n_yes, n_no = 0, 0
    if len(precision_per_statement.verdicts) == len(statements_pred.statements):
        for verdict in precision_per_statement.verdicts:
            match verdict.verdict:
                case VerdictEnum.YES:
                    n_yes += 1
                case VerdictEnum.NO:
                    n_no += 1
        n_tot = n_yes + n_no
        precision = 0 if n_tot == 0 else n_yes / n_tot
    else:
        precision = 0

    if return_statements:
        return precision, dict(gt=statements_gt, pred=statements_pred)

    return precision


def compute_recall(
    *,
    answer_pred: str | StatementExtract,
    answer_gt: str | StatementExtract,
    client: LLMClient,
    template_store: TemplateStore,
    top_p: float,
    temperature: float,
    return_statements: bool = False,
) -> float | tuple[float, dict[str, StatementExtract]]:
    """Computes the Recall

    This function uses the mathematic definition equal to the Recall of the current problem;
    the precision of the inverse problem is the recall of the current problem

    """
    result = compute_precision(
        answer_pred=answer_gt,
        answer_gt=answer_pred,
        client=client,
        template_store=template_store,
        temperature=temperature,
        top_p=top_p,
        return_statements=return_statements,
    )

    if return_statements:
        return_recall, dict_statements = result
        return return_recall, dict(
            gt=dict_statements["pred"], pred=dict_statements["gt"]
        )

    return result
