from clients.data_clients import TemplateStore
from interfaces.interfaces import LLMClient
from metrics.gen_metrics import (
    compute_faithfulness,
    compute_precision,
    compute_recall,
)


def generation_eval_flow(
    *,
    question: str,
    answer_pred: str,
    answer_gt: str,
    llm_client: LLMClient,
    context: list[str],
    template_store: TemplateStore,
    temperature: float = 0.05,
    top_p: float = 1.0,
) -> dict[str, float | dict[str, float] | None]:
    # Evaluate faithfulness i.e. ratio of predicted answer claims that find their root in the provided context
    faithfulness = compute_faithfulness(
        answer_pred=answer_pred,
        context=context,
        client=llm_client,
        template_store=template_store,
        temperature=temperature,
        top_p=top_p,
    )

    # Evaluate answer precision, i.e. the ratio of the predicted answer statements present in ground truth answer
    precision, dict_statements_pred = compute_precision(
        answer_pred=answer_pred,  # statements_pred,
        answer_gt=answer_gt,
        client=llm_client,
        template_store=template_store,
        temperature=temperature,
        top_p=top_p,
        return_statements=True,
    )

    # Evaluate answer recall, i.e. the ratio of ground truth statements present in the predicted answer
    recall = compute_recall(
        answer_pred=dict_statements_pred["pred"],
        answer_gt=dict_statements_pred["gt"],
        client=llm_client,
        template_store=template_store,
        temperature=temperature,
        top_p=top_p,
        return_statements=False,
    )

    return dict(faithfulness=faithfulness, precision=precision, recall=recall)
