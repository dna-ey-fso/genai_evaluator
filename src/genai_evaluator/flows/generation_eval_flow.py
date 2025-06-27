from genai_evaluator.clients.data_clients import TemplateStore
from genai_evaluator.interfaces.interfaces import LLMClient
from genai_evaluator.metrics.deterministic import *
from genai_evaluator.metrics.gen_metrics import (
    compute_faithfulness,
    compute_precision,
    compute_recall,
    compute_relevancy,
)
from genai_evaluator.metrics.metric_type import MetricType


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
    metric_types: list[MetricType] | None = [
        MetricType.FAITHFULNESS,
        MetricType.RELEVANCY,
        MetricType.COSINE_SIMILARITY,
    ],
    return_statements: bool = False,
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
    relevancy = compute_relevancy(
        question=question,
        answer_pred=answer_pred,
        client=llm_client,
        template_store=template_store,
        temperature=temperature,
        top_p=top_p,
        return_statements=return_statements,
    )

    cosine_similarity = compute_cosine_similarity(
        answer_pred=answer_pred,
        answer_ref=answer_gt,
    )

    return dict(
        faithfulness=faithfulness,
        relevancy=relevancy,
        cosine_similarity=cosine_similarity,
    )
