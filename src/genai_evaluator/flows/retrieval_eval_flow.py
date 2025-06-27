from genai_evaluator.clients.data_clients import TemplateStore
from genai_evaluator.interfaces.interfaces import LLMClient
from genai_evaluator.metrics.deterministic import compute_exact_match
from genai_evaluator.metrics.metric_type import MetricType
from genai_evaluator.metrics.ret_metrics import contextual_precision, contextual_recall


def retrieval_eval_flow(
    *,
    question: str,
    answer_pred: str,
    answer_gt: str,
    llm_client: LLMClient,
    retrieved_context: list[str],
    template_store: TemplateStore,
    context_gt: list[str] | None = None,
    temperature: float = 0.05,
    top_p: float = 1.0,
    return_statements: bool = False,
) -> dict[str, float | dict[str, float] | None]:
    contextual_information_precision = contextual_precision(
        question=question,
        answer_gt=answer_pred,
        retrieved_context=retrieved_context,
        llm_client=llm_client,
        template_store=template_store,
        temperature=temperature,
        top_p=top_p,
    )
    contextual_fidelity = contextual_recall(
        question=question,
        answer_pred=answer_pred,
        answer_gt=answer_gt,
        retrieved_context=retrieved_context,
        llm_client=llm_client,
        template_store=template_store,
        temperature=temperature,
        top_p=top_p,
    )
    exact_match = 0
    if context_gt is not None:
        min_length = min(len(retrieved_context), len(context_gt))
        if min_length > 0:
            exact_match = (
                sum(
                    compute_exact_match(
                        answer_pred=retrieved_context[i], answer_ref=context_gt[i]
                    )
                    for i in range(min_length)
                )
                / min_length
            )

    return dict(
        contextual_recall=contextual_fidelity,
        contextual_precision=contextual_information_precision,
        exact_match=exact_match,
    )
