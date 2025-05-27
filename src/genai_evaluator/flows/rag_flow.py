from typing import Any, Dict, List

import pandas as pd
from promptflow.tracing import trace

from genai_evaluator.clients.data_clients import TemplateStore
from genai_evaluator.clients.llm_clients import LLMClient
from genai_evaluator.interfaces.interfaces import Prompt, RoleType, VectorStoreClient
from genai_evaluator.metrics.gen_metrics import (
    compute_faithfulness,
    compute_precision,
    compute_recall,
)
from genai_evaluator.metrics.ret_metrics import compute_relevancy


@trace
def rag_flow(
    question: str,
    answer_gt: str,
    system_prompt: str,
    llm_client: LLMClient = None,
    template_store: TemplateStore = None,
    vector_store: VectorStoreClient = None,
    num_results: int = 5,
    top_p: float = 1.0,
    temperature: float = 0.05,
) -> Dict[str, Any]:
    """
    Full RAG flow:
    vector store creation,
    LLM query with context retrieval, and evaluation of the answer.
    Args:
        question (str): The question to be answered.
        answer (str): The ground truth answer for evaluation.
        llm_client (LLMClient): The LLM client to use for generating answers.
        template_store (TemplateStore): The template store for managing templates.
        vector_store (VectorStore): The vector store for document retrieval.
        evaluate (bool): Whether to evaluate the generated answer.
        num_results (int): Number of results to retrieve from the vector store.
        top_p (float): Top-p sampling parameter for LLM generation.
        temperature (float): Temperature parameter for LLM generation.
        system_prompt (str): Custom system prompt for the LLM.
    Returns:
        Dict[str, Any]: Dictionary containing the results of the query and evaluation.
    """
    if system_prompt is None:
        system_prompt = (
            "Based on the context, answer the question as accurately as possible."
        )

    # Retrieve relevant documents
    retrievals = vector_store.search(question, k=num_results)
    context = [r.get("content", None) for r in retrievals if r is not None]

    # Create the system prompt with context
    context_str = "\n".join(context)

    full_system_prompt = f"{system_prompt}\n\nContext:\n{context_str}"

    # Send prompt to LLM
    response = llm_client.send_prompt(
        prompts=[
            Prompt(role=RoleType.SYSTEM, content=full_system_prompt),
            Prompt(role=RoleType.USER, content=question),
        ],
        temperature=0.05,
        top_p=1.0,
    )

    answer_pred = response["content"]

    relevancy = compute_relevancy(
        question=question,
        answer_pred=answer_pred,
        client=llm_client,
        template_store=template_store,
        temperature=temperature,
        top_p=top_p,
        return_statements=False,
    )

    # Evaluate if requested

    faithfulness = compute_faithfulness(
        answer_pred=answer_pred,
        context=context,
        client=llm_client,
        template_store=template_store,
        temperature=temperature,
        top_p=top_p,
    )

    precision = compute_precision(
        answer_pred=answer_pred,
        answer_gt=answer_gt,
        client=llm_client,
        template_store=template_store,
        temperature=temperature,
        top_p=top_p,
        return_statements=False,
    )

    recall = compute_recall(
        answer_pred=answer_pred,
        answer_gt=answer_gt,
        client=llm_client,
        template_store=template_store,
        temperature=temperature,
        top_p=top_p,
        return_statements=False,
    )

    return dict(
        relevancy=relevancy,
        faithfulness=faithfulness,
        precision=precision,
        recall=recall,
    )


@trace
def batch_rag_evaluation(
    questions: List[str], answers_gt: List[str], system_prompt: str, **kwargs
) -> Dict[str, Any]:
    """
    Run RAG evaluation on multiple question-answer pairs and aggregate results.
    """
    results = []

    for q, a in zip(questions, answers_gt):
        # Run the individual RAG flow
        result = rag_flow(
            question=q, answer_gt=a, system_prompt=system_prompt, **kwargs
        )
        result["question"] = q  # Add the question for reference
        results.append(result)

    # Calculate aggregate metrics
    df = pd.DataFrame(results)

    return {
        "individual_results": results,
        "aggregate_metrics": {
            "avg_relevancy": df["relevancy"].mean(),
            "avg_faithfulness": df["faithfulness"].mean(),
            "avg_precision": df["precision"].mean(),
            "avg_recall": df["recall"].mean(),
        },
    }
