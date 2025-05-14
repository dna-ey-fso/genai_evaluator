from typing import Any, Dict

from loguru import logger
from promptflow.tracing import trace

from clients.data_clients import TemplateStore
from clients.llm_clients import LLMClient
from flows.generation_eval_flow import generation_eval_flow
from flows.retrieval_eval_flow import retrieval_eval_flow
from interfaces.interfaces import Prompt, RoleType, VectorStore


@trace
def rag_flow(
    question: str,
    answer_gt: str,
    system_prompt: str,
    llm_client: LLMClient = None,
    template_store: TemplateStore = None,
    vector_store: VectorStore = None,
    evaluate: bool = False,
    num_results: int = 5,
    top_p: float = 1.0,
    temperature: float = 0.05,
) -> Dict[str, Any]:
    """
    Full RAG flow:
    FAISS vector store creation,
    LLM query with context retrieval, and evaluation of the answer.
    Args:
        question (str): The question to be answered.
        answer (str): The ground truth answer for evaluation.
        llm_client (LLMClient): The LLM client to use for generating answers.
        template_store (TemplateStore): The template store for managing templates.
        vector_store (FAISSVectorStore): The FAISS vector store for document retrieval.
        evaluate (bool): Whether to evaluate the generated answer.
        num_results (int): Number of results to retrieve from the vector store.
        top_p (float): Top-p sampling parameter for LLM generation.
        temperature (float): Temperature parameter for LLM generation.
        system_prompt (str): Custom system prompt for the LLM.
    Returns:
        Dict[str, Any]: Dictionary containing the results of the query and evaluation.
    """

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
    logger.debug(f"LLM response: {response}")

    answer_pred = response["content"]

    # Evaluate if requested
    if evaluate:
        result_ret = retrieval_eval_flow(
            question=question,
            answer_pred=answer_pred,
            answer_gt=answer_gt,
            llm_client=llm_client,
            context=context,
            template_store=template_store,
            temperature=temperature,
            top_p=top_p,
        )
        result_gen = generation_eval_flow(
            question=question,
            answer_pred=answer_pred,
            answer_gt=answer_gt,
            llm_client=llm_client,
            context=context,
            template_store=template_store,
            temperature=temperature,
            top_p=top_p,
        )

    return dict(result_ret=result_ret, result_gen=result_gen)
