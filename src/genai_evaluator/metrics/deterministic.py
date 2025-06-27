import Levenshtein
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")


def compute_cosine_similarity(answer_pred: str, answer_ref: str) -> float:
    """
    Computes cosine similarity between predicted and reference answers using sentence embeddings.

    Args:
        answer_pred (str): The predicted/generated answer.
        answer_ref (str): The ground truth or reference answer.

    Returns:
        float: Cosine similarity score (0 to 1).
    """
    embeddings = model.encode([answer_pred, answer_ref])
    score = cosine_similarity([embeddings[0]], [embeddings[1]])
    # score is an np.float64, convert to float
    return float(score)


def compute_jaccard_similarity(answer_pred: str, answer_ref: str) -> float:
    """
    Computes Jaccard similarity based on word token overlap.

    Args:
        answer_pred (str): The predicted/generated answer.
        answer_ref (str): The ground truth or reference answer.

    Returns:
        float: Jaccard similarity score (0 to 1).
    """
    pred_tokens = set(answer_pred.lower().split())
    ref_tokens = set(answer_ref.lower().split())

    intersection = pred_tokens.intersection(ref_tokens)
    union = pred_tokens.union(ref_tokens)

    return len(intersection) / len(union) if union else 0.0


def compute_levenshtein_ratio(answer_pred: str, answer_ref: str) -> float:
    """
    Computes Levenshtein ratio (edit similarity) between two strings.

    Args:
        answer_pred (str): The predicted/generated answer.
        answer_ref (str): The ground truth or reference answer.

    Returns:
        float: Levenshtein similarity ratio (0 to 1).
    """
    return Levenshtein.ratio(answer_pred, answer_ref)


def compute_bleu_score(answer_pred: str, answer_ref: str) -> float:
    """
    Computes BLEU score (n-gram overlap) between predicted and reference answers.

    Args:
        answer_pred (str): The predicted/generated answer.
        answer_ref (str): The ground truth or reference answer.

    Returns:
        float: BLEU score (0 to 1).
    """
    ref = [answer_ref.split()]
    pred = answer_pred.split()
    smoothie = SmoothingFunction().method4
    return sentence_bleu(ref, pred, smoothing_function=smoothie)


def compute_rouge_l(answer_pred: str, answer_ref: str) -> float:
    """
    Computes ROUGE-L score (Longest Common Subsequence based similarity).

    Args:
        answer_pred (str): The predicted/generated answer.
        answer_ref (str): The ground truth or reference answer.

    Returns:
        float: ROUGE-L F1 score (0 to 1).
    """
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    score = scorer.score(answer_ref, answer_pred)
    return score["rougeL"].fmeasure


def compute_exact_match(answer_pred: str, answer_ref: str) -> float:
    """
    Checks whether the predicted answer exactly matches the reference (case and space normalized).

    Args:
        answer_pred (str): The predicted/generated answer.
        answer_ref (str): The ground truth or reference answer.

    Returns:
        float: 1.0 if exact match, else 0.0.
    """
    return float(answer_pred.strip().lower() == answer_ref.strip().lower())
