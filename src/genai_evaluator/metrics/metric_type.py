from enum import Enum


class MetricType(Enum):
    """
    Enum for different types of metrics.
    """

    ACCURACY = "accuracy"
    RECALL = "recall"
    PRECISION = "precision"
    F1_SCORE = "f1_score"
    BLEU_SCORE = "bleu_score"
    ROUGE_L_SCORE = "rouge_l_score"
    FAITHFULNESS = "faithfulness"
    RELEVANCY = "relevancy"
    JACCARD_SIMILARITY = "jaccard_similarity"
    COSINE_SIMILARITY = "cosine_similarity"
    LEVENSHTEIN_RATIO = "levenshtein_ratio"
    EXACT_MATCH = "exact_match"
