import re
import string
import collections
from nltk.stem import LancasterStemmer


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace.
    From official ReCoRD eval script
    """

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def string_f1_score(prediction, ground_truth):
    """Compute normalized token level F1
    From official ReCoRD eval script
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    """Compute normalized exact match
    From official ReCoRD eval script
    """
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def string_f1_score_lemma(prediction, ground_truth):
    """Compute normalized token level F1
    From official ReCoRD eval script
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score_lemma(prediction, ground_truth):
    """Compute normalized exact match
    From official ReCoRD eval script
    """
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def string_f1_score_stemm(prediction, ground_truth):
    """Compute normalized token level F1
    From official ReCoRD eval script
    """
    stemmer = LancasterStemmer()
    prediction_tokens_before_lemmatizing = normalize_answer(prediction).split()
    ground_truth_tokens_before_lemmatizing = normalize_answer(ground_truth).split()
    prediction_tokens = []
    ground_truth_tokens = []
    for word in prediction_tokens_before_lemmatizing:
        lemma = stemmer.stem(word)
        prediction_tokens.append(lemma)
    for word in ground_truth_tokens_before_lemmatizing:
        lemma = stemmer.stem(word)
        ground_truth_tokens.append(lemma)
    common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score_stemm(prediction, ground_truth):
    stemmer = LancasterStemmer()
    """Compute normalized exact match
    From official ReCoRD eval script
    """
    prediction = normalize_answer(prediction)
    ground_truth = normalize_answer(ground_truth)
    preds = []
    g_truth = []
    for text in prediction:
        lemma = stemmer.stem(text)
        preds.append(text)
    for text in ground_truth:
        lemma = stemmer.stem(text)
        g_truth.append(text)
    return preds == g_truth
