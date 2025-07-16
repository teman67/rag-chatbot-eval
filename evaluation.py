from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from sklearn.metrics import precision_score, recall_score, f1_score

def bleu_score(candidate, reference):
    return sentence_bleu([reference.split()], candidate.split())

def rouge_score_fn(candidate, reference):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    return scorer.score(reference, candidate)

def token_based_precision_recall_f1(candidate, reference):
    y_true = reference.split()
    y_pred = candidate.split()

    true_set = set(y_true)
    pred_set = set(y_pred)

    precision = len(true_set & pred_set) / len(pred_set)
    recall = len(true_set & pred_set) / len(true_set)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {"precision": precision, "recall": recall, "f1": f1}
