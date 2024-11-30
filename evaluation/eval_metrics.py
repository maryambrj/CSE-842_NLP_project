from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from transformers import pipeline
import nltk

nltk.download('wordnet')
nltk.download('omw-1.4')

def evaluate_metrics(reference, candidate):
    """
    Evaluate a candidate text against a reference using multiple metrics.
    """
    reference_tokens = reference.split()
    candidate_tokens = candidate.split()
    
    smoothing_function = SmoothingFunction().method1
    bleu_score = sentence_bleu(
        [reference_tokens],
        candidate_tokens,
        smoothing_function=smoothing_function
    )
    
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = rouge_scorer_obj.score(reference, candidate)
    
    meteor_score = nltk.translate.meteor_score.meteor_score(
        [reference_tokens], candidate_tokens
    )
    
    P, R, F1 = bert_score([candidate], [reference], lang='en')
    bert_f1 = F1.mean().item()
    
    gleu_score = nltk.translate.gleu_score.sentence_gleu(
        [reference_tokens], candidate_tokens
    )
    
    # BLEURT
    # bleurt = pipeline("text-classification", model="google/bleurt-base-128")
    # bleurt_score = bleurt({"text": candidate, "reference": reference})[0]['score']
    
    # # Placeholder for MoverScore implementation

    results = {
        "BLEU": bleu_score,
        "ROUGE-1": rouge_scores['rouge1'].fmeasure,
        "ROUGE-2": rouge_scores['rouge2'].fmeasure,
        "ROUGE-L": rouge_scores['rougeL'].fmeasure,
        "METEOR": meteor_score,
        "BERTScore (F1)": bert_f1,
        "GLEU": gleu_score,
        # "BLEURT": bleurt_score,
        # "MoverScore": mover_score
    }
    
    return results

reference_text = "ground truth"

candidate_text = "generated output"

metrics_results = evaluate_metrics(reference_text, candidate_text)
print(metrics_results)
