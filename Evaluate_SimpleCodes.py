import json
import difflib
import evaluate
from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch
from tabulate import tabulate
import code_bert_score

# Load the JSON file
with open("/home/lpc/miniconda3/envs/tran/Hand_code_pairs_Intermediate_Codellama7b.json", "r") as file:
    data = json.load(file)

# Function to evaluate the translated code
def evaluate_code():
    results = []
    predictions = []
    references = []
    titles = []
    
    for entry in data:
        titles.append(entry["title"])
        predictions.append(entry["translated_python_code"].strip())
        references.append(entry["expected_python_code"].strip())
    
    # BLEU and ROUGE Scores
    bleu_scores = [corpus_bleu([pred], [[ref]]).score for pred, ref in zip(predictions, references)]
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_scores = [rouge.score(ref, pred)["rougeL"].fmeasure for pred, ref in zip(predictions, references)]
    
    # CodeBERT Score
    pred_results = code_bert_score.score(cands=predictions, refs=references, lang='python')
    codebert_scores = pred_results[2].tolist()  # Extracting F1 scores
    
    # Store results
    for title, bleu, rouge_l, codebert in zip(titles, bleu_scores, rouge_scores, codebert_scores):
        results.append([title, bleu, rouge_l, codebert])
    
    return results

# Run evaluation
results = evaluate_code()

# Print results in tabular format
headers = ["Title", "BLEU Score", "ROUGE-L Score", "CodeBERT Score"]
print(tabulate(results, headers=headers, tablefmt="fancy_grid"))
