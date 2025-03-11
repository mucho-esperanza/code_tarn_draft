import json
import difflib
import evaluate
from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch

# Load the JSON file
with open("translation_results_SimpleCodes.json", "r") as file:
    data = json.load(file)

# Load CodeBERT Score model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "microsoft/codebert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
codebert_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0 if device == "cuda" else -1)

# Function to compute CodeBERT similarity score
def codebert_score(pred, ref):
    inputs = tokenizer([pred, ref], padding=True, truncation=True, return_tensors="pt").to(device)
    outputs = model(**inputs)
    score = torch.nn.functional.softmax(outputs.logits, dim=-1)[0][1].item()
    return score

# Function to evaluate the translated code
def evaluate_code():
    results = []
    for entry in data:
        title = entry["title"]
        generated_code = entry["translated_python_code"].strip()
        expected_code = entry["expected_python_code"].strip()
        
        # BLEU Score
        bleu = corpus_bleu([generated_code], [[expected_code]]).score
        
        # ROUGE Score
        rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        rouge_scores = rouge.score(expected_code, generated_code)
        rouge_l = rouge_scores["rougeL"].fmeasure
        
        # CodeBERT Score
        codebert = codebert_score(generated_code, expected_code)
        
        # Store results
        results.append({
            "title": title,
            "BLEU Score": bleu,
            "ROUGE-L Score": rouge_l,
            "CodeBERT Score": codebert
        })
    
    return results

# Run evaluation
results = evaluate_code()

# Print results
for res in results:
    print(res)
