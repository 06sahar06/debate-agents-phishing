#!/usr/bin/env python3
"""
Final Debate Agents for Phishing Classification
Balanced evaluation with proper 50/50 split for accurate metrics
"""

import time
import pandas as pd
import requests
import json
from pathlib import Path
from datetime import datetime
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import concurrent.futures
import random

# Configuration
DATASETS_DIR = Path("datasets/processed")
RESULTS_DIR = Path("evaluation_results_comprehensive")
RESULTS_DIR.mkdir(exist_ok=True)

# Models for debate (all three as requested)
MODELS = [
    {"name": "qwen2.5:3b-instruct", "weight": 1.0},
    {"name": "llama3.2:latest", "weight": 1.0},
    {"name": "gemma:2b", "weight": 1.0}
]

OLLAMA_API = "http://localhost:11434/api/generate"
TIMEOUT = 8
MAX_TOKENS = 10

def build_optimized_prompt(email_text: str) -> str:
    """Optimized prompt for fast, accurate classification"""
    email_snippet = email_text[:150].strip()
    
    prompt = f"""Is this email PHISHING or LEGITIMATE?

PHISHING signs: suspicious links, urgency, asks for passwords, generic greetings
LEGITIMATE signs: normal business communication, specific context, no suspicious elements

Email: "{email_snippet}"

Answer (one word):"""
    
    return prompt

def call_ollama_fast(model: str, prompt: str) -> tuple:
    """Fast Ollama API call"""
    try:
        start_time = time.time()
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": MAX_TOKENS,
                "top_p": 0.9,
                "stop": ["\n", ".", "Email:", "Answer:"]
            }
        }
        
        response = requests.post(OLLAMA_API, json=payload, timeout=TIMEOUT)
        response.raise_for_status()
        
        result = response.json()
        response_text = result.get("response", "").strip()
        inference_time = time.time() - start_time
        
        return response_text, inference_time, None
        
    except Exception as e:
        return "", 0, str(e)[:30]

def extract_classification_fast(response: str) -> str:
    """Fast classification extraction"""
    response_upper = response.upper().strip()
    
    if "PHISHING" in response_upper:
        return "PHISHING"
    elif "LEGITIMATE" in response_upper:
        return "LEGITIMATE"
    elif "LEGIT" in response_upper:
        return "LEGITIMATE"
    elif "SPAM" in response_upper:
        return "PHISHING"
    elif "SCAM" in response_upper:
        return "PHISHING"
    else:
        return "LEGITIMATE"

def classify_email_parallel(email_text: str) -> dict:
    """Classify email using parallel model calls for debate"""
    start_time = time.time()
    
    prompt = build_optimized_prompt(email_text)
    
    # Parallel model calls
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_to_model = {
            executor.submit(call_ollama_fast, model["name"], prompt): model 
            for model in MODELS
        }
        
        for future in concurrent.futures.as_completed(future_to_model):
            model = future_to_model[future]
            try:
                response, inference_time, error = future.result()
                
                if error:
                    classification = "LEGITIMATE"
                else:
                    classification = extract_classification_fast(response)
                
                results.append({
                    "model": model["name"],
                    "classification": classification,
                    "weight": model["weight"],
                    "inference_time": inference_time,
                    "error": error
                })
                
            except Exception as e:
                results.append({
                    "model": model["name"],
                    "classification": "LEGITIMATE",
                    "weight": model["weight"],
                    "inference_time": 0,
                    "error": str(e)[:30]
                })
    
    # Weighted voting (debate resolution)
    votes = []
    for result in results:
        if not result["error"]:
            for _ in range(int(result["weight"])):
                votes.append(result["classification"])
    
    if not votes:
        final_classification = "LEGITIMATE"
        confidence = "error"
    else:
        vote_counts = Counter(votes)
        final_classification = vote_counts.most_common(1)[0][0]
        
        max_votes = max(vote_counts.values())
        if max_votes == len(votes):
            confidence = "unanimous"
        elif max_votes > len(votes) / 2:
            confidence = "majority"
        else:
            confidence = "split"
    
    total_time = time.time() - start_time
    
    return {
        "final_classification": final_classification,
        "confidence": confidence,
        "total_time": total_time,
        "votes": votes,
        "model_results": results
    }

def load_balanced_enron_sample(total_samples: int = 3000) -> tuple:
    """Load balanced 50/50 sample from Enron dataset"""
    print(f"Loading balanced Enron sample: {total_samples} emails (50% phishing, 50% legitimate)")
    
    try:
        df = pd.read_csv(DATASETS_DIR / "enron_clean.csv")
        print(f"Total Enron emails available: {len(df):,}")
        
        # Separate by class
        phishing_emails = df[df['label'] == 1]
        legitimate_emails = df[df['label'] == 0]
        
        print(f"Available phishing: {len(phishing_emails):,}")
        print(f"Available legitimate: {len(legitimate_emails):,}")
        
        # Sample half from each class
        samples_per_class = total_samples // 2
        
        # Sample phishing emails
        if len(phishing_emails) >= samples_per_class:
            phishing_sample = phishing_emails.sample(n=samples_per_class, random_state=42)
        else:
            phishing_sample = phishing_emails
            print(f"Only {len(phishing_emails)} phishing emails available")
        
        # Sample legitimate emails
        if len(legitimate_emails) >= samples_per_class:
            legitimate_sample = legitimate_emails.sample(n=samples_per_class, random_state=42)
        else:
            legitimate_sample = legitimate_emails
            print(f"Only {len(legitimate_emails)} legitimate emails available")
        
        # Combine and shuffle
        combined_df = pd.concat([phishing_sample, legitimate_sample])
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Extract emails and labels
        emails = combined_df["message"].fillna("").astype(str).tolist()
        labels = ["PHISHING" if label == 1 else "LEGITIMATE" for label in combined_df["label"].tolist()]
        
        # Verify balance
        label_counts = Counter(labels)
        print(f"Final sample: {dict(label_counts)} (Total: {len(emails)})")
        
        return emails, labels
        
    except Exception as e:
        print(f"Error loading Enron dataset: {e}")
        return [], []

def calculate_comprehensive_metrics(y_true: list, y_pred: list, times: list) -> dict:
    """Calculate all performance metrics"""
    y_true_bin = [1 if label == "PHISHING" else 0 for label in y_true]
    y_pred_bin = [1 if label == "PHISHING" else 0 for label in y_pred]
    
    accuracy = accuracy_score(y_true_bin, y_pred_bin)
    precision = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    recall = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    
    avg_time = sum(times) / len(times) if times else 0
    emails_per_second = 1.0 / avg_time if avg_time > 0 else 0
    
    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "emails_per_second": round(emails_per_second, 2),
        "avg_time_per_email": round(avg_time, 3),
        "total_time": round(sum(times), 2),
        "sample_count": len(y_true)
    }

def evaluate_balanced_dataset(total_samples: int = 3000):
    """Evaluate on balanced dataset for proper metrics"""
    print(f"\n{'='*70}")
    print(f"BALANCED ENRON DATASET EVALUATION")
    print(f"{'='*70}")
    
    # Load balanced dataset
    emails, true_labels = load_balanced_enron_sample(total_samples)
    
    if not emails:
        print("No emails loaded, skipping evaluation")
        return None
    
    # Process emails with progress tracking
    predictions = []
    times = []
    detailed_results = []
    
    print(f"\nProcessing {len(emails)} emails...")
    start_time = time.time()
    
    for i, (email, true_label) in enumerate(zip(emails, true_labels)):
        # Progress indicator every 100 emails
        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(emails) - i - 1) / rate if rate > 0 else 0
            print(f"Progress: {i+1:4d}/{len(emails)} ({(i+1)/len(emails)*100:5.1f}%) | "
                  f"Rate: {rate:.2f} emails/s | ETA: {eta/60:.1f}min")
        
        result = classify_email_parallel(email)
        pred_label = result["final_classification"]
        
        predictions.append(pred_label)
        times.append(result["total_time"])
        
        detailed_results.append({
            "email_id": i,
            "true_label": true_label,
            "predicted_label": pred_label,
            "confidence": result["confidence"],
            "processing_time": result["total_time"],
            "votes": "|".join(result["votes"]) if result["votes"] else "error"
        })
    
    total_processing_time = time.time() - start_time
    print(f"Completed {len(emails)} emails in {total_processing_time/60:.1f} minutes")
    
    # Calculate metrics
    metrics = calculate_comprehensive_metrics(true_labels, predictions, times)
    
    # Display comprehensive results
    print(f"\nCOMPREHENSIVE RESULTS:")
    print(f"  Sample Size: {metrics['sample_count']:,}")
    print(f"  Accuracy:    {metrics['accuracy']:.4f} ({metrics['accuracy']:.1%})")
    print(f"  Precision:   {metrics['precision']:.4f} ({metrics['precision']:.1%})")
    print(f"  Recall:      {metrics['recall']:.4f} ({metrics['recall']:.1%})")
    print(f"  F1 Score:    {metrics['f1_score']:.4f} ({metrics['f1_score']:.1%})")
    print(f"  Speed:       {metrics['emails_per_second']:.2f} emails/second")
    print(f"  Avg Time:    {metrics['avg_time_per_email']:.3f} seconds/email")
    print(f"  Total Time:  {metrics['total_time']/60:.1f} minutes")
    
    # Show prediction distribution
    pred_counts = Counter(predictions)
    true_counts = Counter(true_labels)
    print(f"  Predictions: {dict(pred_counts)}")
    print(f"  Actual:      {dict(true_counts)}")
    
    # Detailed classification report
    print(f"\nDETAILED CLASSIFICATION REPORT:")
    y_true_bin = [1 if label == "PHISHING" else 0 for label in true_labels]
    y_pred_bin = [1 if label == "PHISHING" else 0 for label in predictions]
    
    report = classification_report(y_true_bin, y_pred_bin, 
                                 target_names=['LEGITIMATE', 'PHISHING'], 
                                 digits=4)
    print(report)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = RESULTS_DIR / f"balanced_evaluation_{timestamp}.csv"
    pd.DataFrame(detailed_results).to_csv(results_file, index=False)
    
    metrics_file = RESULTS_DIR / f"balanced_metrics_{timestamp}.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nResults saved:")
    print(f"  Details: {results_file.name}")
    print(f"  Metrics: {metrics_file.name}")
    
    return metrics

def main():
    """Main evaluation function"""
    print("Final Debate Agents - Balanced Evaluation")
    print("=" * 60)
    print(f"Models: {[m['name'] for m in MODELS]}")
    print(f"Timeout: {TIMEOUT}s | Max Tokens: {MAX_TOKENS}")
    print("-" * 60)
    
    # Quick test
    print("Quick Test:")
    test_email = "URGENT! Your account will be suspended! Click here now: http://fake-bank.com"
    result = classify_email_parallel(test_email)
    print(f"Test classification: {result['final_classification']} ({result['total_time']:.2f}s)")
    
    # Evaluate balanced dataset
    metrics = evaluate_balanced_dataset(total_samples=3000)
    
    if metrics:
        print(f"\nFINAL PERFORMANCE SUMMARY:")
        print(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']:.1%})")
        print(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']:.1%})")
        print(f"  Recall: {metrics['recall']:.4f} ({metrics['recall']:.1%})")
        print(f"  F1 Score: {metrics['f1_score']:.4f} ({metrics['f1_score']:.1%})")
        print(f"  Speed: {metrics['emails_per_second']:.2f} emails/second")
        print(f"  Total Emails: {metrics['sample_count']:,}")
    
    print("\nBalanced evaluation completed!")

if __name__ == "__main__":
    main()