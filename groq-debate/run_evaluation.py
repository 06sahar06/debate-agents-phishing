"""
Main evaluation script for running debate system on phishing datasets.

Usage:
    python run_evaluation.py --dataset combined --size 200
    python run_evaluation.py --dataset enron --batch-size 30
    python run_evaluation.py --api groq --model llama-3.1-70b-versatile
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Add current directory to path (debate_evaluation folder)
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from src.models import EmailData
from src.debate_orchestrator import DebateOrchestrator, BatchDebateProcessor
from src.evaluator import MetricsCalculator
from config.settings import (
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
    DEFAULT_API_PROVIDER,
    DEFAULT_MODEL_NAME,
    DEFAULT_BATCH_SIZE,
    DEFAULT_BATCH_DELAY,
    ML_BASELINE
)


def load_dataset(dataset_name: str, sample_size: int = None, stratify: bool = True):
    """Load and prepare email dataset."""
    
    print(f"\n📂 Loading dataset: {dataset_name}")
    
    if dataset_name == "sample":
        # Load small stratified sample from combined
        phishing_df = pd.read_csv(PROCESSED_DATA_DIR / "phishing_clean.csv")
        legit_df = pd.read_csv(PROCESSED_DATA_DIR / "legit_clean.csv")
        
        # Sample from each
        sample_per_class = (sample_size or 50) // 2
        phishing_sample = phishing_df.sample(n=min(sample_per_class, len(phishing_df)), random_state=42)
        legit_sample = legit_df.sample(n=min(sample_per_class, len(legit_df)), random_state=42)
        
        df = pd.concat([phishing_sample, legit_sample], ignore_index=True)
        
    elif dataset_name == "combined":
        # Load phishing + legit + sample from enron
        phishing_df = pd.read_csv(PROCESSED_DATA_DIR / "phishing_clean.csv")
        legit_df = pd.read_csv(PROCESSED_DATA_DIR / "legit_clean.csv")
        
        # Add enron sample (balanced)
        enron_df = pd.read_csv(PROCESSED_DATA_DIR / "enron_clean.csv")
        enron_sample = enron_df.groupby('label').sample(n=1000, random_state=42)
        
        # Combine
        df = pd.concat([phishing_df, legit_df, enron_sample], ignore_index=True)
        
    elif dataset_name == "enron":
        df = pd.read_csv(PROCESSED_DATA_DIR / "enron_clean.csv")
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Apply sample size if specified
    if sample_size and len(df) > sample_size:
        if stratify and 'label' in df.columns:
            df = df.groupby('label').sample(n=sample_size//2, random_state=42)
        else:
            df = df.sample(n=sample_size, random_state=42)
    
    # Convert to EmailData objects
    emails = []
    for idx, row in df.iterrows():
        # Handle different column names
        if 'message' in row and 'body' not in row:
            body = row['message']
        else:
            body = row.get('body', '')
        
        # Ensure urls is a string (some datasets store counts/ints)
        urls_value = row.get('urls', None)
        if urls_value is not None and not isinstance(urls_value, str):
            urls_value = str(urls_value)
        email = EmailData(
            sender=row.get('sender', 'unknown'),
            receiver=row.get('receiver', 'unknown'),
            subject=row.get('subject', ''),
            body=str(body),
            label=int(row['label']),
            date=row.get('date', None),
            urls=urls_value,
            email_id=f"{dataset_name}_{idx}",
            dataset_source=dataset_name
        )
        emails.append(email)
    
    print(f"  Loaded {len(emails)} emails")
    print(f"  Phishing: {sum(e.label == 1 for e in emails)}")
    print(f"  Legitimate: {sum(e.label == 0 for e in emails)}")
    
    return emails


def save_results(results, metrics, config, output_path):
    """Save evaluation results to JSON file."""
    
    output_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "dataset": config['dataset'],
            "api_provider": config['api_provider'],
            "model": config['model_name'],
            "total_emails": len(results),
            "debate_rounds": config['debate_rounds']
        },
        "metrics": metrics.to_dict(),
        "results": [
            {
                "prediction": r.prediction,
                "true_label": r.true_label,
                "correct": r.correct,
                "confidence": r.confidence,
                "processing_time": r.processing_time,
                "reasoning": r.reasoning if not r.correct or config.get('save_all', False) else None,
                "debate_log": [
                    {"agent": m.agent, "message": m.message, "round": m.round}
                    for m in r.debate_log
                ] if not r.correct or config.get('save_all', False) else None
            }
            for r in results
        ]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {output_path}")


def main():
    """Main evaluation function."""
    
    parser = argparse.ArgumentParser(description="Evaluate debate system on phishing datasets")
    
    # Dataset options
    parser.add_argument('--dataset', type=str, default='combined',
                        choices=['sample', 'combined', 'enron'],
                        help='Dataset to evaluate on')
    parser.add_argument('--size', type=int, default=None,
                        help='Sample size (for testing)')
    parser.add_argument('--stratify', action='store_true', default=True,
                        help='Maintain class balance in samples')
    
    # API options
    parser.add_argument('--api', type=str, default=DEFAULT_API_PROVIDER,
                        choices=['groq', 'ollama', 'together', 'huggingface'],
                        help='API provider to use')
    parser.add_argument('--model', type=str, default='llama-3.1-8b-instant',
                        help='Model name/ID')
    parser.add_argument('--api-key', type=str, default=None,
                        help='API key (or set GROQ_API_KEY env var)')
    
    # Debate options
    parser.add_argument('--rounds', type=int, default=2,
                        help='Number of debate rounds (1-3)')
    parser.add_argument('--temperature', type=float, default=DEFAULT_TEMPERATURE,
                        help='Sampling temperature (0.0-1.0)')
    
    # Batch processing
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                        help='Batch size for rate limiting')
    parser.add_argument('--batch-delay', type=float, default=DEFAULT_BATCH_DELAY,
                        help='Delay between batches (seconds)')
    
    # Output options
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path')
    parser.add_argument('--save-all', action='store_true',
                        help='Save full debate logs for all emails')
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv(Path(__file__).parent / ".env")
    
    # Get API key
    api_key = args.api_key or os.getenv("GROQ_API_KEY")
    if args.api == "groq" and not api_key:
        print("❌ Error: GROQ_API_KEY not set. Please set it in .env file or use --api-key")
        print("   Get your free API key at: https://console.groq.com/")
        return 1
    
    # Print configuration
    print("\n" + "="*80)
    print("DEBATE EVALUATION CONFIGURATION")
    print("="*80)
    print(f"  Dataset: {args.dataset}")
    if args.size:
        print(f"  Sample size: {args.size}")
    print(f"  API Provider: {args.api}")
    print(f"  Model: {args.model}")
    print(f"  Debate rounds: {args.rounds}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Batch delay: {args.batch_delay}s")
    print("="*80)
    
    # Load dataset
    emails = load_dataset(args.dataset, args.size, args.stratify)
    
    # Initialize orchestrator
    print(f"\n🤖 Initializing {args.api} orchestrator...")
    orchestrator = DebateOrchestrator(
        api_provider=args.api,
        api_key=api_key,
        model_name=args.model,
        temperature=args.temperature,
        debate_rounds=args.rounds
    )
    
    # Initialize batch processor
    processor = BatchDebateProcessor(
        orchestrator=orchestrator,
        batch_size=args.batch_size,
        delay_between_batches=args.batch_delay
    )
    
    # Run evaluation
    print(f"\n🔬 Starting evaluation on {len(emails)} emails...")
    print(f"   Estimated time: {len(emails) * 3 / 60:.1f} minutes")
    
    results = processor.process_emails(emails, verbose=True)
    
    # Calculate metrics
    print(f"\n📊 Calculating metrics...")
    metrics = MetricsCalculator.calculate_metrics(results)
    
    # Print metrics
    MetricsCalculator.print_metrics(metrics)
    
    # Compare with ML baseline
    if args.dataset in ML_BASELINE:
        print(f"\n📈 Comparison with ML Baseline:")
        ml_metrics = ML_BASELINE[args.dataset]
        print(f"\n  ML Model: {ml_metrics['model']}")
        print(f"  ML F1: {ml_metrics['f1_score']:.4f} | Debate F1: {metrics.f1_score:.4f}")
        print(f"  ML Recall: {ml_metrics['recall']:.4f} | Debate Recall: {metrics.recall:.4f}")
        print(f"  Difference: {(metrics.f1_score - ml_metrics['f1_score'])*100:+.2f}% F1")
    
    # Save results
    output_path = args.output or (
        RESULTS_DIR / f"{args.dataset}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    
    config = {
        'dataset': args.dataset,
        'api_provider': args.api,
        'model_name': args.model,
        'debate_rounds': args.rounds,
        'save_all': args.save_all
    }
    
    save_results(results, metrics, config, output_path)
    
    # Error analysis
    error_analysis = MetricsCalculator.get_error_analysis(results)
    print(f"\n⚠️  Error Analysis:")
    print(f"  False Positives: {error_analysis['fp_count']}")
    print(f"  False Negatives: {error_analysis['fn_count']}")
    
    # Save error examples
    if error_analysis['fp_count'] > 0 or error_analysis['fn_count'] > 0:
        error_path = output_path.parent / f"{output_path.stem}_errors.json"
        error_data = {
            "false_positives": [
                {
                    "email_id": r.email_id,
                    "reasoning": r.reasoning,
                    "confidence": r.confidence
                }
                for r in error_analysis['false_positives'][:10]  # Save first 10
            ],
            "false_negatives": [
                {
                    "email_id": r.email_id,
                    "reasoning": r.reasoning,
                    "confidence": r.confidence
                }
                for r in error_analysis['false_negatives'][:10]
            ]
        }
        with open(error_path, 'w', encoding='utf-8') as f:
            json.dump(error_data, f, indent=2, ensure_ascii=False)
        print(f"  Error examples saved to: {error_path}")
    
    print(f"\n✅ Evaluation complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
