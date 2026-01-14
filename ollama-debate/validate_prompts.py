"""
Validation script to test prompts on a small sample before running full evaluation.

Usage:
    python validate_prompts.py --size 10
"""

import argparse
import sys
from pathlib import Path

# Add current directory to path (debate_evaluation folder)
sys.path.insert(0, str(Path(__file__).parent))

import os
from dotenv import load_dotenv

from src.models import EmailData
from src.debate_orchestrator import DebateOrchestrator
from run_evaluation import load_dataset
from config.settings import RESULTS_DIR


def validate_single_email(orchestrator, email, show_full_debate=True):
    """Run debate on single email and display results."""
    
    print("\n" + "="*80)
    print(f"EMAIL: {email.email_id}")
    print("="*80)
    print(f"From: {email.sender}")
    print(f"Subject: {email.subject}")
    print(f"Body (first 200 chars): {email.body[:200]}...")
    print(f"True Label: {'PHISHING' if email.label == 1 else 'LEGITIMATE'}")
    print("="*80)
    
    # Run debate
    print("\nRunning debate...")
    result = orchestrator.run_debate(email)
    
    # Show results
    print(f"\n🎯 PREDICTION: {result.prediction.upper()}")
    print(f"📊 CONFIDENCE: {result.confidence:.2f}")
    print(f"✓ CORRECT: {result.correct}")
    print(f"⏱️  TIME: {result.processing_time:.2f}s")
    
    if show_full_debate:
        print("\n" + "="*80)
        print("DEBATE TRANSCRIPT")
        print("="*80)
        
        for msg in result.debate_log:
            print(f"\n[Round {msg.round}] {msg.agent}:")
            print("-" * 80)
            print(msg.message)
    
    print("\n" + "="*80)
    print("JUDGE'S REASONING")
    print("="*80)
    print(result.reasoning)
    print("="*80)
    
    return result


def main():
    """Validate prompts on small sample."""
    
    parser = argparse.ArgumentParser(description="Validate debate prompts on sample emails")
    parser.add_argument('--size', type=int, default=10,
                        help='Number of emails to test')
    parser.add_argument('--api', type=str, default='groq',
                        help='API provider')
    parser.add_argument('--model', type=str, default='llama-3.1-8b-instant',
                        help='Model name (Groq 8B instant)')
    parser.add_argument('--show-debates', action='store_true', default=True,
                        help='Show full debate transcripts')
    parser.add_argument('--pause', action='store_true', default=False,
                        help='Pause after each email (default: no pause)')
    
    args = parser.parse_args()
    
    # Load environment
    load_dotenv(Path(__file__).parent / ".env")
    
    # Check API key (only for cloud providers)
    if args.api == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("❌ Error: GROQ_API_KEY not set in .env file")
            print("   Get your free API key at: https://console.groq.com/")
            return 1
    elif args.api == "together":
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            print("❌ Error: TOGETHER_API_KEY not set in .env file")
            return 1
    elif args.api == "ollama":
        # Ollama doesn't need API key
        api_key = None
        print("Using local Ollama (no API key needed)")
    else:
        api_key = None
    
    print("\n" + "="*80)
    print("PROMPT VALIDATION")
    print("="*80)
    print(f"Testing {args.size} emails with {args.api}/{args.model}")
    print("="*80)
    
    # Load sample
    emails = load_dataset("sample", sample_size=args.size, stratify=True)
    
    # Initialize orchestrator
    orchestrator = DebateOrchestrator(
        api_provider=args.api,
        model_name=args.model,
        debate_rounds=2
    )
    
    # Test on each email
    results = []
    correct_count = 0
    
    for i, email in enumerate(emails):
        print(f"\n\n{'='*80}")
        print(f"TESTING EMAIL {i+1}/{len(emails)}")
        print(f"{'='*80}")
        
        result = validate_single_email(orchestrator, email, args.show_debates)
        results.append(result)
        
        if result.correct:
            correct_count += 1
        
        # Pause only if requested
        if args.pause and i < len(emails) - 1:
            response = input("\n\nContinue to next email? (y/n, default=y): ").strip().lower()
            if response == 'n':
                break
    
    # Summary
    print("\n\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    print(f"Emails tested: {len(results)}")
    print(f"Correct: {correct_count}/{len(results)} ({correct_count/len(results)*100:.1f}%)")
    print(f"Average confidence: {sum(r.confidence for r in results)/len(results):.2f}")
    print(f"Average time: {sum(r.processing_time for r in results)/len(results):.2f}s")
    
    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    accuracy = correct_count / len(results)
    if accuracy >= 0.85:
        print("✅ Prompts performing well! Ready for full evaluation.")
    elif accuracy >= 0.70:
        print("⚠️  Prompts working but could be improved.")
        print("   Consider:")
        print("   - Reviewing incorrect predictions")
        print("   - Adjusting agent instructions")
        print("   - Testing with different model")
    else:
        print("❌ Prompts need improvement.")
        print("   - Review debate transcripts")
        print("   - Check if agents are identifying relevant indicators")
        print("   - Consider rephrasing instructions")
    
    print("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
