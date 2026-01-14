"""
Compare debate system results with ML baseline and generate comprehensive report.

Usage:
    python compare_results.py --debate-results results/combined_results.json
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add current directory to path (debate_evaluation folder)
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import ML_BASELINE, RESULTS_DIR


def load_debate_results(filepath):
    """Load debate evaluation results from JSON."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_comparison_report(debate_results, ml_metrics, output_path):
    """Generate comprehensive markdown comparison report."""
    
    debate_metrics = debate_results['metrics']
    metadata = debate_results['metadata']
    dataset = metadata.get('dataset', 'unknown')
    
    report = f"""# Phishing Detection: ML vs Multi-Agent Debate System

## Evaluation Report

**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Dataset:** {dataset}  
**Total Emails:** {debate_metrics['total_emails']}  
**Phishing:** {debate_metrics['phishing_count']} | **Legitimate:** {debate_metrics['legitimate_count']}

---

## Performance Comparison

### Overall Metrics

| Metric | ML Baseline | Debate System | Difference | Winner |
|--------|-------------|---------------|------------|---------|
| **Accuracy** | {ml_metrics['accuracy']:.4f} ({ml_metrics['accuracy']*100:.2f}%) | {debate_metrics['accuracy']:.4f} ({debate_metrics['accuracy']*100:.2f}%) | {(debate_metrics['accuracy']-ml_metrics['accuracy'])*100:+.2f}% | {'🏆 Debate' if debate_metrics['accuracy'] > ml_metrics['accuracy'] else '🏆 ML'} |
| **Precision** | {ml_metrics['precision']:.4f} ({ml_metrics['precision']*100:.2f}%) | {debate_metrics['precision']:.4f} ({debate_metrics['precision']*100:.2f}%) | {(debate_metrics['precision']-ml_metrics['precision'])*100:+.2f}% | {'🏆 Debate' if debate_metrics['precision'] > ml_metrics['precision'] else '🏆 ML'} |
| **Recall** | {ml_metrics['recall']:.4f} ({ml_metrics['recall']*100:.2f}%) | {debate_metrics['recall']:.4f} ({debate_metrics['recall']*100:.2f}%) | {(debate_metrics['recall']-ml_metrics['recall'])*100:+.2f}% | {'🏆 Debate' if debate_metrics['recall'] > ml_metrics['recall'] else '🏆 ML'} |
| **F1 Score** | {ml_metrics['f1_score']:.4f} ({ml_metrics['f1_score']*100:.2f}%) | {debate_metrics['f1_score']:.4f} ({debate_metrics['f1_score']*100:.2f}%) | {(debate_metrics['f1_score']-ml_metrics['f1_score'])*100:+.2f}% | {'🏆 Debate' if debate_metrics['f1_score'] > ml_metrics['f1_score'] else '🏆 ML'} |

### Confusion Matrix

**Debate System:**
- True Positives: {debate_metrics['true_positives']}
- True Negatives: {debate_metrics['true_negatives']}
- False Positives: {debate_metrics['false_positives']}
- False Negatives: {debate_metrics['false_negatives']}

### Confidence Analysis

| Metric | Debate System |
|--------|---------------|
| Average Confidence | {debate_metrics['avg_confidence']:.4f} |
| Avg (Correct Predictions) | {debate_metrics['avg_confidence_correct']:.4f} |
| Avg (Incorrect Predictions) | {debate_metrics['avg_confidence_incorrect']:.4f} |

**Insight:** {'✅ Good calibration - higher confidence on correct predictions' if debate_metrics['avg_confidence_correct'] > debate_metrics['avg_confidence_incorrect'] else '⚠️ Poor calibration - similar confidence on correct/incorrect'}

---

## Processing Performance

### Speed Comparison

| System | Time per Email | Total Time ({debate_metrics['total_emails']} emails) | Throughput |
|--------|----------------|------------------------------------------------------|------------|
| **ML Baseline** | ~10ms | ~{debate_metrics['total_emails']*0.01:.1f}s ({debate_metrics['total_emails']*0.01/60:.2f} min) | ~100 emails/sec |
| **Debate System** | {debate_metrics['avg_time_per_email']:.2f}s | {debate_metrics['total_time']:.1f}s ({debate_metrics['total_time']/60:.1f} min) | {debate_metrics['total_emails']/debate_metrics['total_time']:.2f} emails/sec |
| **Speed Ratio** | - | ML is {debate_metrics['avg_time_per_email']/0.01:.0f}x faster | - |

### Cost Analysis

| System | API Cost | Total Cost ({debate_metrics['total_emails']} emails) |
|--------|----------|------------------------------------------------------|
| **ML Baseline** | $0 (local) | $0 |
| **Debate System (Groq Free)** | $0 (free tier) | $0 |

---

## Error Analysis

### Error Distribution

| Error Type | Count | Percentage |
|------------|-------|------------|
| False Positives (Legit → Phishing) | {debate_metrics['false_positives']} | {debate_metrics['false_positives']/debate_metrics['total_emails']*100:.2f}% |
| False Negatives (Phishing → Legit) | {debate_metrics['false_negatives']} | {debate_metrics['false_negatives']/debate_metrics['total_emails']*100:.2f}% |
| Uncertain Predictions | {debate_metrics['uncertain_count']} | {debate_metrics['uncertain_count']/debate_metrics['total_emails']*100:.2f}% |
| Failed Debates | {debate_metrics['failed_count']} | {debate_metrics['failed_count']/debate_metrics['total_emails']*100:.2f}% |

### Critical Analysis

**False Negatives (Missed Phishing):** {debate_metrics['false_negatives']}
- ⚠️ This is the most critical error for phishing detection
- ML baseline missed: ~{int((1-ml_metrics['recall'])*debate_metrics['phishing_count'])} phishing emails
- Debate system missed: {debate_metrics['false_negatives']} phishing emails

**False Positives (Legitimate flagged as Phishing):** {debate_metrics['false_positives']}
- Impact: User annoyance, potential legitimate email blocking
- ML baseline: ~{int((1-ml_metrics['precision'])*debate_metrics['legitimate_count'])} false positives
- Debate system: {debate_metrics['false_positives']} false positives

---

## Key Insights

### Strengths of Debate System

1. **Explainability** ✅
   - Provides detailed reasoning for every decision
   - Technical and behavioral analysis transparent
   - Human-reviewable audit trail

2. **Reasoning Quality** ✅
   - Identifies specific phishing indicators
   - Considers multiple perspectives (technical + psychological)
   - Structured argumentation process

### Weaknesses of Debate System

1. **Speed** ⚠️
   - {debate_metrics['avg_time_per_email']/0.01:.0f}x slower than ML
   - Not suitable for real-time bulk processing
   - Takes {debate_metrics['total_time']/60:.1f} minutes vs {debate_metrics['total_emails']*0.01/60:.2f} minutes for ML

2. **Accuracy** {'⚠️' if debate_metrics['f1_score'] < ml_metrics['f1_score'] else '✅'}
   - F1 score: {debate_metrics['f1_score']:.2%} vs ML: {ml_metrics['f1_score']:.2%}
   - Difference: {(debate_metrics['f1_score']-ml_metrics['f1_score'])*100:+.2f}%

---

## Recommendations

### Use ML System When:
- ✅ Processing large email batches (>1000 emails)
- ✅ Real-time filtering required
- ✅ Speed is critical
- ✅ Accuracy > Explainability

### Use Debate System When:
- ✅ Explainability required (regulatory compliance)
- ✅ High-stakes individual emails (executive protection)
- ✅ Understanding ML failures (research/analysis)
- ✅ Generating training data with reasoning

### Hybrid Approach:
1. Use ML for initial classification (fast, accurate)
2. Route uncertain cases (confidence < 0.85) to debate system
3. Get explainability for ~10-20% of emails
4. Best of both worlds: speed + transparency

---

## Model Information

### ML Baseline
- **Model:** {ml_metrics.get('model', 'Voting Ensemble')}
- **Features:** TF-IDF (3000 features) + engineered features
- **Training:** Supervised learning on labeled data

### Debate System
- **API Provider:** {metadata.get('api_provider', 'unknown')}
- **Model:** {metadata.get('model', 'unknown')}
- **Debate Rounds:** {metadata.get('debate_rounds', 'unknown')}
- **Agents:** Technical Analyst + Social Engineering Expert + Judge

---

## Conclusion

"""

    # Add conclusion based on results
    f1_diff = debate_metrics['f1_score'] - ml_metrics['f1_score']
    
    if f1_diff >= -0.05:  # Within 5%
        conclusion = f"""The debate system achieved **competitive performance** with ML baseline (F1 difference: {f1_diff*100:+.2f}%).

**Key Takeaway:** Debate system provides **explainable AI** with minimal accuracy trade-off ({abs(f1_diff)*100:.1f}% difference). This makes it valuable for:
- Regulatory compliance scenarios
- High-stakes email analysis
- Understanding phishing patterns
- Educational purposes

**Recommendation:** Implement **hybrid system** - use ML for bulk processing, debate for uncertain/critical cases.
"""
    else:  # More than 5% worse
        conclusion = f"""The debate system performed **below ML baseline** (F1 difference: {f1_diff*100:+.2f}%).

**Key Takeaway:** While debate system provides excellent explainability, the **accuracy gap** is significant. Current ML model is superior for phishing detection.

**Recommendation:** 
- Use ML as primary detection system
- Use debate system for research and understanding failures
- Consider fine-tuning debate prompts or using larger models
- Evaluate if explainability benefits justify accuracy trade-off
"""
    
    report += conclusion
    report += f"\n---\n\n**Report Generated:** {datetime.now().isoformat()}\n"
    
    # Save report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n📄 Comparison report saved to: {output_path}")
    
    return report


def main():
    """Main comparison function."""
    
    parser = argparse.ArgumentParser(description="Compare debate results with ML baseline")
    parser.add_argument('--debate-results', type=str, required=True,
                        help='Path to debate evaluation results JSON')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for comparison report')
    
    args = parser.parse_args()
    
    # Load debate results
    print(f"\n📂 Loading debate results: {args.debate_results}")
    debate_results = load_debate_results(args.debate_results)
    
    # Get dataset from results
    dataset = debate_results['metadata'].get('dataset', 'combined')
    
    # Get ML baseline
    if dataset not in ML_BASELINE:
        print(f"⚠️  Warning: No ML baseline for dataset '{dataset}', using 'combined'")
        dataset = 'combined'
    
    ml_metrics = ML_BASELINE[dataset]
    
    print(f"  Dataset: {dataset}")
    print(f"  ML Model: {ml_metrics['model']}")
    print(f"  Debate System: {debate_results['metadata'].get('model', 'unknown')}")
    
    # Generate output path
    output_path = args.output or (
        Path(args.debate_results).parent / f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    )
    
    # Generate report
    print(f"\n📊 Generating comparison report...")
    report = generate_comparison_report(debate_results, ml_metrics, output_path)
    
    # Print summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    debate_metrics = debate_results['metrics']
    print(f"\n🎯 F1 Score:")
    print(f"  ML: {ml_metrics['f1_score']:.4f} ({ml_metrics['f1_score']*100:.2f}%)")
    print(f"  Debate: {debate_metrics['f1_score']:.4f} ({debate_metrics['f1_score']*100:.2f}%)")
    print(f"  Difference: {(debate_metrics['f1_score']-ml_metrics['f1_score'])*100:+.2f}%")
    
    print(f"\n🎯 Recall (Critical for Phishing):")
    print(f"  ML: {ml_metrics['recall']:.4f} ({ml_metrics['recall']*100:.2f}%)")
    print(f"  Debate: {debate_metrics['recall']:.4f} ({debate_metrics['recall']*100:.2f}%)")
    print(f"  Difference: {(debate_metrics['recall']-ml_metrics['recall'])*100:+.2f}%")
    
    print(f"\n⏱️  Speed:")
    print(f"  ML: ~10ms per email")
    print(f"  Debate: {debate_metrics['avg_time_per_email']:.2f}s per email")
    print(f"  ML is {debate_metrics['avg_time_per_email']/0.01:.0f}x faster")
    
    print("\n" + "="*80)
    print(f"✅ Full report saved to: {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
