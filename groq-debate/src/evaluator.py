"""
Evaluation metrics calculator for phishing detection debates.
"""

from typing import List, Dict
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from src.models import DebateResult, EvaluationMetrics


class MetricsCalculator:
    """Calculate evaluation metrics from debate results."""
    
    @staticmethod
    def calculate_metrics(results: List[DebateResult]) -> EvaluationMetrics:
        """
        Calculate comprehensive metrics from debate results.
        
        Args:
            results: List of DebateResult objects
            
        Returns:
            EvaluationMetrics object
        """
        # Extract predictions and labels
        predictions = []
        true_labels = []
        confidences = []
        confidences_correct = []
        confidences_incorrect = []
        processing_times = []
        
        uncertain_count = 0
        failed_count = 0
        
        for result in results:
            # Convert prediction to binary (0/1)
            if result.prediction == "phishing":
                pred = 1
            elif result.prediction == "legitimate":
                pred = 0
            else:  # uncertain or failed
                uncertain_count += 1
                if result.confidence == 0.0:
                    failed_count += 1
                # Treat uncertain as incorrect for metrics
                pred = 1 - result.true_label  # Force incorrect
            
            predictions.append(pred)
            true_labels.append(result.true_label)
            confidences.append(result.confidence)
            processing_times.append(result.processing_time)
            
            if result.correct:
                confidences_correct.append(result.confidence)
            else:
                confidences_incorrect.append(result.confidence)
        
        # Convert to numpy arrays
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, zero_division=0)
        recall = recall_score(true_labels, predictions, zero_division=0)
        f1 = f1_score(true_labels, predictions, zero_division=0)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
        
        # Confidence statistics
        avg_confidence = np.mean(confidences)
        avg_confidence_correct = np.mean(confidences_correct) if confidences_correct else 0.0
        avg_confidence_incorrect = np.mean(confidences_incorrect) if confidences_incorrect else 0.0
        
        # Processing statistics
        total_time = sum(processing_times)
        avg_time = np.mean(processing_times)
        
        # Dataset composition
        phishing_count = int(np.sum(true_labels == 1))
        legitimate_count = int(np.sum(true_labels == 0))
        
        return EvaluationMetrics(
            total_emails=len(results),
            phishing_count=phishing_count,
            legitimate_count=legitimate_count,
            predictions=[r.prediction for r in results],
            true_labels=true_labels.tolist(),
            accuracy=float(accuracy),
            precision=float(precision),
            recall=float(recall),
            f1_score=float(f1),
            true_positives=int(tp),
            true_negatives=int(tn),
            false_positives=int(fp),
            false_negatives=int(fn),
            avg_confidence=float(avg_confidence),
            avg_confidence_correct=float(avg_confidence_correct),
            avg_confidence_incorrect=float(avg_confidence_incorrect),
            total_time=float(total_time),
            avg_time_per_email=float(avg_time),
            uncertain_count=uncertain_count,
            failed_count=failed_count
        )
    
    @staticmethod
    def print_metrics(metrics: EvaluationMetrics, verbose: bool = True):
        """Print metrics in a formatted way."""
        print("\n" + "="*80)
        print("EVALUATION METRICS")
        print("="*80)
        
        print(f"\n📊 Dataset Composition:")
        print(f"  Total emails: {metrics.total_emails}")
        print(f"  Phishing: {metrics.phishing_count} ({metrics.phishing_count/metrics.total_emails*100:.1f}%)")
        print(f"  Legitimate: {metrics.legitimate_count} ({metrics.legitimate_count/metrics.total_emails*100:.1f}%)")
        
        print(f"\n🎯 Performance Metrics:")
        print(f"  Accuracy:  {metrics.accuracy:.4f} ({metrics.accuracy*100:.2f}%)")
        print(f"  Precision: {metrics.precision:.4f} ({metrics.precision*100:.2f}%)")
        print(f"  Recall:    {metrics.recall:.4f} ({metrics.recall*100:.2f}%)")
        print(f"  F1 Score:  {metrics.f1_score:.4f} ({metrics.f1_score*100:.2f}%)")
        
        print(f"\n📈 Confusion Matrix:")
        print(f"  True Positives:  {metrics.true_positives}")
        print(f"  True Negatives:  {metrics.true_negatives}")
        print(f"  False Positives: {metrics.false_positives}")
        print(f"  False Negatives: {metrics.false_negatives}")
        
        print(f"\n💭 Confidence Statistics:")
        print(f"  Average confidence: {metrics.avg_confidence:.4f}")
        print(f"  Avg (correct):      {metrics.avg_confidence_correct:.4f}")
        print(f"  Avg (incorrect):    {metrics.avg_confidence_incorrect:.4f}")
        
        print(f"\n⏱️  Processing Statistics:")
        print(f"  Total time: {metrics.total_time:.1f}s ({metrics.total_time/60:.1f} min)")
        print(f"  Avg per email: {metrics.avg_time_per_email:.2f}s")
        print(f"  Throughput: {metrics.total_emails/metrics.total_time:.2f} emails/sec")
        
        if metrics.uncertain_count > 0 or metrics.failed_count > 0:
            print(f"\n⚠️  Issues:")
            print(f"  Uncertain predictions: {metrics.uncertain_count}")
            print(f"  Failed debates: {metrics.failed_count}")
        
        print("="*80)
    
    @staticmethod
    def get_error_analysis(results: List[DebateResult]) -> Dict:
        """Analyze errors to identify patterns."""
        false_positives = []  # Predicted phishing, actually legitimate
        false_negatives = []  # Predicted legitimate, actually phishing
        
        for result in results:
            if not result.correct:
                if result.prediction == "phishing" and result.true_label == 0:
                    false_positives.append(result)
                elif result.prediction == "legitimate" and result.true_label == 1:
                    false_negatives.append(result)
        
        return {
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'fp_count': len(false_positives),
            'fn_count': len(false_negatives)
        }
    
    @staticmethod
    def compare_with_ml(
        debate_metrics: EvaluationMetrics,
        ml_metrics: Dict
    ) -> Dict:
        """
        Compare debate system metrics with ML baseline.
        
        Args:
            debate_metrics: EvaluationMetrics from debate system
            ml_metrics: Dictionary with ML metrics (accuracy, precision, recall, f1)
            
        Returns:
            Dictionary with comparison results
        """
        comparison = {
            'debate': {
                'accuracy': debate_metrics.accuracy,
                'precision': debate_metrics.precision,
                'recall': debate_metrics.recall,
                'f1_score': debate_metrics.f1_score
            },
            'ml': ml_metrics,
            'differences': {
                'accuracy': debate_metrics.accuracy - ml_metrics['accuracy'],
                'precision': debate_metrics.precision - ml_metrics['precision'],
                'recall': debate_metrics.recall - ml_metrics['recall'],
                'f1_score': debate_metrics.f1_score - ml_metrics['f1_score']
            }
        }
        
        # Determine winners
        comparison['winners'] = {
            'accuracy': 'debate' if comparison['differences']['accuracy'] > 0 else 'ml',
            'precision': 'debate' if comparison['differences']['precision'] > 0 else 'ml',
            'recall': 'debate' if comparison['differences']['recall'] > 0 else 'ml',
            'f1_score': 'debate' if comparison['differences']['f1_score'] > 0 else 'ml'
        }
        
        return comparison
