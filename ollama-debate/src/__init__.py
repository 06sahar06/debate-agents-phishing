"""
Initialize the debate_evaluation package.
"""

from src.models import EmailData, DebateMessage, DebateResult, EvaluationMetrics
from src.debate_orchestrator import DebateOrchestrator, BatchDebateProcessor
from src.evaluator import MetricsCalculator

__all__ = [
    'EmailData',
    'DebateMessage',
    'DebateResult',
    'EvaluationMetrics',
    'DebateOrchestrator',
    'BatchDebateProcessor',
    'MetricsCalculator'
]
