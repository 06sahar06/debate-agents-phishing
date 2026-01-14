"""
Configuration settings for debate evaluation.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT.parent / "datasets"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

# Ensure results directory exists
RESULTS_DIR.mkdir(exist_ok=True)

# API Settings (defaults, can be overridden)
DEFAULT_API_PROVIDER = "groq"
DEFAULT_MODEL_NAME = "llama-3.1-8b-instant"  # Groq model ID (available)
DEFAULT_TEMPERATURE = 0.1  # Lower for consistent classification
DEFAULT_MAX_TOKENS = 1024

# Groq model options
GROQ_MODELS = {
    "llama-3.3-70b": "llama-3.3-70b-versatile",  # New best quality
    "llama-3.1-70b": "llama-3.1-70b-instruct",  # Prior best
    "llama-3.1-8b": "llama-3.1-8b-instant",  # Faster
    "mixtral-8x7b": "mixtral-8x7b-32768",  # Good balance
    "gemma-2-9b": "gemma2-9b-it"  # Alternative
}

# Ollama model options
OLLAMA_MODELS = {
    "llama-3.1-70b": "llama3.1:70b",
    "llama-3.1-8b": "llama3.1:8b",
    "mistral-7b": "mistral:7b",
    "phi-3": "phi3:latest"
}

# Debate Settings
DEFAULT_DEBATE_ROUNDS = 2  # Opening + 1 rebuttal
MIN_DEBATE_ROUNDS = 1
MAX_DEBATE_ROUNDS = 3

# Batch Processing Settings (for rate limit compliance)
DEFAULT_BATCH_SIZE = 30  # Groq free tier: 30 requests/minute
DEFAULT_BATCH_DELAY = 60.0  # Seconds between batches
DEFAULT_MAX_RETRIES = 3

# Evaluation Settings
AVAILABLE_DATASETS = ["sample", "combined", "enron"]
DEFAULT_DATASET = "combined"

# Sample sizes for testing
SAMPLE_SIZES = {
    "small": 50,    # Quick test
    "medium": 200,  # Validation
    "large": 500    # Thorough test
}

# Stratification settings
STRATIFY_BY_LABEL = True  # Maintain class balance in samples

# Output Settings
SAVE_ALL_DEBATES = False  # Save full transcripts for all emails
SAVE_ERRORS_ONLY = True   # Save full transcripts only for misclassifications
OUTPUT_FORMAT = "json"    # "json" or "csv"

# ML Baseline Metrics (from your results)
ML_BASELINE = {
    "combined": {
        "model": "Voting Ensemble",
        "accuracy": 0.9755,
        "precision": 0.9613,
        "recall": 0.9900,
        "f1_score": 0.9755
    },
    "enron": {
        "model": "Voting Ensemble",
        "accuracy": 0.9888,
        "precision": 0.9813,
        "recall": 0.9964,
        "f1_score": 0.9888
    }
}

# Logging Settings
VERBOSE = True
LOG_LEVEL = "INFO"
