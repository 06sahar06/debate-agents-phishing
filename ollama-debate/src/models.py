"""
Data models for phishing email debate system.
Uses Pydantic for validation and type safety.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime


class EmailData(BaseModel):
    """Structured email data for phishing analysis."""
    
    # Core fields (from your datasets)
    sender: str
    receiver: Optional[str] = "unknown"
    subject: str
    body: str
    label: int  # 0 = legitimate, 1 = phishing
    date: Optional[str] = None
    
    # Optional fields (from phishing/legit datasets)
    urls: Optional[str] = None
    
    # Email ID for tracking
    email_id: Optional[str] = None
    dataset_source: Optional[str] = None  # "phishing", "legit", "enron"


class DebateMessage(BaseModel):
    """A single message in the debate."""
    
    agent: str  # "Technical Analyst", "Social Engineering Expert", "Judge"
    message: str
    round: int
    timestamp: Optional[datetime] = None


class DebateResult(BaseModel):
    """Output from a phishing email debate."""
    
    # Email being analyzed
    email_id: str
    
    # Prediction
    prediction: Literal["phishing", "legitimate", "uncertain"]
    confidence: float = Field(ge=0.0, le=1.0)
    
    # Reasoning
    reasoning: str
    debate_log: List[DebateMessage]
    
    # Ground truth and evaluation
    true_label: int  # 0 or 1
    correct: bool
    
    # Metadata
    processing_time: float  # seconds
    model_used: str
    total_tokens: Optional[int] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class EvaluationConfig(BaseModel):
    """Configuration for debate evaluation."""
    
    # API settings
    api_provider: Literal["groq", "ollama", "together", "huggingface"] = "groq"
    api_key: Optional[str] = None
    model_name: str = "llama-3.1-8b-instant"  # Groq model ID (available)
    
    # Debate settings
    debate_rounds: int = 2  # Opening + 1 rebuttal
    temperature: float = 0.3  # Lower for consistent classification
    max_tokens: int = 1024  # Per agent response
    
    # Evaluation settings
    batch_size: int = 30  # For rate limit compliance (Groq free tier)
    delay_between_batches: float = 60.0  # seconds
    max_retries: int = 3
    
    # Dataset settings
    dataset: Literal["sample", "combined", "enron"] = "combined"
    sample_size: Optional[int] = None  # For testing
    stratified: bool = True  # Maintain class balance in samples
    
    # Output settings
    save_all_debates: bool = False  # Save full transcripts for all
    save_errors: bool = True  # Save full transcripts for misclassifications
    output_dir: str = "results"


class EvaluationMetrics(BaseModel):
    """Evaluation metrics for the debate system."""
    
    # Dataset info
    total_emails: int
    phishing_count: int
    legitimate_count: int
    
    # Predictions
    predictions: List[str]  # ["phishing", "legitimate", "uncertain"]
    true_labels: List[int]  # [0, 1, 0, 1, ...]
    
    # Performance metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    
    # Confusion matrix
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    
    # Confidence statistics
    avg_confidence: float
    avg_confidence_correct: float
    avg_confidence_incorrect: float
    
    # Processing statistics
    total_time: float  # seconds
    avg_time_per_email: float
    total_tokens: Optional[int] = None
    
    # Error analysis
    uncertain_count: int  # Emails where prediction was "uncertain"
    failed_count: int  # Emails where debate failed
    
    def to_dict(self):
        """Convert to dictionary for saving."""
        return self.model_dump()


class ComparisonResult(BaseModel):
    """Comparison between ML and Debate systems."""
    
    # System identifiers
    ml_model_name: str = "Voting Ensemble"
    debate_model_name: str
    
    # Performance comparison
    ml_metrics: dict  # From your ML results
    debate_metrics: EvaluationMetrics
    
    # Agreement analysis
    agreement_rate: float  # % of emails where both agree
    ml_correct_debate_wrong: int
    debate_correct_ml_wrong: int
    both_correct: int
    both_wrong: int
    
    # Error analysis
    debate_unique_errors: List[str]  # Email IDs where only debate failed
    ml_unique_errors: List[str]  # Email IDs where only ML failed
    
    # Qualitative analysis
    reasoning_quality_score: Optional[float] = None  # Manual assessment
    
    def to_dict(self):
        """Convert to dictionary for saving."""
        return self.model_dump()
