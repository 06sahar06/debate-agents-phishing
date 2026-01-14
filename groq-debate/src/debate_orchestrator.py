"""
Multi-agent debate orchestrator for phishing detection.
Supports multiple API providers: Groq, Ollama, Together AI, Hugging Face.
"""

import os
import time
import re
from typing import Dict, List, Optional, Tuple
from groq import Groq
import requests

from src.models import EmailData, DebateMessage, DebateResult
from config.prompts import (
    get_technical_analyst_prompt,
    get_social_engineering_prompt,
    get_judge_prompt
)


class DebateOrchestrator:
    """Orchestrates phishing detection debates with multiple LLM providers."""
    
    def __init__(
        self,
        api_provider: str = "groq",
        api_key: Optional[str] = None,
        model_name: str = "llama-3.1-8b-instant",
        temperature: float = 0.1,
        max_tokens: int = 1024,
        debate_rounds: int = 2
    ):
        """
        Initialize debate orchestrator.
        
        Args:
            api_provider: "groq", "ollama", "together", or "huggingface"
            api_key: API key (not needed for Ollama)
            model_name: Model identifier
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens per response
            debate_rounds: Number of debate rounds (1-3)
        """
        self.api_provider = api_provider.lower()
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.debate_rounds = debate_rounds
        
        # Initialize API client
        if self.api_provider == "groq":
            if not self.api_key:
                raise ValueError("GROQ_API_KEY must be set for Groq provider")
            self.client = Groq(api_key=self.api_key)
        elif self.api_provider == "ollama":
            self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        elif self.api_provider == "together":
            if not self.api_key:
                raise ValueError("TOGETHER_API_KEY must be set for Together provider")
            self.base_url = "https://api.together.xyz/v1/chat/completions"
        elif self.api_provider == "huggingface":
            if not self.api_key:
                raise ValueError("HF_API_KEY must be set for Hugging Face provider")
            self.base_url = f"https://api-inference.huggingface.co/models/{model_name}"
        else:
            raise ValueError(f"Unsupported API provider: {api_provider}")
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM with appropriate API."""
        if self.api_provider == "groq":
            return self._call_groq(prompt)
        elif self.api_provider == "ollama":
            return self._call_ollama(prompt)
        elif self.api_provider == "together":
            return self._call_together(prompt)
        elif self.api_provider == "huggingface":
            return self._call_huggingface(prompt)
    
    def _call_groq(self, prompt: str) -> str:
        """Call Groq API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Groq API error: {str(e)}")
    
    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama local API."""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens
                    }
                }
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            raise RuntimeError(f"Ollama API error: {str(e)}")
    
    def _call_together(self, prompt: str) -> str:
        """Call Together AI API."""
        try:
            response = requests.post(
                self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens
                }
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"Together AI API error: {str(e)}")
    
    def _call_huggingface(self, prompt: str) -> str:
        """Call Hugging Face Inference API."""
        try:
            response = requests.post(
                self.base_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"inputs": prompt, "parameters": {
                    "temperature": self.temperature,
                    "max_new_tokens": self.max_tokens
                }}
            )
            response.raise_for_status()
            result = response.json()
            if isinstance(result, list):
                return result[0].get("generated_text", "")
            return result.get("generated_text", "")
        except Exception as e:
            raise RuntimeError(f"Hugging Face API error: {str(e)}")
    
    def run_debate(self, email: EmailData) -> DebateResult:
        """
        Run complete debate on email and return result.
        
        Args:
            email: EmailData object with email content
            
        Returns:
            DebateResult with prediction, reasoning, and debate log
        """
        start_time = time.time()
        debate_log = []
        
        # Convert email to dict for prompt formatting
        email_dict = {
            'sender': email.sender,
            'receiver': email.receiver or 'unknown',
            'subject': email.subject,
            'body': email.body,
            'urls': email.urls or '',
            'label': email.label
        }
        
        try:
            # Round 1: Opening arguments
            for round_num in range(1, self.debate_rounds + 1):
                print(f"  Round {round_num}...", end="", flush=True)
                
                # Technical Analyst
                tech_prompt = get_technical_analyst_prompt(
                    email_dict, debate_log, round_num
                )
                tech_response = self._call_llm(tech_prompt)
                debate_log.append({
                    'agent': 'Technical Analyst',
                    'message': tech_response,
                    'round': round_num
                })
                
                # Social Engineering Expert
                social_prompt = get_social_engineering_prompt(
                    email_dict, debate_log, round_num
                )
                social_response = self._call_llm(social_prompt)
                debate_log.append({
                    'agent': 'Social Engineering Expert',
                    'message': social_response,
                    'round': round_num
                })
                
                print(" ✓")
            
            # Final: Judge decision
            print("  Judge deciding...", end="", flush=True)
            judge_prompt = get_judge_prompt(email_dict, debate_log)
            judge_response = self._call_llm(judge_prompt)
            print(" ✓")
            
            # Parse judge response
            prediction, confidence, reasoning = self._parse_judge_response(judge_response)
            
            # Calculate results
            processing_time = time.time() - start_time
            correct = (
                (prediction == "phishing" and email.label == 1) or
                (prediction == "legitimate" and email.label == 0)
            )
            
            return DebateResult(
                email_id=email.email_id or f"{email.dataset_source}_{hash(email.body)[:8]}",
                prediction=prediction,
                confidence=confidence,
                reasoning=reasoning,
                debate_log=[DebateMessage(**msg) for msg in debate_log],
                true_label=email.label,
                correct=correct,
                processing_time=processing_time,
                model_used=f"{self.api_provider}/{self.model_name}"
            )
            
        except Exception as e:
            # Return failed result
            processing_time = time.time() - start_time
            return DebateResult(
                email_id=email.email_id or "unknown",
                prediction="uncertain",
                confidence=0.0,
                reasoning=f"Debate failed: {str(e)}",
                debate_log=[DebateMessage(**msg) for msg in debate_log],
                true_label=email.label,
                correct=False,
                processing_time=processing_time,
                model_used=f"{self.api_provider}/{self.model_name}"
            )
    
    def _parse_judge_response(self, response: str) -> Tuple[str, float, str]:
        """
        Parse judge response to extract classification, confidence, and reasoning.
        
        Returns:
            (prediction, confidence, reasoning)
        """
        # Extract classification
        class_match = re.search(
            r"CLASSIFICATION:\s*(PHISHING|LEGITIMATE)",
            response,
            re.IGNORECASE
        )
        if class_match:
            prediction = class_match.group(1).lower()
        else:
            # Fallback: check for phishing/legitimate in response
            response_lower = response.lower()
            if "phishing" in response_lower and "legitimate" not in response_lower:
                prediction = "phishing"
            elif "legitimate" in response_lower and "phishing" not in response_lower:
                prediction = "legitimate"
            else:
                prediction = "uncertain"
        
        # Extract reasoning
        reasoning_match = re.search(
            r"REASONING:\s*(.+?)(?=\n\nCONFIDENCE:|CONFIDENCE:|$)",
            response,
            re.DOTALL | re.IGNORECASE
        )
        reasoning = reasoning_match.group(1).strip() if reasoning_match else response
        
        # Extract confidence
        conf_match = re.search(
            r"CONFIDENCE:\s*(0?\.\d+|1\.0|0|1)",
            response,
            re.IGNORECASE
        )
        confidence = float(conf_match.group(1)) if conf_match else 0.5
        
        # Ensure confidence is in valid range
        confidence = max(0.0, min(1.0, confidence))
        
        return prediction, confidence, reasoning


# ============================================================================
# BATCH PROCESSING WITH RATE LIMITING
# ============================================================================

class BatchDebateProcessor:
    """Process debates in batches with rate limiting for free API tiers."""
    
    def __init__(
        self,
        orchestrator: DebateOrchestrator,
        batch_size: int = 30,
        delay_between_batches: float = 60.0,
        max_retries: int = 3
    ):
        """
        Initialize batch processor.
        
        Args:
            orchestrator: DebateOrchestrator instance
            batch_size: Number of emails per batch (for rate limiting)
            delay_between_batches: Seconds to wait between batches
            max_retries: Maximum retry attempts for failed debates
        """
        self.orchestrator = orchestrator
        self.batch_size = batch_size
        self.delay_between_batches = delay_between_batches
        self.max_retries = max_retries
    
    def process_emails(
        self,
        emails: List[EmailData],
        verbose: bool = True
    ) -> List[DebateResult]:
        """
        Process list of emails in batches.
        
        Args:
            emails: List of EmailData objects
            verbose: Print progress information
            
        Returns:
            List of DebateResult objects
        """
        results = []
        total_emails = len(emails)
        num_batches = (total_emails + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, total_emails)
            batch = emails[start_idx:end_idx]
            
            if verbose:
                print(f"\nBatch {batch_idx + 1}/{num_batches} "
                      f"(emails {start_idx + 1}-{end_idx}/{total_emails})")
            
            # Process batch
            for email_idx, email in enumerate(batch):
                if verbose:
                    print(f"  [{start_idx + email_idx + 1}/{total_emails}] "
                          f"Email {email.email_id or 'unknown'}...", end="")
                
                # Retry logic
                for attempt in range(self.max_retries):
                    try:
                        result = self.orchestrator.run_debate(email)
                        results.append(result)
                        
                        if verbose:
                            status = "✓" if result.correct else "✗"
                            print(f" {status} ({result.prediction}, "
                                  f"{result.confidence:.2f}, "
                                  f"{result.processing_time:.1f}s)")
                        break
                        
                    except Exception as e:
                        if attempt < self.max_retries - 1:
                            if verbose:
                                print(f" Retry {attempt + 1}...", end="")
                            time.sleep(2 ** attempt)  # Exponential backoff
                        else:
                            if verbose:
                                print(f" Failed: {str(e)}")
                            # Add failed result
                            results.append(DebateResult(
                                email_id=email.email_id or "unknown",
                                prediction="uncertain",
                                confidence=0.0,
                                reasoning=f"Failed after {self.max_retries} attempts: {str(e)}",
                                debate_log=[],
                                true_label=email.label,
                                correct=False,
                                processing_time=0.0,
                                model_used=self.orchestrator.model_name
                            ))
            
            # Delay between batches (except for last batch)
            if batch_idx < num_batches - 1:
                if verbose:
                    print(f"  Waiting {self.delay_between_batches}s before next batch...")
                time.sleep(self.delay_between_batches)
        
        return results
