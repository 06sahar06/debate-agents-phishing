# Debate Agents for Phishing Detection

Multi-agent debate system for email phishing detection using different LLM providers.

## Folders

- **groq-debate/** - Groq API implementation (fast, rate-limited free tier)
- **ollama-debate/** - Ollama local implementation (unlimited, runs locally)

## Quick Start

### Groq Setup
```bash
cd groq-debate
pip install -r requirements.txt
export GROQ_API_KEY=your_key_here
python validate_prompts.py --size 10 --api groq --model llama-3.1-8b-instant
```

### Ollama Setup
```bash
cd ollama-debate
pip install -r requirements.txt
ollama pull llama3.1:8b
python validate_prompts.py --size 10 --api ollama --model llama3.1:8b
```

## Performance

- **Groq (8B)**: 60% accuracy on 20 emails, ~60s per email
- **Ollama (8B)**: 70% accuracy on 10 emails, ~1400s per email
- **ML Baseline**: 97.55% F1 (for comparison)

## Files

- `validate_prompts.py` - Test prompts on sample emails
- `run_evaluation.py` - Full dataset evaluation
- `compare_results.py` - Compare debate vs ML results
- `config/prompts.py` - Agent prompts (Technical Analyst, Social Engineer, Judge)
- `config/settings.py` - Configuration and API settings
- `src/debate_orchestrator.py` - Multi-agent orchestration
- `src/models.py` - Data models
- `src/evaluator.py` - Metrics calculation
