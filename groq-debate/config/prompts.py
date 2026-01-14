"""
Phishing-specific agent prompts optimized for debate-based classification.
These prompts are designed to work with open-source LLMs (Llama, Mixtral, etc.)
"""

# ============================================================================
# TECHNICAL SECURITY ANALYST (Agent 1)
# ============================================================================

TECHNICAL_ANALYST_PROMPT = """You are a Technical Email Security Analyst. Your job is to analyze emails for phishing indicators using technical evidence.

**Your expertise:**
- URL analysis (malicious links, domain spoofing, IP addresses in URLs)
- Email header analysis (sender domain authenticity, spoofing)
- Phishing signatures (known patterns, suspicious TLDs)
- Technical red flags (URL shorteners, mismatched domains)

**Analysis approach:**
1. Examine sender email address for spoofing or suspicious domains
2. Check for suspicious URLs, IP addresses, or link patterns
3. Look for technical inconsistencies
4. Identify known phishing indicators

**Email to analyze:**
---
From: {sender}
To: {receiver}
Subject: {subject}

Body:
{body}

URLs detected: {urls_info}
---

{debate_context}

**Instructions:**
- Be specific: Quote exact suspicious elements from the email
- Use evidence: Reference concrete technical indicators
- Be concise: 2-3 short paragraphs maximum
- State your assessment: Clearly say if you think it's PHISHING or LEGITIMATE
- Explain why: Give 2-3 key technical reasons

**Your technical assessment:**"""


# ============================================================================
# SOCIAL ENGINEERING EXPERT (Agent 2)
# ============================================================================

SOCIAL_ENGINEERING_PROMPT = """You are a Social Engineering Expert. Your job is to analyze emails for psychological manipulation tactics used in phishing.

**Your expertise:**
- Urgency and pressure tactics ("Act now!", "Account will be suspended")
- Authority impersonation (pretending to be banks, tech companies, executives)
- Emotional manipulation (fear, greed, curiosity)
- Language analysis (grammar, tone, phrasing)
- Context plausibility (does the scenario make sense?)

**Analysis approach:**
1. Identify emotional triggers and urgency language
2. Check if sender-content relationship is plausible
3. Look for authority impersonation attempts
4. Analyze grammar, tone, and professionalism
5. Evaluate if the request/scenario makes sense

**Email to analyze:**
---
From: {sender}
To: {receiver}
Subject: {subject}

Body:
{body}
---

{debate_context}

**Instructions:**
- Identify tactics: Quote specific manipulation attempts
- Analyze language: Note unusual phrasing or grammar
- Evaluate plausibility: Does this scenario make sense?
- Be concise: 2-3 short paragraphs maximum
- State your assessment: Clearly say if you think it's PHISHING or LEGITIMATE
- Explain why: Give 2-3 key behavioral/psychological reasons

**Your behavioral assessment:**"""


# ============================================================================
# SECURITY JUDGE (Final Decision Maker)
# ============================================================================

JUDGE_PROMPT = """You are a Chief Information Security Officer making the final decision on whether this email is phishing or legitimate.

**Your task:**
1. Review arguments from both the Technical Analyst and Social Engineering Expert
2. Weigh the evidence (technical indicators vs behavioral patterns)
3. Make a clear binary decision: PHISHING or LEGITIMATE
4. Provide confidence score (0.0 to 1.0)
5. Explain your reasoning based on the strongest evidence

**Email summary:**
---
From: {sender}
Subject: {subject}
Body preview: {body_preview}
---

**Debate transcript:**
{debate_history}

**Instructions:**
You MUST respond in this EXACT format:

CLASSIFICATION: [Write exactly "PHISHING" or "LEGITIMATE" - nothing else]

REASONING: [In 2-3 sentences, explain the key factors that led to your decision. Reference specific evidence from the analysts.]

CONFIDENCE: [A number between 0.0 and 1.0, where 1.0 = absolutely certain. For example: 0.85]

**Your final decision:**"""


# ============================================================================
# PROMPT FORMATTING FUNCTIONS
# ============================================================================

def format_urls_info(urls: str, body: str) -> str:
    """Format URL information for technical analyst."""
    if not urls or urls.strip() == "":
        return "None found in email"
    
    # Check if URLs are present in body
    url_list = [u.strip() for u in urls.split(',') if u.strip()]
    if not url_list:
        return "None found in email"
    
    # Analyze URL characteristics
    info_parts = [f"Found {len(url_list)} URL(s): {', '.join(url_list[:3])}"]
    
    # Check for suspicious patterns
    suspicious = []
    for url in url_list:
        url_lower = url.lower()
        if any(tld in url_lower for tld in ['.tk', '.ml', '.ga', '.cf', '.gq']):
            suspicious.append("suspicious TLD")
        if 'bit.ly' in url_lower or 'tinyurl' in url_lower:
            suspicious.append("URL shortener")
        if any(char.isdigit() for char in url.split('.')[0]) and '://' in url:
            suspicious.append("possible IP address")
    
    if suspicious:
        info_parts.append(f"Suspicious indicators: {', '.join(set(suspicious))}")
    
    return ". ".join(info_parts)


def format_debate_context(debate_history: list, current_round: int) -> str:
    """Format debate history for context."""
    if not debate_history or current_round == 1:
        return "**Current round:** Opening arguments (no prior debate)"
    
    context = f"**Current round:** Round {current_round}\n\n**Previous arguments:**\n"
    for msg in debate_history:
        context += f"\n{msg['agent']} (Round {msg['round']}):\n{msg['message']}\n"
    
    return context


def get_technical_analyst_prompt(email: dict, debate_history: list = None, round_num: int = 1) -> str:
    """Generate technical analyst prompt."""
    urls_info = format_urls_info(email.get('urls', ''), email['body'])
    debate_context = format_debate_context(debate_history or [], round_num)
    
    return TECHNICAL_ANALYST_PROMPT.format(
        sender=email['sender'],
        receiver=email.get('receiver', 'unknown'),
        subject=email['subject'],
        body=email['body'][:1000],  # Limit body length
        urls_info=urls_info,
        debate_context=debate_context
    )


def get_social_engineering_prompt(email: dict, debate_history: list = None, round_num: int = 1) -> str:
    """Generate social engineering expert prompt."""
    debate_context = format_debate_context(debate_history or [], round_num)
    
    return SOCIAL_ENGINEERING_PROMPT.format(
        sender=email['sender'],
        receiver=email.get('receiver', 'unknown'),
        subject=email['subject'],
        body=email['body'][:1000],  # Limit body length
        debate_context=debate_context
    )


def get_judge_prompt(email: dict, debate_history: list) -> str:
    """Generate judge prompt with full debate history."""
    # Format debate history
    history_text = ""
    for msg in debate_history:
        history_text += f"\n**{msg['agent']} (Round {msg['round']}):**\n{msg['message']}\n"
    
    return JUDGE_PROMPT.format(
        sender=email['sender'],
        subject=email['subject'],
        body_preview=email['body'][:300] + "..." if len(email['body']) > 300 else email['body'],
        debate_history=history_text
    )


# ============================================================================
# VALIDATION PROMPT (for testing prompt quality)
# ============================================================================

VALIDATION_PROMPT = """Evaluate the quality of this phishing classification debate.

**Email:**
Subject: {subject}
From: {sender}
True Label: {true_label}

**Debate Result:**
Prediction: {prediction}
Confidence: {confidence}

**Debate Reasoning:**
{reasoning}

**Evaluation criteria:**
1. Relevance: Does the reasoning mention relevant phishing indicators?
2. Specificity: Does it reference specific elements from the email?
3. Accuracy: Is the technical/behavioral analysis correct?
4. Logic: Is the reasoning sound and well-structured?

**Score each criterion (1-5) and provide brief feedback:**

RELEVANCE: [score]
SPECIFICITY: [score]
ACCURACY: [score]
LOGIC: [score]

OVERALL QUALITY: [score]

FEEDBACK: [brief explanation]"""


def get_validation_prompt(email: dict, prediction: str, confidence: float, reasoning: str) -> str:
    """Generate validation prompt for assessing debate quality."""
    true_label = "Phishing" if email['label'] == 1 else "Legitimate"
    
    return VALIDATION_PROMPT.format(
        subject=email['subject'],
        sender=email['sender'],
        true_label=true_label,
        prediction=prediction,
        confidence=confidence,
        reasoning=reasoning
    )
