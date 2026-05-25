import json
import logging
from .resume_generator import call_llm
from .prompts import ATS_SCORER_PROMPT

def analyze_ats_match(user_text: str, job_description: str) -> dict:
    """
    Ranks the user's semantic fit and gaps against the JD using an OpenRouter LLM call.
    Returns a dictionary with match_score, missing_skills, and suggestions.
    """
    if not user_text.strip() or not job_description.strip():
        return {
            "match_score": 0,
            "missing_skills": ["No input provided"],
            "suggestions": ["Please provide both your profile data and the target job description."]
        }
        
    prompt = ATS_SCORER_PROMPT.format(
        user_text=user_text,
        job_description=job_description
    )
    
    raw_response = call_llm(prompt)
    
    if raw_response.startswith("⚠️"):
        return {
            "match_score": 0,
            "missing_skills": ["API Error"],
            "suggestions": [raw_response]
        }
    
    # Robust JSON extraction
    import re
    match = re.search(r'\{(?:[^{}]|(?R))*\}|\{.*\}', raw_response, re.DOTALL)
    if match:
        json_str = match.group(0)
    else:
        json_str = raw_response
        
    try:
        data = json.loads(json_str.strip())
        return data
    except Exception as e:
        logging.error("Failed to parse ATS LLM output into JSON: %s\nRAW OUTPUT: %s", str(e), raw_response)
        return {
            "match_score": 0,
            "missing_skills": ["Error parsing AI response"],
            "suggestions": ["The AI returned unstructured text. Please try calculating again."]
        }
