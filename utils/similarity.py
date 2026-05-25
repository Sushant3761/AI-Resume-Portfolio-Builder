import json
import logging
from .resume_generator import call_llm
from .prompts import ATS_SCORER_PROMPT

def extract_json(text: str) -> str:
    """Safely extracts the first valid JSON object starting with '{' and ending with '}'."""
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        return text[start:end+1]
    return text

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
    json_str = extract_json(raw_response)
        
    try:
        data = json.loads(json_str.strip())
        # Validate that the returned data has the required fields
        if not isinstance(data, dict):
            raise ValueError("Expected dictionary")
        # Ensure default keys exist
        data.setdefault("match_score", 0)
        data.setdefault("missing_skills", [])
        data.setdefault("suggestions", [])
        return data
    except Exception as e:
        logging.error("Failed to parse ATS LLM output into JSON: %s\nRAW OUTPUT: %s", str(e), raw_response)
        return {
            "match_score": 0,
            "missing_skills": ["Error parsing AI response"],
            "suggestions": [
                "The AI returned unstructured text. Please try calculating again.",
                f"Raw snippet: {raw_response[:200]}"
            ]
        }
