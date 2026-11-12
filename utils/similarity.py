import json
import logging
from .resume_generator import call_llm, extract_json
from .prompts import ATS_SCORER_PROMPT


logger = logging.getLogger(__name__)

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
            "match_score": None,
            "missing_skills": [],
            "suggestions": [],
            "error": "ATS analysis could not be completed. Please try again.",
        }
    
    # Robust JSON extraction
    try:
        json_str = extract_json(raw_response)
        data = json.loads(json_str.strip())
        if not isinstance(data, dict):
            raise ValueError("Expected object")
        required_fields = {"match_score", "missing_skills", "suggestions"}
        if not required_fields.issubset(data):
            raise ValueError("Missing required ATS fields")
        match_score = data["match_score"]
        missing_skills = data["missing_skills"]
        suggestions = data["suggestions"]
        if isinstance(match_score, bool) or not isinstance(match_score, int) or not 0 <= match_score <= 100:
            raise ValueError("Invalid match score")
        if not isinstance(missing_skills, list) or not all(isinstance(item, str) for item in missing_skills):
            raise ValueError("Invalid missing skills")
        if not isinstance(suggestions, list) or not all(isinstance(item, str) for item in suggestions):
            raise ValueError("Invalid suggestions")
        return {
            "match_score": match_score,
            "missing_skills": missing_skills,
            "suggestions": suggestions,
        }
    except (ValueError, TypeError) as exc:
        logger.warning("ATS response could not be parsed or validated (category=%s)", type(exc).__name__)
        return {
            "match_score": None,
            "missing_skills": [],
            "suggestions": [],
            "error": "The AI returned an incomplete or malformed ATS result. Please try calculating again.",
        }
