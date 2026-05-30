import os
import logging
import requests
import streamlit as st
import json
from .prompts import RESUME_PROMPT, COVER_LETTER_PROMPT, PORTFOLIO_PROMPT, ANTI_HALLUCINATION

# Setup basic logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

API_URL = "https://openrouter.ai/api/v1/chat/completions"
PRIMARY_MODEL = "meta-llama/llama-3-8b-instruct:free"
FALLBACK_MODEL = "openai/gpt-3.5-turbo"

class APIError(Exception):
    """Custom exception raised on API failure to prevent Streamlit caching."""
    pass

def get_api_key():
    try:
        return st.secrets["OPENROUTER_API_KEY"]
    except Exception:
        return os.environ.get("OPENROUTER_API_KEY")

def call_llm_raw(prompt: str, use_fallback_model: bool = False) -> str:
    """
    Performs the raw HTTP request to the OpenRouter API.
    Handles retries, rate limits, and errors directly.
    """
    api_key = get_api_key()
    if not api_key:
        logging.error("API Key check failed: OPENROUTER_API_KEY is None or empty.")
        return "⚠️ API key not configured. Add it to .env (local) or Streamlit Secrets (cloud)."
        
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "AI Career Platform"
    }
    
    current_model = FALLBACK_MODEL if use_fallback_model else PRIMARY_MODEL
    
    payload = {
        "model": current_model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }
    
    max_retries = 3
    attempt = 0
    last_error_msg = ""
    
    while attempt < max_retries:
        attempt += 1
        try:
            logging.debug(f"Attempting API call {attempt}/{max_retries} with model {current_model}")
            response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 401:
                last_error_msg = "⚠️ Invalid API Key. Please check your configuration."
                break
            elif response.status_code == 402:
                last_error_msg = "⚠️ API quota exceeded. Please check your billing details."
                break
            elif response.status_code == 429:
                logging.error("API failed: Rate limit exceeded (429) on attempt %d", attempt)
                last_error_msg = "⚠️ AI service rate limited. Please wait a few seconds and try again."
                continue
            elif response.status_code == 404:
                logging.error("API failed: Model Not Found (404) on attempt %d", attempt)
                if attempt == 1 and not use_fallback_model:
                     return call_llm_raw(prompt, use_fallback_model=True)
                last_error_msg = "⚠️ AI service model temporarily disabled."
                continue
            elif response.status_code == 400:
                last_error_msg = "⚠️ Bad API request. Check the input format."
                break
                
            response.raise_for_status()
            data = response.json()
            
            if "choices" in data and len(data["choices"]) > 0:
                content = data["choices"][0].get("message", {}).get("content", "")
                if not content:
                    logging.error("API failed: Empty response format: %s", response.text)
                    last_error_msg = "⚠️ AI service returned an empty response. Please try again later."
                    continue
                return content.strip()
            elif "error" in data:
                error_message = data["error"].get("message", "Unknown Error") if isinstance(data["error"], dict) else str(data["error"])
                logging.error("API failed with embedded error on attempt %d: %s", attempt, error_message)
                last_error_msg = "⚠️ AI service temporarily unavailable. Please try again later."
                
                if attempt == 1 and not use_fallback_model:
                     return call_llm_raw(prompt, use_fallback_model=True)
                     
                continue
            else:
                logging.error("API failed: Unexpected JSON format %s", response.text)
                last_error_msg = "⚠️ AI service temporarily unavailable. Please try again later."
                continue
                
        except requests.exceptions.Timeout as e:
            logging.error("API failed: Timeout error on attempt %d - %s", attempt, str(e))
            last_error_msg = "⚠️ AI service connection timed out. Please try again later."
        except requests.exceptions.RequestException as e:
            logging.error("API failed: Request exception on attempt %d - %s", attempt, str(e))
            last_error_msg = "⚠️ AI service temporarily unavailable."
        except Exception as e:
            logging.error("API failed: Unknown exception on attempt %d - %s", attempt, str(e))
            last_error_msg = "⚠️ AI service encountered an unexpected error."
            
    return last_error_msg

@st.cache_data(show_spinner=False)
def _cached_call_llm(prompt: str, use_fallback_model: bool) -> str:
    """Cached wrapper that raises APIError on failure so Streamlit won't cache it."""
    res = call_llm_raw(prompt, use_fallback_model)
    if res.startswith("⚠️") or res.startswith("Error"):
        raise APIError(res)
    return res

def call_llm(prompt: str, use_fallback_model: bool = False) -> str:
    """Wrapper that leverages caching and handles exceptions gracefully."""
    try:
        return _cached_call_llm(prompt, use_fallback_model)
    except APIError as e:
        return str(e)

def extract_json(text: str) -> str:
    """Safely extracts the first valid JSON object starting with '{' and ending with '}'."""
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        return text[start:end+1]
    return text

def _get_safe_field(data: dict, key: str) -> str:
    """Helper to ensure empty fields are explicitly treated to prevent hallucination."""
    val = data.get(key, '').strip()
    return val if val else "Not specified"

def generate_resume(data: dict) -> str:
    """Generates an ATS-optimized professional resume dynamically against user inputs."""
    name = data.get('name', '').strip()
    target_role = _get_safe_field(data, 'target_role')
    
    if not name or target_role == "Not specified":
        return "Error: Name and Target Role are required for operations."
        
    prompt = RESUME_PROMPT.format(
        anti_hallucination=ANTI_HALLUCINATION,
        target_role=target_role,
        name=name,
        education=_get_safe_field(data, 'education'),
        skills=_get_safe_field(data, 'skills'),
        projects=_get_safe_field(data, 'projects'),
        achievements=_get_safe_field(data, 'achievements'),
        experience=_get_safe_field(data, 'experience')
    )
    
    return call_llm(prompt)

def generate_cover_letter(data: dict) -> str:
    """Generates a concise Cover Letter strictly mapped to actual skills."""
    name = data.get('name', '').strip()
    target_role = _get_safe_field(data, 'target_role')
    
    if not name or target_role == "Not specified":
        return "Error: Name and Target Role are required for operations."
        
    prompt = COVER_LETTER_PROMPT.format(
        anti_hallucination=ANTI_HALLUCINATION,
        name=name,
        target_role=target_role,
        skills=_get_safe_field(data, 'skills'),
        projects=_get_safe_field(data, 'projects'),
        experience=_get_safe_field(data, 'experience')
    )
    
    return call_llm(prompt)

def generate_portfolio_data(data: dict) -> dict:
    """
    Generates JSON payload for the Portfolio generator.
    """
    name = data.get('name', '').strip()
    
    prompt = PORTFOLIO_PROMPT.format(
        anti_hallucination=ANTI_HALLUCINATION,
        name=name,
        target_role=_get_safe_field(data, 'target_role'),
        skills=_get_safe_field(data, 'skills'),
        experience=_get_safe_field(data, 'experience'),
        projects=_get_safe_field(data, 'projects')
    )
    
    raw_response = call_llm(prompt)
    
    # Check if raw_response is a fallback error message
    if raw_response.startswith("⚠️"):
        return {"about": raw_response, "projects": []}
    
    # Robust JSON extraction
    json_str = extract_json(raw_response)
        
    try:
        json_data = json.loads(json_str.strip())
        if not isinstance(json_data, dict):
            raise ValueError("Expected dictionary structure")
        json_data.setdefault("about", "Not specified")
        json_data.setdefault("projects", [])
        return json_data
    except Exception as e:
        logging.error("Failed to parse LLM Output into JSON: %s\nRAW OUTPUT: %s", str(e), raw_response)
        return {
            "about": "⚠️ Unable to generate portfolio data layout. AI returned unstructured text.",
            "projects": []
        }

def improve_resume_with_ai(current_text: str, target_role: str) -> str:
    """
    Refines and optimizes the current resume content using the LLM.
    Enforces action verbs, stronger metrics, and strict anti-hallucination.
    """
    prompt = f"""You are an executive resume writer. Take the following resume and improve it:
- Enhance description lines with strong, punchy action verbs.
- Elevate metrics where possible to emphasize business impact.
- Tailor technical vocabulary closer to a standard {target_role} profile.
- Strictly do NOT invent fake education details, certifications, or past employers.
- Keep the structure identical.

Current Resume:
{current_text}

Return ONLY the upgraded markdown resume content. Do not write any introduction, commentary, or explanation.
"""
    return call_llm(prompt)

def refine_cover_letter_tone(current_text: str, tone: str) -> str:
    """
    Adjusts the tone of the cover letter according to user requests.
    Tone can be 'shorter', 'more professional', or 'more technical'.
    """
    tone_instruction = ""
    if tone == "shorter":
        tone_instruction = "Make the text highly concise, removing fluff or redundant sentences while preserving key accomplishments. Aim for ~150 words."
    elif tone == "more professional":
        tone_instruction = "Elevate the language to a polished, executive-level tone. Emphasize professionalism, industry maturity, and crisp alignment."
    elif tone == "more technical":
        tone_instruction = "Incorporate deep, industry-standard technology terms and vocabulary. Align engineering achievements with professional language."
        
    prompt = f"""You are an expert copywriter. Take the following cover letter and adjust its tone:
- Tone adjustment request: {tone_instruction}
- Do NOT invent fake jobs, names, or credentials.
- Keep the core candidate accomplishments intact.

Current Cover Letter:
{current_text}

Return ONLY the refined markdown cover letter content. Do not write any intro, outro, or conversational filler.
"""
    return call_llm(prompt)
