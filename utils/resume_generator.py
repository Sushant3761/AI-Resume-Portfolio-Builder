import os
import logging
import requests
import streamlit as st
import json
from .prompts import RESUME_PROMPT, COVER_LETTER_PROMPT, PORTFOLIO_PROMPT, ANTI_HALLUCINATION

# Setup basic logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

API_URL = "https://openrouter.ai/api/v1/chat/completions"
PRIMARY_MODEL = "mistralai/mistral-7b-instruct"
FALLBACK_MODEL = "openai/gpt-3.5-turbo"

def get_api_key() -> str:
    """Safely retrieves the API key, favoring Streamlit Secrets over runtime OS Env variables."""
    # Priority 1: Streamlit Secrets
    try:
        if "OPENROUTER_API_KEY" in st.secrets:
            return st.secrets["OPENROUTER_API_KEY"]
    except Exception as e:
        logging.debug("Could not read from st.secrets: %s", str(e))
        
    # Priority 2: OS Environment
    return os.environ.get("OPENROUTER_API_KEY", "")

def call_llm(prompt: str, use_fallback_model: bool = False) -> str:
    """
    Helper function to make robust calls to the OpenRouter API.
    Handles API failures, logs errors, and implements a fallback message.
    """
    api_key = get_api_key()
    
    if not api_key:
        logging.error("API Key check failed: OPENROUTER_API_KEY is None or empty.")
        return "⚠️ API key not configured. Add it to .env (local) or Streamlit Secrets (cloud)."
        
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://example.com/myapp", # Recommended required headers for OpenRouter
        "X-Title": "AI Career Builder"
    }
    
    current_model = FALLBACK_MODEL if use_fallback_model else PRIMARY_MODEL
    
    payload = {
        "model": current_model,
        "messages": [
            {
                "role": "system",
                "content": "You are a highly analytical technical advisor following explicit constraints."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.5,
        "max_tokens": 800
    }
    
    max_retries = 3
    attempt = 0
    last_error_msg = ""
    
    while attempt < max_retries:
        attempt += 1
        try:
            logging.debug(f"Attempting API call {attempt}/{max_retries} with model {current_model}")
            response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            
            # Debug visibility explicitly logging status code and partial body length or text (safely)
            logging.error(f"DEBUG - API Status: {response.status_code}")
            logging.error(f"DEBUG - API Response snippet: {response.text[:200]}...")
            
            if response.status_code == 429:
                logging.error("API failed: Rate limit exceeded (429) on attempt %d", attempt)
                last_error_msg = "⚠️ AI service rate limited. Please wait a few seconds and try again."
                continue
                
            response.raise_for_status()
            data = response.json()
            
            # ... processing
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
                
                # If the error is model-related or massive failure, attempt to switch model
                if attempt == 1 and not use_fallback_model:
                     return call_llm(prompt, use_fallback_model=True)
                     
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
    Raises ValueError if strict JSON isn't returned.
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
        # We simulate a fallback empty JSON if API fails, so the app doesn't crash but shows "Not specified"
        return {"about": raw_response, "projects": []}
    
    # Robust JSON extraction using python parsing mechanics to strip extraneous text
    import re
    match = re.search(r'\{(?:[^{}]|(?R))*\}|\{.*\}', raw_response, re.DOTALL)
    if match:
        json_str = match.group(0)
    else:
        json_str = raw_response # fallback
        
    try:
        json_data = json.loads(json_str.strip())
        return json_data
    except Exception as e:
        logging.error("Failed to parse LLM Output into JSON: %s\nRAW OUTPUT: %s", str(e), raw_response)
        return {
            "about": "⚠️ Unable to generate portfolio data layout. AI returned unstructured text.",
            "projects": []
        }
