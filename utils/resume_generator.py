import json
import logging
import os
import random
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import requests
import streamlit as st
from .prompts import RESUME_PROMPT, COVER_LETTER_PROMPT, PORTFOLIO_PROMPT, ANTI_HALLUCINATION

logger = logging.getLogger(__name__)

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_RETRIES = 3
REQUEST_TIMEOUT_SECONDS = 15.0
TOTAL_REQUEST_BUDGET_SECONDS = 45.0
MAX_RETRY_DELAY_SECONDS = 8.0
BACKOFF_BASE_SECONDS = 1.0
BACKOFF_JITTER_SECONDS = 0.5
RETRYABLE_STATUSES = {408, 429, 500, 502, 503, 504}

_HTTP_ERROR_CATEGORIES = {
    400: "invalid_request",
    401: "authentication_error",
    402: "insufficient_credits",
    403: "access_denied",
    404: "model_unavailable",
    408: "timeout",
    429: "rate_limited",
    500: "service_unavailable",
    502: "service_unavailable",
    503: "service_unavailable",
    504: "service_unavailable",
}

class APIError(Exception):
    """Custom exception raised on API failure to prevent Streamlit caching."""
    pass

def _setting(name: str, default: str = "") -> str:
    """Read a non-empty setting from Streamlit secrets, then the environment."""
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    if value is None or not str(value).strip():
        value = os.environ.get(name, default)
    return str(value).strip() if value is not None else ""


def get_api_key():
    """Load the API key without logging or exposing its value."""
    return _setting("OPENROUTER_API_KEY") or None


def _generation_configuration():
    """Return cache-relevant generation settings; deliberately excludes credentials."""
    return (
        _setting("OPENROUTER_MODEL"),
        _setting("OPENROUTER_FALLBACK_MODEL"),
        _setting("OPENROUTER_SITE_URL", "http://localhost:8501"),
        _setting("OPENROUTER_APP_NAME", "AI Career Platform"),
    )


def _error_category(status_code: int) -> str:
    return _HTTP_ERROR_CATEGORIES.get(status_code, "http_error")


def _retry_after_seconds(value: str, now: datetime = None):
    """Parse Retry-After delta-seconds or HTTP-date without exposing header text."""
    if not value:
        return None
    try:
        seconds = float(value.strip())
        return max(0.0, seconds)
    except (TypeError, ValueError):
        pass

    try:
        retry_at = parsedate_to_datetime(value)
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        return max(0.0, (retry_at - current).total_seconds())
    except (TypeError, ValueError, OverflowError):
        return None


def _response_error_category(data: dict, status_code: int) -> str:
    """Classify structured errors by numeric code only; ignore provider messages."""
    category = _error_category(status_code)
    if isinstance(data, dict) and isinstance(data.get("error"), dict):
        code = data["error"].get("code")
        try:
            embedded_status = int(code)
        except (TypeError, ValueError):
            return category
        return _error_category(embedded_status)
    return category


def _backoff_delay(attempt: int, response=None) -> float:
    """Return a bounded Retry-After delay or bounded exponential delay with jitter."""
    if response is not None:
        headers = getattr(response, "headers", {}) or {}
        retry_after = _retry_after_seconds(headers.get("Retry-After"))
        if retry_after is not None:
            return retry_after
    exponential = BACKOFF_BASE_SECONDS * (2 ** max(0, attempt - 1))
    jitter = random.uniform(0.0, BACKOFF_JITTER_SECONDS)
    return min(MAX_RETRY_DELAY_SECONDS, exponential + jitter)


def _request_model(
    prompt: str,
    model: str,
    api_key: str,
    deadline: float = None,
    site_url: str = None,
    app_name: str = None,
):
    """Make bounded requests for one model. Returns (content, sanitized category).

    The monotonic deadline is best-effort: Requests' timeout limits socket
    inactivity rather than total wall-clock time. Late responses are discarded,
    and no new attempt starts once the deadline is exhausted.
    """
    started = time.monotonic()
    deadline = deadline if deadline is not None else started + TOTAL_REQUEST_BUDGET_SECONDS
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": site_url or _setting("OPENROUTER_SITE_URL", "http://localhost:8501"),
        "X-Title": app_name or _setting("OPENROUTER_APP_NAME", "AI Career Platform"),
    }
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    last_error = "service_unavailable"

    for attempt in range(1, MAX_RETRIES + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None, "timeout"

        try:
            response = requests.post(
                API_URL,
                headers=headers,
                json=payload,
                timeout=min(REQUEST_TIMEOUT_SECONDS, remaining),
            )
        except requests.exceptions.Timeout:
            category = "timeout"
            logger.warning("OpenRouter request failed (category=%s, attempt=%d/%d)", category, attempt, MAX_RETRIES)
            last_error = category
            response = None
        except requests.exceptions.ConnectionError:
            category = "connection_error"
            logger.warning("OpenRouter request failed (category=%s, attempt=%d/%d)", category, attempt, MAX_RETRIES)
            last_error = category
            response = None
        except requests.exceptions.RequestException as exc:
            category = "network_error"
            logger.warning(
                "OpenRouter request failed (category=%s, exception=%s, attempt=%d/%d)",
                category,
                type(exc).__name__,
                attempt,
                MAX_RETRIES,
            )
            last_error = category
            response = None

        # requests.post has consumed the response body before returning. Reject
        # late responses before inspecting their contents or returning success.
        if time.monotonic() >= deadline:
            logger.warning("OpenRouter request deadline exhausted (attempt=%d/%d)", attempt, MAX_RETRIES)
            return None, "timeout"

        if response is not None:
            status = response.status_code
            if status < 200 or status >= 300:
                category = _response_error_category({}, status)
                if status not in RETRYABLE_STATUSES:
                    logger.warning("OpenRouter rejected request (HTTP %d, category=%s)", status, category)
                    return None, category
                last_error = category
                logger.warning(
                    "OpenRouter request received HTTP %d (category=%s, attempt=%d/%d)",
                    status,
                    category,
                    attempt,
                    MAX_RETRIES,
                )
            else:
                try:
                    data = response.json()
                except (ValueError, requests.exceptions.JSONDecodeError):
                    data = None
                    last_error = "invalid_response"
                    logger.warning("OpenRouter returned invalid JSON (attempt=%d/%d)", attempt, MAX_RETRIES)

                if isinstance(data, dict) and "error" in data:
                    category = _response_error_category(data, status)
                    logger.warning("OpenRouter returned a structured error (HTTP %d, category=%s)", status, category)
                    if category in {
                        "authentication_error",
                        "insufficient_credits",
                        "access_denied",
                        "invalid_request",
                        "model_unavailable",
                    }:
                        return None, category
                    last_error = category
                elif isinstance(data, dict):
                    choices = data.get("choices")
                    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                        message = choices[0].get("message")
                        content = message.get("content") if isinstance(message, dict) else None
                        if isinstance(content, str) and content.strip():
                            if time.monotonic() >= deadline:
                                logger.warning("OpenRouter response processing exhausted the request deadline")
                                return None, "timeout"
                            return content.strip(), None
                    last_error = "empty_response"
                    logger.warning("OpenRouter returned an empty completion (HTTP %d)", status)
                elif data is not None:
                    last_error = "invalid_response"

        if attempt == MAX_RETRIES:
            break

        delay = _backoff_delay(attempt, response)
        if delay > MAX_RETRY_DELAY_SECONDS:
            logger.warning("OpenRouter retry delay exceeds configured maximum (category=%s)", last_error)
            return None, last_error
        remaining = deadline - time.monotonic()
        if remaining <= delay:
            return None, "timeout" if last_error != "rate_limited" else last_error
        if delay > 0:
            time.sleep(min(delay, remaining))
        if time.monotonic() >= deadline:
            return None, "timeout" if last_error != "rate_limited" else last_error

    logger.error("OpenRouter request exhausted retries (category=%s)", last_error)
    return None, last_error


def _user_error(category: str) -> str:
    messages = {
        "authentication_error": "OpenRouter rejected the API key. Check OPENROUTER_API_KEY.",
        "insufficient_credits": "The OpenRouter account has insufficient credits for this request.",
        "access_denied": "OpenRouter denied access to the configured model.",
        "model_unavailable": "The configured OpenRouter model is unavailable. Check OPENROUTER_MODEL.",
        "rate_limited": "OpenRouter rate limited the request. Please wait and try again.",
        "timeout": "The OpenRouter request timed out. Please try again.",
        "connection_error": "Could not connect to OpenRouter. Check the network and try again.",
        "network_error": "A network error interrupted the OpenRouter request. Please try again.",
        "invalid_request": "OpenRouter rejected the request. Check the model and request configuration.",
        "empty_response": "OpenRouter returned an empty response. Please try again.",
        "invalid_response": "OpenRouter returned an invalid response. Please try again.",
        "service_unavailable": "OpenRouter is temporarily unavailable. Please try again.",
    }
    return f"⚠️ {messages.get(category, messages['service_unavailable'])}"


def call_llm_raw(prompt: str, use_fallback_model: bool = False, _configuration=None) -> str:
    """Call configured models within a shared, best-effort elapsed-time budget.

    The budget prevents new attempts and rejects late responses. A blocking
    Requests call can exceed it if the peer continually sends data, because
    Requests exposes socket-inactivity timeouts rather than a hard total timeout.
    """
    api_key = get_api_key()
    if not api_key:
        logger.error("OpenRouter API key is not configured")
        return "⚠️ OpenRouter API key is not configured. Set OPENROUTER_API_KEY in Streamlit secrets or the process environment."

    configuration = _configuration or _generation_configuration()
    primary_model, fallback_model, site_url, app_name = configuration
    if use_fallback_model:
        if not fallback_model:
            logger.error("Explicit fallback model was requested but is not configured")
            return "⚠️ No fallback model is configured. Set OPENROUTER_FALLBACK_MODEL to enable it."
        model = fallback_model
    else:
        if not primary_model:
            logger.error("OpenRouter primary model is not configured")
            return "⚠️ OpenRouter model is not configured. Set OPENROUTER_MODEL in Streamlit secrets or the process environment."
        model = primary_model

    deadline = time.monotonic() + TOTAL_REQUEST_BUDGET_SECONDS
    content, error = _request_model(prompt, model, api_key, deadline, site_url, app_name)
    if content is not None:
        return content

    if (
        not use_fallback_model
        and error == "model_unavailable"
        and fallback_model
        and fallback_model != primary_model
        and time.monotonic() < deadline
    ):
        logger.info("Primary model unavailable; trying the explicitly configured fallback")
        content, error = _request_model(prompt, fallback_model, api_key, deadline, site_url, app_name)
        if content is not None:
            return content

    return _user_error(error or "service_unavailable")

@st.cache_data(show_spinner=False)
def _cached_call_llm(prompt: str, use_fallback_model: bool, configuration: tuple) -> str:
    """Cached wrapper that raises APIError on failure so Streamlit won't cache it."""
    res = call_llm_raw(prompt, use_fallback_model, _configuration=configuration)
    if res.startswith("⚠️") or res.startswith("Error"):
        raise APIError(res)
    return res

def call_llm(prompt: str, use_fallback_model: bool = False) -> str:
    """Wrapper that leverages caching and handles exceptions gracefully."""
    try:
        return _cached_call_llm(prompt, use_fallback_model, _generation_configuration())
    except APIError as e:
        return str(e)

def extract_json(text: str) -> str:
    """Extract the first valid JSON object without regex or brace-count shortcuts."""
    if not isinstance(text, str):
        raise ValueError("JSON response must be text")
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return text[index:index + end]
    raise ValueError("No valid JSON object found")


def _validate_portfolio_data(value: dict) -> dict:
    if not isinstance(value, dict):
        raise ValueError("Portfolio response must be an object")
    about = value.get("about")
    projects = value.get("projects")
    if not isinstance(about, str) or not about.strip():
        raise ValueError("Portfolio about field must be a non-empty string")
    if not isinstance(projects, list):
        raise ValueError("Portfolio projects field must be a list")

    validated_projects = []
    for project in projects:
        if not isinstance(project, dict):
            raise ValueError("Portfolio project entries must be objects")
        title = project.get("title")
        description = project.get("description")
        if not isinstance(title, str) or not title.strip():
            raise ValueError("Portfolio project title must be a non-empty string")
        if not isinstance(description, str) or not description.strip():
            raise ValueError("Portfolio project description must be a non-empty string")
        validated_projects.append({"title": title.strip(), "description": description.strip()})

    return {"about": about.strip(), "projects": validated_projects}

def _get_safe_field(data: dict, key: str) -> str:
    """Helper to ensure empty fields are explicitly treated to prevent hallucination."""
    val = data.get(key, '')
    val = val.strip() if isinstance(val, str) else ''
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
    name = _get_safe_field(data, 'name')
    
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
    
    try:
        json_str = extract_json(raw_response)
        json_data = json.loads(json_str.strip())
        return _validate_portfolio_data(json_data)
    except (ValueError, TypeError) as exc:
        logger.warning("Portfolio response could not be parsed or validated (category=%s)", type(exc).__name__)
        return {
            "about": "⚠️ Unable to generate portfolio data. The AI response was incomplete or malformed. Please try again.",
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
