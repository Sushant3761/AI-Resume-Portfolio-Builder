import os
import requests
import streamlit as st

OPENROUTER_API_KEY = (
    os.getenv("OPENROUTER_API_KEY") 
    or st.secrets.get("OPENROUTER_API_KEY")
)
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "mistralai/mistral-7b-instruct"

def call_llm(prompt: str) -> str:
    """
    Helper function to make robust calls to the OpenRouter API.
    Handles API failures, missing tokens, and extracts the generated text clearly.
    """
    if not OPENROUTER_API_KEY:
        return "API key not configured. Add it to .env (local) or Streamlit Secrets (cloud)."
        
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a professional resume and career advisor."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.7,
        "max_tokens": 450
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        
        # Handle rate limits or other HTTP errors early
        if response.status_code == 429:
            return "Error: Rate limit exceeded for OpenRouter API. Please try again later."
            
        response.raise_for_status()
        data = response.json()
        
        if "choices" in data and len(data["choices"]) > 0:
            content = data["choices"][0].get("message", {}).get("content", "")
            if not content:
                return "Error: Empty response received from the API."
            return content.strip()
        elif "error" in data:
            error_message = data["error"].get("message", "Unknown Error") if isinstance(data["error"], dict) else str(data["error"])
            return f"API Error: {error_message}"
        else:
            return "Error: Unexpected response format from OpenRouter API."
            
    except requests.exceptions.Timeout:
        return "Error: Request to OpenRouter API timed out."
    except requests.exceptions.RequestException as e:
        return f"Error connecting to OpenRouter API: {str(e)}"


def generate_resume(data: dict) -> str:
    """
    Generates an ATS-optimized professional resume based on user details.
    Uses strong prompt engineering to enforce the structure and tone.
    """
    name = data.get('name', '').strip()
    education = data.get('education', '').strip()
    skills = data.get('skills', '').strip()
    projects = data.get('projects', '').strip()
    achievements = data.get('achievements', '').strip()
    experience = data.get('experience', '').strip()
    target_role = data.get('target_role', '').strip()
    
    # Basic validation
    if not name or not target_role:
        return "Error: Name and Target Role are required for resume generation."
    
    # Fill defaults for optional empty fields
    education = education if education else "N/A"
    skills = skills if skills else "N/A"
    projects = projects if projects else "N/A"
    achievements = achievements if achievements else "N/A"
    experience = experience if experience else "N/A"

    prompt = f"""You are a senior technical recruiter and ATS resume optimizer.

Generate a STRICT one-page resume (MAX 450 words).

MANDATORY RULES:
- Add explicit headers in ALL CAPS: PROFESSIONAL SUMMARY, TECHNICAL SKILLS, PROJECTS, EXPERIENCE, EDUCATION, ACHIEVEMENTS.
- Maximum 450 words.
- Maximum 3-line professional summary.
- Maximum 2 bullet points per project.
- Maximum 3 achievements.
- No long paragraphs.
- No repetition.
- No generic phrases like "dynamic", "highly motivated", "passionate".
- Focus only on relevant technical strengths.
- Do not remove measurable metrics (use measurable impact if possible).
- Keep concise bullet format.
- Ensure all sections are fully completed before stopping.
- Ensure EXPERIENCE section is never truncated. If nearing token limit, prioritize completing EXPERIENCE and EDUCATION.

STRUCTURE:

NAME
Contact Information (single line)

PROFESSIONAL SUMMARY (3 lines max)

TECHNICAL SKILLS
Grouped and concise.

PROJECTS
Project Name
• Bullet
• Bullet

EXPERIENCE (if provided)
Role – Company
• Bullet
• Bullet

EDUCATION

ACHIEVEMENTS (max 3 bullets)

Target Role: Software Engineer

Align content with:
- Software validation
- Bug fixing
- Code maintenance
- Testing techniques
- Feature enhancement
- Documentation
- Collaboration

Student Details:
Name: {name}
Education: {education}
Skills: {skills}
Projects: {projects}
Achievements: {achievements}
Experience: {experience}

Return only formatted resume text.
No extra commentary.
- If output exceeds 450 words, rewrite it shorter automatically."""

    return call_llm(prompt)


def generate_cover_letter(data: dict) -> str:
    """
    Generates a concise and impactful cover letter tailored to a specific target role.
    """
    name = data.get('name', '').strip()
    target_role = data.get('target_role', '').strip()
    skills = data.get('skills', '').strip()
    projects = data.get('projects', '').strip()
    
    if not target_role:
        return "Error: Target Role is required for cover letter generation."

    skills = skills if skills else "N/A"
    projects = projects if projects else "N/A"

    prompt = f"""Write a concise and impactful cover letter for a Software Engineer role.

MANDATORY RULES:
- 200–230 words maximum.
- Maximum 4 short paragraphs.
- Avoid long sentences.
- No placeholders like [Company Name].
- No bracketed text.
- No generic phrases ("I am passionate", "dynamic individual").
- Tone: Concise, Confident, Professional, Direct, Engineering-focused, Results-oriented.

STRUCTURE:

Opening paragraph (Sharper, more direct):
- State clear intent for the Software Engineer role immediately.
- Directly connect core skills to technical needs without fluff.

Middle paragraph:
- Highlight 1–2 strong projects or technical contributions.
- Show measurable impact metrics and practical results.

Third paragraph:
- Highlight experience in debugging, testing, code maintenance, and feature enhancement.
- Subtly reference stakeholder collaboration, code reviews, and peer-reviewing modifications.

Closing paragraph:
- Express interest clearly.
- Keep short and confident.

Applicant Details:
Name: {name}
Key Skills: {skills}
Key Projects: {projects}

Return only clean formatted letter text.
No subject line.
No placeholders.
No markdown formatting.
If output exceeds 230 words, rewrite it shorter automatically."""
    
    return call_llm(prompt)


def generate_portfolio_summary(data: dict) -> str:
    """
    Generates a short, professional portfolio summary for personal websites or LinkedIn.
    """
    name = data.get('name', '').strip()
    skills = data.get('skills', '').strip()
    experience = data.get('experience', '').strip()

    skills = skills if skills else "N/A"
    experience = experience if experience else "N/A"

    prompt = f"""You are a technical portfolio strategist.

MANDATORY RULES:
- Limit output to 150–180 words maximum.
- Remove generic phrases like: "dynamic", "passionate", "build the future", "cutting-edge".
- Tone: Crisp, Professional, Impact-focused, Engineering-oriented.
- Suitable for: LinkedIn, Personal website, GitHub bio.
- Return plain text only. No markdown formatting.
- If output exceeds 180 words, rewrite it shorter automatically.

HIGHLIGHT:
- AI Resume Builder project
- API integration
- ML pipeline development
- Spring Boot backend
- AWS/cloud exposure
- Internship under IBM AICTE Edunet

Candidate Details:
Name: {name}
Core Skills: {skills}
Experience Background: {experience}

Return only the portfolio summary text."""

    return call_llm(prompt)
