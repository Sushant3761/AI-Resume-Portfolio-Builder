"""
Prompts configuration file.
Defines dynamic and robust prompts for OpenRouter API interactions, enforcing structued outputs and 
guarding against hallucinated values.
"""

# Common anti-hallucination instruction
ANTI_HALLUCINATION = """
CRITICAL RULE: If any parameter or data field provided by the user is missing or extremely brief, 
do NOT generate fake content, schools, companies, or buzzwords to fill the gap. 
Explicitly write: "Not specified" for that section. Use ONLY the provided input.
"""

RESUME_PROMPT = """You are an advanced executive technical recruiter and ATS resume optimizer.

Generate a STRICT one-page professional resume tailored to the Target Role.

MANDATORY RULES:
- Add explicit headers in ALL CAPS: PROFESSIONAL SUMMARY, TECHNICAL SKILLS, PROJECTS, EXPERIENCE, EDUCATION, ACHIEVEMENTS.
- Maximum 450 words.
- Maximum 3-line professional summary.
- Maximum 2 bullet points per project.
- Maximum 3 achievements.
- Focus only on relevant technical strengths mapping to {target_role}.
- Do not remove measurable metrics (use measurable impact if possible).
- Ensure all sections are fully completed before stopping.
{anti_hallucination}
- Format output cleanly using markdown:
  - Use ## for main headings
  - Use strict markdown bullet points (*)
  - Avoid long paragraphs.

Applicant Data:
Name: {name}
Target Role: {target_role}
Education: {education}
Skills: {skills}
Projects: {projects}
Achievements: {achievements}
Experience: {experience}

Return ONLY the formatted markdown resume text. No extra commentary or AI acknowledgment.
"""

COVER_LETTER_PROMPT = """Write a concise, impactful, and tailored cover letter for the specified target role.

MANDATORY RULES:
- 200–230 words maximum.
- Maximum 4 short paragraphs.
- Avoid long sentences, bracketed text, or placeholders.
- Tone: Concise, Confident, Professional, Direct, Results-oriented.
{anti_hallucination}
- Format output cleanly using markdown.

Applicant Data:
Name: {name}
Target Role: {target_role}
Key Skills: {skills}
Key Projects: {projects}
Experience: {experience}

Return ONLY the cleanly formatted markdown letter text without any subject line or AI acknowledgment.
"""

PORTFOLIO_PROMPT = """You are an AI generating dynamic text for a developer portfolio website.

MANDATORY RULES:
- Tone: Crisp, Professional, Impact-focused, Engineering-oriented.
- Suitable for embedding on a live HTML website.
{anti_hallucination}

Return output in STRICT JSON format exactly as follows:
{
  "about": "A concise 2-3 sentence engaging professional summary about the applicant's focus and trajectory.",
  "projects": [
    {
      "title": "Name of the project",
      "description": "Explicitly state: Problem solved -> Technical Solution implemented -> Measurable Impact. Avoid generic fluff. (2-3 sentences max)"
    }
  ]
}

- Extract up to 4 the best projects explicitly mentioned in the user's input.
- If projects are empty or weak, return an empty list or simply explicitly state "Not specified" in description.
- DO NOT return markdown fences like ```json at the beginning or end of your string. Return just the raw JSON structured object starting with { and ending with }.

Candidate Details:
Name: {name}
Target Role: {target_role}
Core Skills: {skills}
Experience Background: {experience}
Projects Provided: {projects}
"""
