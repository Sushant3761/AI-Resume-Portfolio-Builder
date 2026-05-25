import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import streamlit.components.v1 as components

# Required Imports
from utils.resume_generator import (
    generate_resume,
    generate_cover_letter,
    generate_portfolio_data
)
from utils.portfolio_generator import build_portfolio_html
from utils.similarity import (
    analyze_ats_match
)
import re

def _clean_filename(filename: str) -> str:
    """Removes special characters from filenames to prevent OS path issues."""
    filename = os.path.basename(filename)
    return re.sub(r'[^\w\-_\. ]', '_', filename)

def main():
    st.set_page_config(page_title="AI Career Developer Platform", page_icon="💻", layout="wide")

    # Injecting custom high-end Glassmorphic CSS styling
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;700;800&display=swap');
        
        /* Global layout and typography overrides */
        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        /* Custom Padding and Spacing */
        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }
        
        /* Gradient Titles */
        .main-header {
            font-size: 2.8rem;
            font-weight: 800;
            background: linear-gradient(135deg, #6366f1, #a855f7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.1rem;
            letter-spacing: -1.5px;
        }
        
        .sub-header {
            color: #64748b;
            margin-bottom: 1.8rem;
            font-size: 1.1rem;
            font-weight: 400;
        }
        
        /* Slate Styled Section Panels */
        .section-panel {
            background-color: rgba(30, 41, 59, 0.35);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 14px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }
        
        /* Interactive Neon Gradient Buttons */
        .stButton>button {
            border-radius: 8px !important;
            background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
            color: white !important;
            font-weight: 600 !important;
            border: none !important;
            padding: 0.6rem 1.2rem !important;
            transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.2) !important;
            width: 100%;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 18px rgba(99, 102, 241, 0.4) !important;
            background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
            color: #f8fafc !important;
        }
        
        .stButton>button:active {
            transform: translateY(0px) !important;
        }
        
        /* Custom styled sidebar adjustments */
        [data-testid="stSidebar"] {
            background-color: #0b0f19 !important;
            border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
        }
        
        /* Modernized Tabs Selection */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            background-color: rgba(15, 23, 42, 0.3);
            padding: 8px;
            border-radius: 14px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            margin-bottom: 1.5rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 42px;
            white-space: pre;
            background-color: transparent;
            border-radius: 9px;
            color: #64748b;
            font-weight: 500;
            transition: all 0.2s ease;
            padding: 0 20px;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: rgba(99, 102, 241, 0.15) !important;
            color: #a5b4fc !important;
            font-weight: 600 !important;
            box-shadow: inset 0 0 0 1px rgba(99, 102, 241, 0.25) !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Initialize all user profile fields in st.session_state for robust data persistence
    profile_fields = {
        "name": "",
        "target_role": "",
        "email": "",
        "github": "",
        "linkedin": "",
        "education": "",
        "skills": "",
        "experience": "",
        "projects": "",
        "achievements": "",
        "job_description": ""
    }
    
    for key, default in profile_fields.items():
        if key not in st.session_state:
            st.session_state[key] = default

    # Sidebar
    st.sidebar.title("AI Career Platform")
    st.sidebar.info("Build professional, ATS-optimized materials and live portfolios locally.")
    st.sidebar.caption("Powered by OpenRouter API.")
    
    if st.sidebar.button("Reset / Clear All", use_container_width=True):
        # Cleanly wipe all session state keys
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.toast("Profile and caches reset successfully!", icon="🧹")
        st.rerun()
    
    st.markdown('<div class="main-header">AI Resume & Portfolio Builder</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Provide your details to dynamically generate custom career artifacts.</div>', unsafe_allow_html=True)

    # Profiling Section (Bypassing st.form completely to achieve auto-saving/persistence)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div style="font-size: 1.25rem; font-weight: 700; color: #f8fafc; margin-bottom: 0.5rem;">👤 Personal Information</div>', unsafe_allow_html=True)
        st.text_input("Name *", key="name", placeholder="John Doe")
        st.text_input("Target Job Role *", key="target_role", placeholder="Machine Learning Engineer")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.text_input("Email", key="email", placeholder="john@example.com")
        with c2:
            st.text_input("GitHub URL", key="github", placeholder="https://github.com/john")
        with c3:
            st.text_input("LinkedIn URL", key="linkedin", placeholder="https://linkedin.com/in/john")
        
    with col2:
        st.markdown('<div style="font-size: 1.25rem; font-weight: 700; color: #f8fafc; margin-bottom: 0.5rem;">🎯 Target Job Specification</div>', unsafe_allow_html=True)
        st.text_area("Job Description (For ATS Scorer)", key="job_description", help="Paste the JD to match your profile against it.", placeholder="Paste JD here...", height=140)

    st.markdown('<div style="font-size: 1.25rem; font-weight: 700; color: #f8fafc; margin-top: 1rem; margin-bottom: 0.5rem;">💻 Technical Profiling</div>', unsafe_allow_html=True)
    edu_col, skill_col = st.columns(2)
    
    with edu_col:
        st.text_area("Education", key="education", placeholder="B.Sc. in Computer Science, 2020-2024")
        st.text_area("Achievements", key="achievements", placeholder="1st Place Hackathon, Dean's List")
    with skill_col:
        st.text_area("Core Skills *", key="skills", placeholder="Python, C++, Machine Learning, React")
        st.text_area("Experience", key="experience", placeholder="Software Engineering Intern at XYZ")
        
    st.text_area("Projects", key="projects", placeholder="1. E-commerce API (Node.js) - Scaled to 10k reqs\n2. AI Builder (Python)")
    
    # Check profile completion status
    is_profile_ready = bool(st.session_state.name.strip()) and bool(st.session_state.target_role.strip())

    st.divider()

    # Dynamic data mapping
    user_data = {
        "name": st.session_state.name,
        "target_role": st.session_state.target_role,
        "email": st.session_state.email,
        "github": st.session_state.github,
        "linkedin": st.session_state.linkedin,
        "education": st.session_state.education,
        "skills": st.session_state.skills,
        "projects": st.session_state.projects,
        "achievements": st.session_state.achievements,
        "experience": st.session_state.experience
    }

    # Tab interface for cleaner UX
    tab1, tab2, tab3, tab4 = st.tabs(["📄 Resume", "✉️ Cover Letter", "🌐 Portfolio Website", "📊 ATS Scorer"])

    with tab1:
        st.subheader("Generate Resume")
        if not is_profile_ready:
            st.warning("⚠️ Please provide 'Name' and 'Target Job Role' in the inputs above to unlock this generator.")
        else:
            if st.button("Generate / Preview Professional Resume"):
                st.toast("Generating Resume...", icon='⏳')
                with st.spinner("AI is analyzing and formatting your resume..."):
                    resume_output = generate_resume(user_data)
                    
                    if resume_output.startswith("⚠️") or resume_output.startswith("Error"):
                        st.error(resume_output)
                    else:
                        st.success("Resume ready!")
                        st.markdown("### Preview")
                        st.markdown(resume_output)
                        st.info("💡 To save as an ATS-friendly PDF: Press **Ctrl+P** (Windows) or **Cmd+P** (Mac) and select 'Save as PDF'.")

    with tab2:
        st.subheader("Generate Cover Letter")
        if not is_profile_ready:
            st.warning("⚠️ Please provide 'Name' and 'Target Job Role' in the inputs above to unlock this generator.")
        else:
            if st.button("Generate / Preview Tailored Cover Letter"):
                st.toast("Generating Cover Letter...", icon='⏳')
                with st.spinner("Writing the perfect introduction..."):
                    cl_output = generate_cover_letter(user_data)
                    
                    if cl_output.startswith("⚠️") or cl_output.startswith("Error"):
                        st.error(cl_output)
                    else:
                        st.success("Cover letter ready!")
                        st.markdown("### Preview")
                        st.markdown(cl_output)
                        st.info("💡 To save as a PDF: Press **Ctrl+P** (Windows) or **Cmd+P** (Mac) and select 'Save as PDF'.")

    with tab3:
        st.subheader("Generate Live Portfolio Website")
        if not is_profile_ready:
            st.warning("⚠️ Please provide 'Name' and 'Target Job Role' in the inputs above to unlock this generator.")
        else:
            st.markdown("Instantly build an attractive, code-ready single HTML file.")
            theme = st.selectbox("Select Theme", ["portfolio_modern", "portfolio_minimal"], index=0, format_func=lambda x: "Neon Modern (Dark)" if "modern" in x else "Clean Minimal (Light)")
            
            if st.button("Generate / Preview Live Portfolio"):
                if not st.session_state.skills.strip():
                    st.warning("We highly recommend providing some skills before generating a portfolio!")
                    
                st.toast("Generating Portfolio Data...", icon='⏳')
                with st.spinner("Instructing LLM to output Semantic JSON structure..."):
                    portfolio_json = generate_portfolio_data(user_data)
                    
                    if isinstance(portfolio_json.get("about"), str) and portfolio_json.get("about", "").startswith("⚠️"):
                        st.error(portfolio_json["about"])
                    else:
                        final_html = build_portfolio_html(portfolio_json, theme, user_data)
                        st.success("Portfolio HTML compiled successfully!")
                        safe_name = _clean_filename(st.session_state.name)
                        
                        c1, c2 = st.columns([1, 3])
                        with c1:
                            st.download_button(
                                label="⬇️ Download index.html",
                                data=final_html,
                                file_name=f"portfolio_{safe_name}.html",
                                mime="text/html",
                                type="primary",
                                use_container_width=True
                            )
                        
                        st.markdown("### Live Preview")
                        components.html(final_html, height=600, scrolling=True)

    with tab4:
        st.subheader("ATS Keyword Scoring Engine")
        if not st.session_state.job_description.strip():
            st.warning("Please provide a Job Description in the top right box to analyze against.")
        elif not st.session_state.skills.strip() and not st.session_state.projects.strip() and not st.session_state.experience.strip():
            st.warning("Please provide at least your Skills, Experience, or Projects to evaluate.")
        else:
            if st.button("Calculate / Recalculate ATS Match Score"):
                st.toast("Analyzing Semantic Fit with AI...", icon='⏳')
                with st.spinner("LLM is evaluating your profile against the JD..."):
                    combined_user_text = f"{st.session_state.skills} {st.session_state.projects} {st.session_state.experience} {st.session_state.education} {st.session_state.achievements}"
                    
                    ats_result = analyze_ats_match(combined_user_text, st.session_state.job_description)
                    match_score = ats_result.get("match_score", 0)
                    missing_keywords = ats_result.get("missing_skills", [])
                    suggestions = ats_result.get("suggestions", [])

                    st.markdown("### ATS AI Analysis Result")
                    
                    if match_score >= 80:
                        st.success(f"Excellent Match! Score: {match_score}%")
                    elif match_score >= 50:
                        st.info(f"Good Match! Score: {match_score}%")
                    else:
                        st.warning(f"Needs Improvement. Score: {match_score}%")
                        
                    st.progress(match_score / 100.0)

                    if missing_keywords:
                        st.markdown("**Critical Missing Skills (Consider adding these to your profile):**")
                        for kw in missing_keywords:
                            st.markdown(f"🔴 `{kw}`")
                    else:
                        st.success("Amazing! Your profile covers all major technical keywords found in the Job Description.")
                        
                    if suggestions:
                        st.markdown("**AI Suggestions for Improvement:**")
                        for sug in suggestions:
                            st.markdown(f"💡 {sug}")

if __name__ == "__main__":
    main()
