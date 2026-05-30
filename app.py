import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import streamlit.components.v1 as components

# Required Imports
from utils.resume_generator import (
    generate_resume,
    generate_cover_letter,
    generate_portfolio_data,
    improve_resume_with_ai,
    refine_cover_letter_tone
)
from utils.portfolio_generator import build_portfolio_html
from utils.similarity import (
    analyze_ats_match
)
from utils.pdf_generator import compile_pdf
from utils.docx_generator import compile_docx
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
        "job_description": "",
        "resume_text": "",
        "resume_versions": [],
        "cover_letter_text": "",
        "cover_letter_versions": [],
        "ats_match_result": None
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
        st.subheader("Generate & Professional Editor")
        if not is_profile_ready:
            st.warning("⚠️ Please provide 'Name' and 'Target Job Role' in the inputs above to unlock this generator.")
        else:
            c1, c2 = st.columns([1, 1])
            with c1:
                if st.button("🔁 Generate / Regenerate Base Resume", type="secondary"):
                    st.toast("Generating base resume...", icon='⏳')
                    with st.spinner("AI is formatting your resume..."):
                        resume_output = generate_resume(user_data)
                        if not (resume_output.startswith("⚠️") or resume_output.startswith("Error")):
                            st.session_state["resume_text"] = resume_output
                            st.session_state["resume_versions"] = [resume_output]
                            st.toast("Base resume generated successfully!", icon="✅")
                            st.rerun()
            
            # If resume_text exists in session_state, display editor and tools
            if st.session_state["resume_text"]:
                st.markdown("### ✏️ Interactive Resume Editor")
                edited_resume = st.text_area(
                    "You can modify any text below directly. Click 'Save & Recheck' to update downloads and ATS review.",
                    value=st.session_state["resume_text"],
                    key="live_resume_editor",
                    height=450
                )
                
                # Version restoration bar
                if len(st.session_state["resume_versions"]) > 1:
                    st.markdown("**Restoration History:**")
                    cols = st.columns(min(len(st.session_state["resume_versions"]), 5))
                    for idx, val in enumerate(st.session_state["resume_versions"][:5]):
                        with cols[idx]:
                            if st.button(f"Version {idx+1}", key=f"restore_res_{idx}"):
                                st.session_state["resume_text"] = val
                                st.toast(f"Rolled back to Version {idx+1}!", icon="⏪")
                                st.rerun()
                
                # Control Actions Grid
                act1, act2 = st.columns(2)
                with act1:
                    if st.button("💾 Save Changes & Recheck ATS", type="primary", use_container_width=True):
                        new_text = st.session_state["live_resume_editor"]
                        st.session_state["resume_text"] = new_text
                        if not st.session_state["resume_versions"] or st.session_state["resume_versions"][-1] != new_text:
                            st.session_state["resume_versions"].append(new_text)
                        
                        # Trigger ATS re-check if JD is present
                        if st.session_state.job_description.strip():
                            with st.spinner("Recalculating ATS fit score..."):
                                ats_result = analyze_ats_match(new_text, st.session_state.job_description)
                                st.session_state["ats_match_result"] = ats_result
                        st.toast("Changes saved and ATS score updated!", icon="✅")
                        st.rerun()
                with act2:
                    if st.button("🪄 AI ImproveAchievements", use_container_width=True):
                        with st.spinner("AI is polishing achievements and metrics..."):
                            improved = improve_resume_with_ai(st.session_state["live_resume_editor"], st.session_state.target_role)
                            if not (improved.startswith("⚠️") or improved.startswith("Error")):
                                st.session_state["resume_text"] = improved
                                st.session_state["resume_versions"].append(improved)
                                st.toast("Resume improved successfully!", icon="✨")
                                st.rerun()
                            else:
                                st.error(improved)
                                
                # Direct PDF/Word Downloads
                st.markdown("### ⬇️ Export / Download Options")
                down1, down2 = st.columns(2)
                with down1:
                    pdf_bytes = compile_pdf(st.session_state["resume_text"])
                    safe_name = _clean_filename(st.session_state.name)
                    st.download_button(
                        label="📄 Download ATS PDF",
                        data=pdf_bytes,
                        file_name=f"resume_{safe_name}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                with down2:
                    docx_bytes = compile_docx(st.session_state["resume_text"])
                    st.download_button(
                        label="📝 Download Word DOCX",
                        data=docx_bytes,
                        file_name=f"resume_{safe_name}.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        use_container_width=True
                    )

    with tab2:
        st.subheader("Generate & Professional Editor")
        if not is_profile_ready:
            st.warning("⚠️ Please provide 'Name' and 'Target Job Role' in the inputs above to unlock this generator.")
        else:
            c1, c2 = st.columns([1, 1])
            with c1:
                if st.button("🔁 Generate / Regenerate Base Cover Letter", type="secondary"):
                    st.toast("Generating base cover letter...", icon='⏳')
                    with st.spinner("Writing tailored introduction..."):
                        cl_output = generate_cover_letter(user_data)
                        if not (cl_output.startswith("⚠️") or cl_output.startswith("Error")):
                            st.session_state["cover_letter_text"] = cl_output
                            st.session_state["cover_letter_versions"] = [cl_output]
                            st.toast("Base cover letter generated successfully!", icon="✅")
                            st.rerun()
            
            if st.session_state["cover_letter_text"]:
                st.markdown("### ✏️ Interactive Cover Letter Editor")
                edited_cl = st.text_area(
                    "You can modify any text below directly. Click 'Save Changes' to update downloads.",
                    value=st.session_state["cover_letter_text"],
                    key="live_cl_editor",
                    height=400
                )
                
                # Restoration History
                if len(st.session_state["cover_letter_versions"]) > 1:
                    st.markdown("**Restoration History:**")
                    cols = st.columns(min(len(st.session_state["cover_letter_versions"]), 5))
                    for idx, val in enumerate(st.session_state["cover_letter_versions"][:5]):
                        with cols[idx]:
                            if st.button(f"Version {idx+1}", key=f"restore_cl_{idx}"):
                                st.session_state["cover_letter_text"] = val
                                st.toast(f"Rolled back to Version {idx+1}!", icon="⏪")
                                st.rerun()
                
                # Tone polishing actions
                st.markdown("**🎭 Quick AI Tone Refinements:**")
                tone1, tone2, tone3, save_cl = st.columns(4)
                with tone1:
                    if st.button("✂️ Shorter", use_container_width=True):
                        with st.spinner("Making it concise..."):
                            shortened = refine_cover_letter_tone(st.session_state["live_cl_editor"], "shorter")
                            if not (shortened.startswith("⚠️") or shortened.startswith("Error")):
                                st.session_state["cover_letter_text"] = shortened
                                st.session_state["cover_letter_versions"].append(shortened)
                                st.toast("Cover letter shortened!", icon="✨")
                                st.rerun()
                with tone2:
                    if st.button("💼 Professional", use_container_width=True):
                        with st.spinner("Elevating tone..."):
                            prof = refine_cover_letter_tone(st.session_state["live_cl_editor"], "more professional")
                            if not (prof.startswith("⚠️") or prof.startswith("Error")):
                                st.session_state["cover_letter_text"] = prof
                                st.session_state["cover_letter_versions"].append(prof)
                                st.toast("Tone made professional!", icon="✨")
                                st.rerun()
                with tone3:
                    if st.button("💻 Technical", use_container_width=True):
                        with st.spinner("Adding technical depth..."):
                            tech = refine_cover_letter_tone(st.session_state["live_cl_editor"], "more technical")
                            if not (tech.startswith("⚠️") or tech.startswith("Error")):
                                st.session_state["cover_letter_text"] = tech
                                st.session_state["cover_letter_versions"].append(tech)
                                st.toast("Tone made more technical!", icon="✨")
                                st.rerun()
                with save_cl:
                    if st.button("💾 Save Changes", type="primary", use_container_width=True):
                        new_cl_text = st.session_state["live_cl_editor"]
                        st.session_state["cover_letter_text"] = new_cl_text
                        if not st.session_state["cover_letter_versions"] or st.session_state["cover_letter_versions"][-1] != new_cl_text:
                            st.session_state["cover_letter_versions"].append(new_cl_text)
                        st.toast("Cover letter saved!", icon="✅")
                        st.rerun()
                
                # Direct Downloads
                st.markdown("### ⬇️ Export / Download Options")
                down1, down2 = st.columns(2)
                with down1:
                    pdf_bytes_cl = compile_pdf(st.session_state["cover_letter_text"])
                    safe_name = _clean_filename(st.session_state.name)
                    st.download_button(
                        label="📄 Download PDF Letter",
                        data=pdf_bytes_cl,
                        file_name=f"cover_letter_{safe_name}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                with down2:
                    docx_bytes_cl = compile_docx(st.session_state["cover_letter_text"])
                    st.download_button(
                        label="📝 Download Word DOCX",
                        data=docx_bytes_cl,
                        file_name=f"cover_letter_{safe_name}.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        use_container_width=True
                    )

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
            # Check if there is an existing ATS score calculation in session state
            if st.session_state["ats_match_result"]:
                ats_result = st.session_state["ats_match_result"]
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
            
            # Button to calculate / recalculate manually
            if st.button("Calculate / Recalculate ATS Match Score"):
                st.toast("Analyzing Semantic Fit with AI...", icon='⏳')
                with st.spinner("LLM is evaluating your profile against the JD..."):
                    # Use the edited resume text if available, otherwise compile default profile fields
                    if st.session_state["resume_text"]:
                        combined_user_text = st.session_state["resume_text"]
                    else:
                        combined_user_text = f"{st.session_state.skills} {st.session_state.projects} {st.session_state.experience} {st.session_state.education} {st.session_state.achievements}"
                    
                    ats_result = analyze_ats_match(combined_user_text, st.session_state.job_description)
                    st.session_state["ats_match_result"] = ats_result
                    st.toast("ATS analysis completed!", icon="✅")
                    st.rerun()

if __name__ == "__main__":
    main()
