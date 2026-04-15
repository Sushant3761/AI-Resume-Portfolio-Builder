import os
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
    calculate_similarity,
    generate_improvement_suggestions
)
from utils.pdf_generator import (
    create_resume_pdf,
    create_cover_letter_pdf,
    _clean_filename
)

def main():
    st.set_page_config(page_title="AI Career Developer Platform", page_icon="💻", layout="wide")

    # Injecting Custom CSS to make it look nicer
    st.markdown("""
        <style>
        .stButton>button { width: 100%; border-radius: 6px; }
        .main-header { font-size: 2.5rem; font-weight: 800; margin-bottom: 0px; color: #1e88e5;}
        .sub-header { color: #555; margin-bottom: 30px; font-size: 1.1rem; }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar
    st.sidebar.title("AI Career Platform")
    st.sidebar.info("Build professional, ATS-optimized materials and live portfolios locally.")
    st.sidebar.caption("Powered by OpenRouter API & Scikit-Learn.")
    
    if st.sidebar.button("Reset / Clear All", use_container_width=True):
        st.session_state.clear()
        st.rerun()
    
    st.markdown('<div class="main-header">AI Resume & Portfolio Builder</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Provide your details to dynamically generate custom career artifacts.</div>', unsafe_allow_html=True)

    with st.form("user_input_form"):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Personal Information")
            name = st.text_input("Name *", placeholder="John Doe")
            target_role = st.text_input("Target Job Role *", placeholder="Machine Learning Engineer")
            
        with col2:
            st.subheader("Target Job Specification")
            job_description = st.text_area("Job Description (For ATS Scorer)", help="Paste the JD to match your profile against it.", placeholder="Paste JD here...", height=130)

        st.subheader("Technical Profiling")
        edu_col, skill_col = st.columns(2)
        with edu_col:
            education = st.text_area("Education", placeholder="B.Sc. in Computer Science, 2020-2024")
            achievements = st.text_area("Achievements", placeholder="1st Place Hackathon, Dean's List")
        with skill_col:
            skills = st.text_area("Core Skills *", placeholder="Python, C++, Machine Learning, React")
            experience = st.text_area("Experience", placeholder="Software Engineering Intern at XYZ")
            
        projects = st.text_area("Projects", placeholder="1. E-commerce API (Node.js) - Scaled to 10k reqs\n2. AI Builder (Python)")
        
        submit_button = st.form_submit_button("Save Profile Defaults", type="primary")

    if not name.strip() or not target_role.strip():
        st.warning("⚠️ Please provide 'Name' and 'Target Job Role' to unlock generators.")
        return

    st.divider()

    # Data collection dictionary
    user_data = {
        "name": name,
        "education": education,
        "skills": skills,
        "projects": projects,
        "achievements": achievements,
        "experience": experience,
        "target_role": target_role
    }

    # Tab interface for cleaner UX
    tab1, tab2, tab3, tab4 = st.tabs(["📄 Resume", "✉️ Cover Letter", "🌐 Portfolio Website", "📊 ATS Scorer"])

    with tab1:
        st.subheader("Generate Resume")
        if st.button("Generate / Regenerate Professional Resume"):
            st.toast("Generating Resume...", icon='⏳')
            with st.spinner("AI is analyzing and formatting your resume..."):
                resume_output = generate_resume(user_data)
                
                if resume_output.startswith("⚠️") or resume_output.startswith("Error"):
                    st.error(resume_output)
                else:
                    st.success("Resume Generated Successfully!")
                    st.markdown("### Preview")
                    st.markdown(resume_output) # Rendered markdown
                    
                    safe_name = _clean_filename(name) if name else "user"
                    pdf_path = f"resume_{safe_name}.pdf"
                    
                    pdf_result = create_resume_pdf(resume_output, pdf_path)
                    if os.path.exists(pdf_result):
                        with open(pdf_result, "rb") as pdf_file:
                            st.download_button(
                                label="Download Resume (PDF)",
                                data=pdf_file,
                                file_name=pdf_path,
                                mime="application/pdf"
                            )

    with tab2:
        st.subheader("Generate Cover Letter")
        if st.button("Generate / Regenerate Tailored Cover Letter"):
            st.toast("Generating Cover Letter...", icon='⏳')
            with st.spinner("Writing the perfect introduction..."):
                cl_output = generate_cover_letter(user_data)
                
                if cl_output.startswith("⚠️") or cl_output.startswith("Error"):
                    st.error(cl_output)
                else:
                    st.success("Cover Letter Generated Successfully!")
                    st.markdown("### Preview")
                    st.markdown(cl_output)
                    
                    safe_name = _clean_filename(name) if name else "user"
                    pdf_path = f"coverletter_{safe_name}.pdf"
                    
                    pdf_result = create_cover_letter_pdf(cl_output, pdf_path)
                    if os.path.exists(pdf_result):
                        with open(pdf_result, "rb") as pdf_file:
                            st.download_button(
                                label="Download Cover Letter (PDF)",
                                data=pdf_file,
                                file_name=pdf_path,
                                mime="application/pdf"
                            )

    with tab3:
        st.subheader("Generate Live Portfolio Website")
        st.markdown("Instantly build an attractive, code-ready single HTML file.")
        
        theme = st.selectbox("Select Theme", ["portfolio_modern", "portfolio_minimal"], index=0, format_func=lambda x: "Neon Modern (Dark)" if "modern" in x else "Clean Minimal (Light)")
        
        if st.button("Generate / Regenerate Live Portfolio"):
            if not skills:
                st.warning("We highly recommend providing some skills before generating a portfolio!")
                
            st.toast("Generating Portfolio Data...", icon='⏳')
            with st.spinner("Instructing LLM to output Semantic JSON structure..."):
                # Fetch structured data
                portfolio_json = generate_portfolio_data(user_data)
                
                if isinstance(portfolio_json.get("about"), str) and portfolio_json.get("about", "").startswith("⚠️"):
                    st.error(portfolio_json["about"])
                else:
                    # Compile HTML template
                    final_html = build_portfolio_html(portfolio_json, theme, user_data)
                    
                    st.success("Portfolio HTML generated securely!")
                    
                    # File download setup
                    safe_name = _clean_filename(name)
                    
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
        
        if st.button("Calculate / Recalculate ATS Match Score"):
            if not job_description:
                st.warning("Please provide a Job Description in the top right box to compare against.")
            elif not skills and not projects and not experience:
                st.warning("Please provide at least your Skills or Projects to calculate a match score.")
            else:
                st.toast("Calculating Similarity...", icon='⏳')
                with st.spinner("Vectorizing text properties with scikit-learn TF-IDF..."):
                    combined_user_text = f"{skills} {projects} {experience} {education} {achievements}"
                    
                    match_score = calculate_similarity(combined_user_text, job_description)
                    missing_keywords = generate_improvement_suggestions(combined_user_text, job_description)

                    st.markdown("### ATS Analysis Result")
                    
                    if match_score >= 80:
                        st.success(f"Excellent Match! Score: {match_score}%")
                    elif match_score >= 50:
                        st.info(f"Good Match! Score: {match_score}%")
                    else:
                        st.warning(f"Needs Improvement. Score: {match_score}%")
                        
                    st.progress(match_score / 100.0)

                    if missing_keywords:
                        st.markdown("**Missed Keywords from JD (Consider adding these to your skills/experience):**")
                        # Format as chips/badges visually
                        for kw in missing_keywords:
                            st.markdown(f"🔴 `{kw}`")
                    else:
                        st.success("Amazing! Your profile covers all major technical keywords found in the Job Description.")

if __name__ == "__main__":
    main()
