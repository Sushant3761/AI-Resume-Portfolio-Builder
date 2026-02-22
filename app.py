import os
import streamlit as st

# Required Imports
from utils.resume_generator import (
    generate_resume,
    generate_cover_letter,
    generate_portfolio_summary
)
from utils.similarity import (
    calculate_similarity,
    generate_improvement_suggestions
)
from utils.pdf_generator import (
    create_resume_pdf,
    create_cover_letter_pdf,
    create_portfolio_pdf
)

def main():
    st.set_page_config(page_title="AI Resume & Portfolio Builder", page_icon="📄", layout="wide")

    # Sidebar
    st.sidebar.title("Resume & Portfolio Builder")
    st.sidebar.info("Build your professional AI-powered materials locally using OpenRouter API & Scikit-Learn.")

    # Main Details
    st.title("AI Resume & Portfolio Builder")
    st.markdown("Enter your professional details below to generate an ATS-friendly resume, cover letter, or portfolio summary.")

    with st.form("user_input_form"):
        st.header("Personal Information")
        name = st.text_input("Name", placeholder="John Doe")

        st.header("Professional Details")
        education = st.text_area("Education", placeholder="B.Sc. in Computer Science, University of tech, 2020-2024")
        skills = st.text_area("Skills", placeholder="Python, Machine Learning, Streamlit, SQL")
        projects = st.text_area("Projects", placeholder="1. AI Resume Builder\n2. E-commerce Platform")
        achievements = st.text_area("Achievements", placeholder="1st Place Hackathon, Dean's List")
        experience = st.text_area("Experience (Optional)", placeholder="Software Engineering Intern at XYZ Corp")

        st.header("Target Role")
        target_role = st.text_input("Target Job Role", placeholder="Machine Learning Engineer")
        job_description = st.text_area("Optional Job Description", help="Paste the job description you are targeting to get a match score and customized materials.", placeholder="Paste JD here...")

        submit_button = st.form_submit_button("Save Details")

    st.divider()

    # Dictionary collection
    user_data = {
        "name": name,
        "education": education,
        "skills": skills,
        "projects": projects,
        "achievements": achievements,
        "experience": experience,
        "target_role": target_role
    }

    st.subheader("Actions")
    
    # We will use tabs or columns. Columns work well.
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("Generate Resume", use_container_width=True, type="primary"):
            if not name or not target_role:
                st.warning("Please provide at least your Name and Target Job Role.")
            else:
                with st.spinner("Generating Resume..."):
                    try:
                        resume_output = generate_resume(user_data)
                        st.success("Resume Generated Successfully!")
                        st.text_area("Generated Resume", resume_output, height=400)
                        
                        # PDF Generation
                        safe_name = name.replace(" ", "_").lower() if name else "user"
                        pdf_path = f"resume_{safe_name}.pdf"
                        
                        create_resume_pdf(resume_output, pdf_path)
                        
                        with open(pdf_path, "rb") as pdf_file:
                            st.download_button(
                                label="Download Resume as PDF",
                                data=pdf_file,
                                file_name=pdf_path,
                                mime="application/pdf",
                                use_container_width=True
                            )
                    except Exception as e:
                        st.error(f"Error generating resume or PDF: {e}")

    with col2:
        if st.button("Generate Cover Letter", use_container_width=True, type="primary"):
            if not name or not target_role:
                st.warning("Please provide at least your Name and Target Job Role.")
            else:
                with st.spinner("Generating Cover Letter..."):
                    try:
                        cl_output = generate_cover_letter(user_data)
                        st.success("Cover Letter Generated Successfully!")
                        st.text_area("Generated Cover Letter", cl_output, height=400)
                        
                        # PDF Generation
                        safe_name = name.replace(" ", "_").lower() if name else "user"
                        pdf_path = f"cover_letter_{safe_name}.pdf"
                        
                        create_cover_letter_pdf(cl_output, pdf_path)
                        
                        with open(pdf_path, "rb") as pdf_file:
                            st.download_button(
                                label="Download Cover Letter as PDF",
                                data=pdf_file,
                                file_name=pdf_path,
                                mime="application/pdf",
                                use_container_width=True
                            )
                    except Exception as e:
                        st.error(f"Error generating cover letter or PDF: {e}")

    with col3:
        if st.button("Generate Portfolio", use_container_width=True, type="primary"):
            if not name or not skills:
                st.warning("Please provide at least your Name and Skills.")
            else:
                with st.spinner("Generating Portfolio Summary..."):
                    try:
                        port_output = generate_portfolio_summary(user_data)
                        st.success("Portfolio Summary Generated Successfully!")
                        st.text_area("Generated Portfolio Summary", port_output, height=200)
                        
                        # PDF Generation
                        safe_name = name.replace(" ", "_").lower() if name else "user"
                        pdf_path = f"portfolio_{safe_name}.pdf"
                        
                        create_portfolio_pdf(port_output, pdf_path)
                        
                        with open(pdf_path, "rb") as pdf_file:
                            st.download_button(
                                label="Download Portfolio as PDF",
                                data=pdf_file,
                                file_name=pdf_path,
                                mime="application/pdf",
                                use_container_width=True
                            )
                    except Exception as e:
                        st.error(f"Error generating portfolio summary or PDF: {e}")

    with col4:
        if st.button("Check Match Score", use_container_width=True, type="primary"):
            if not skills and not projects and not experience:
                st.warning("Please provide at least your Skills or Projects to calculate a match score.")
            elif not job_description:
                st.warning("Please provide a Job Description to compare against.")
            else:
                with st.spinner("Calculating ATS Match Score..."):
                    # Combine text for holistic matching
                    combined_user_text = f"{skills} {projects} {experience}"
                    
                    match_score = calculate_similarity(combined_user_text, job_description)
                    missing_keywords = generate_improvement_suggestions(combined_user_text, job_description)

                    st.subheader("ATS Match Analysis")
                    
                    if match_score >= 80:
                        st.success(f"Excellent Match! Score: {match_score}%")
                    elif match_score >= 50:
                        st.info(f"Good Match! Score: {match_score}%")
                    else:
                        st.warning(f"Needs Improvement. Score: {match_score}%")
                        
                    st.progress(match_score / 100)

                    if missing_keywords:
                        st.markdown("**Consider adding these keywords from the Job Description to improve your score:**")
                        for kw in missing_keywords:
                            st.markdown(f"- `{kw}`")
                    else:
                        st.success("Your profile covers all the major keywords found in the Job Description!")

if __name__ == "__main__":
    main()
