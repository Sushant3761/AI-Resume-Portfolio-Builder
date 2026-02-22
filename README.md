# AI Resume & Portfolio Builder

## 1. Project Overview
The AI Resume & Portfolio Builder is a locally run web application that leverages Large Language Models (LLMs) and Natural Language Processing (NLP) techniques to help users generate professional career materials. Built with Streamlit, the application enables users to input their educational and professional backgrounds to generate ATS-friendly resumes, targeted cover letters, and portfolio summaries, alongside an ATS match scoring system.

## 2. Problem Statement
Job seekers often struggle to align their existing backgrounds with specific job descriptions, suffering from poor Applicant Tracking System (ATS) parsing and ineffective formatting. Furthermore, manually tailoring resumes and cover letters for every application is a time-consuming and generic process that often fails to highlight the technical alignment required for engineering roles.

## 3. Solution Description
This project provides a pipeline to automate the creation of strictly formatted, constraint-driven document generations using the OpenRouter API. It ingests user data and target job descriptions, applies TF-IDF and Cosine Similarity to evaluate keyword match alignment, and then generates professional, length-constrained PDF documents designed to pass ATS screenings and appeal to technical recruiters.

## 4. Core Features
- **Resume Generation**: Outputs a stricly one-page, recruiter-ready resume via LLM API.
- **Cover Letter Generation**: Creates a targeted, concise cover letter focusing on technical impact.
- **Portfolio Summary Generation**: Generates a professional introduction for websites or LinkedIn.
- **ATS Match Score Calculation**: Calculates keyword similarity between user background and job descriptions.
- **Missing Keyword Detection**: Identifies critical missing terms from the user's profile.
- **PDF Export**: Converts the generated text sequences into downloadable PDF formats (Resume, Cover Letter, Portfolio).

## 5. AI Integration Details
- **API Provider**: OpenRouter
- **Endpoint**: `https://openrouter.ai/api/v1/chat/completions`
- **Model**: `mistralai/mistral-7b-instruct`
- **Parameters**: `max_tokens=450`, `temperature=0.7`
- **Prompt Engineering**: The system utilizes rigid prompt templates commanding specific strict token limits, exact paragraph counts, and negative constraints (e.g., removing fluff words). 
- **Safety Safeguards**: Word limit rewrite commands are injected to enforce length restrictions.
- **Error Handling**: Implements explicit HTTP 429 (Rate Limit) logic and 30-second timeouts. API keys are safely loaded via `.env`.

## 6. Machine Learning Implementation
- **Algorithms**: 
  - **TF-IDF Vectorization**: Parses user and job description strings to evaluate term relevance.
  - **Cosine Similarity**: Calculates the angular distance between the generated TF-IDF matrices to return a match percentage.
- **Text Processing**: Employs Regex preprocessing for punctuation stripping and whitespace normalization, alongside set-based logic for keyword comparisons.
- **Limitation**: The vectorizer limits semantic comparison strictly to `max_features=50` broadly.

## 7. Technology Stack
- **Python**: Core backend logic.
- **Streamlit**: Web interface framework.
- **Scikit-learn**: Vectorization and similarity computation.
- **ReportLab**: PDF document generation.
- **Requests**: HTTP interactions with OpenRouter API.
- **python-dotenv**: Local environment configuration.

## 8. Project Structure
```text
Resume&Portfolio Builder/
├── .env                  # Environment variables (API Keys)
├── app.py                # Main Streamlit application frontend
├── requirements.txt      # Python dependencies
└── utils/                # Backend logic modules
    ├── pdf_generator.py      # PDF building mechanisms
    ├── resume_generator.py   # LLM API connection and prompt engineering
    └── similarity.py         # TF-IDF and NLP metric logic
```

## 9. Setup Instructions
1. Ensure Python 3.8+ is installed.
2. Clone or extract the project repository.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file in the root directory and add your API key:
   ```env
   OPENROUTER_API_KEY="your_api_key_here"
   ```
5. Run the application:
   ```bash
   streamlit run app.py
   ```

## 10. Limitations
- **No Database Persistence**: All user inputs format per session state; reloading the page clears data.
- **No File Uploads**: Users cannot copy logic from an existing PDF/Docx file; inputs must be pasted manually.
- **Sequential Execution**: Lacks asynchronous API generation limits generation to single-thread UI locks.
- **Lexical Limitations**: The ATS scorer is strictly lexical (exact keyword matching capped at 50 features) and does not utilize contextual embeddings for semantic meaning.

## 11. Conclusion
The AI Resume & Portfolio Builder successfully demonstrates the practical integration of LLMs with traditional NLP metrics (TF-IDF) inside a localized Streamlit environment. By actively restricting the AI via rigid prompt engineering, the application guarantees concise, professional outputs optimized for the modern job application pipeline without requiring complex cloud deployment architectures.
