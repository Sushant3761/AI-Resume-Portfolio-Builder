import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def _clean_text(text: str) -> str:
    """Helper function to clean and normalize text before processing."""
    if not text:
        return ""
        
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation and special characters, keeping only alphanumeric and spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def calculate_similarity(user_text: str, job_description: str) -> float:
    """
    Calculates the match percentage between user text and job description
    using TF-IDF Vectorization and Cosine Similarity.
    
    Args:
        user_text (str): Combined text of user's skills, experience, projects, etc.
        job_description (str): Target job description text
        
    Returns:
        float: Match percentage rounded to 2 decimal places (0.0 to 100.0)
    """
    if not user_text.strip() or not job_description.strip():
        return 0.0
        
    cleaned_user_text = _clean_text(user_text)
    cleaned_jd = _clean_text(job_description)
    
    # Initialize TF-IDF Vectorizer with English stop words removed
    vectorizer = TfidfVectorizer(stop_words='english')
    
    try:
        # Fit and transform the texts into TF-IDF vectors
        tfidf_matrix = vectorizer.fit_transform([cleaned_user_text, cleaned_jd])
        
        # Calculate Cosine Similarity between the first vector (user) and second (JD)
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        
        # Convert to percentage and round
        match_percentage = round(cosine_sim[0][0] * 100, 2)
        return match_percentage
    except ValueError:
        # Happens if vocab is empty after removing stop words
        return 0.0

def generate_improvement_suggestions(user_text: str, job_description: str) -> list:
    """
    Identifies important keywords missing from the user's text but present in the JD.
    
    Args:
        user_text (str): Combined text of user's skills, experience, projects, etc.
        job_description (str): Target job description text
        
    Returns:
        list: List of missing keywords
    """
    if not job_description.strip():
        return []
        
    cleaned_user_text = _clean_text(user_text)
    cleaned_jd = _clean_text(job_description)
    
    # We use TF-IDF to find the most important words in the job description
    vectorizer = TfidfVectorizer(stop_words='english', max_features=50) # top 50 keywords broadly
    
    try:
        # Fit on the job description text only to get JD-specific important words
        vectorizer.fit([cleaned_jd])
        jd_keywords = set(vectorizer.get_feature_names_out())
        
        # Tokenize user text into words
        user_words = set(cleaned_user_text.split())
        
        # Find which important JD keywords are NOT in the user text
        missing_keywords = list(jd_keywords - user_words)
        
        # Sort alphabetically for better presentation
        missing_keywords.sort()
        return missing_keywords
    except ValueError:
        # Happens if vocab is empty
        return []
