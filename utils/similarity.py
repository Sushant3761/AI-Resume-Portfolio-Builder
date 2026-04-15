import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Normalization mapping for technical keywords vulnerable to regex stripping
TECH_MAPPING = {
    'c++': 'cpp',
    'c#': 'csharp',
    'node.js': 'nodejs',
    '.net': 'dotnet',
    'vue.js': 'vuejs',
    'react.js': 'reactjs',
    'f#': 'fsharp',
    'g++': 'gpp'
}

def _clean_text(text: str) -> str:
    """Helper function to cleanly normalize text while protecting tech buzzwords."""
    if not text:
        return ""
        
    text = text.lower()
    
    # Pre-emptively normalize vulnerable tech keywords
    for keyword, normalized in TECH_MAPPING.items():
        # Using word boundary doesn't work well with . or + symbols natively in standard python re without escaping
        # simple global replace is usually fine for these specific strings given their unique signatures
        text = text.replace(keyword, normalized)
    
    # Regex allowed characters: alphanumeric, spaces, and we explicitly keep common tech symbols just in case
    # #, +, . 
    # Actually, replacing them with spaces is fine NOW because we already mapped c++ to cpp!
    # So we can safely strip the remaining puntuation
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # Remove extra spaces
    return re.sub(r'\s+', ' ', text).strip()

def calculate_similarity(user_text: str, job_description: str) -> float:
    """
    Calculates the match percentage between user text and job description
    using TF-IDF Vectorization and Cosine Similarity.
    """
    if not user_text.strip() or not job_description.strip():
        return 0.0
        
    cleaned_user_text = _clean_text(user_text)
    cleaned_jd = _clean_text(job_description)
    
    vectorizer = TfidfVectorizer(stop_words='english')
    
    try:
        tfidf_matrix = vectorizer.fit_transform([cleaned_user_text, cleaned_jd])
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return round(cosine_sim[0][0] * 100, 2)
    except ValueError:
        return 0.0

def generate_improvement_suggestions(user_text: str, job_description: str) -> list:
    """
    Identifies important keywords missing from the user's text but present in the JD.
    """
    if not job_description.strip():
        return []
        
    cleaned_user_text = _clean_text(user_text)
    cleaned_jd = _clean_text(job_description)
    
    vectorizer = TfidfVectorizer(stop_words='english', max_features=50)
    
    try:
        vectorizer.fit([cleaned_jd])
        jd_keywords = set(vectorizer.get_feature_names_out())
        user_words = set(cleaned_user_text.split())
        
        missing_keywords = list(jd_keywords - user_words)
        missing_keywords.sort()
        
        # Optionally, we reverse the mapping for missing keywords so they look nice to the user (e.g. cpp back to c++)
        reverse_mapping = {v: k for k, v in TECH_MAPPING.items()}
        for i, word in enumerate(missing_keywords):
            if word in reverse_mapping:
                missing_keywords[i] = reverse_mapping[word]
                
        return missing_keywords
    except ValueError:
        return []
