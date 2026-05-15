"""
similarity.py — TF-IDF + Cosine Similarity for resume-JD matching.

Library: scikit-learn
Method:
  1. Build TF-IDF matrix from [resume_text, jd_text]
  2. Compute cosine similarity between the two vectors
  3. Return match score as a float (0.0 – 1.0)
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def compute_tfidf_similarity(resume_text: str, jd_text: str) -> float:
    """
    Compute cosine similarity between resume and job description
    using TF-IDF vectorization.

    Args:
        resume_text: Preprocessed resume text
        jd_text:     Preprocessed job description text

    Returns:
        Similarity score between 0.0 and 1.0
    """
    if not resume_text.strip() or not jd_text.strip():
        return 0.0

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),   # Unigrams + bigrams for richer matching
        min_df=1,
        stop_words="english",
        sublinear_tf=True,    # Apply log normalization to TF
    )

    try:
        tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
        score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return round(float(score), 4)
    except Exception:
        return 0.0


def get_match_percentage(score: float) -> int:
    """Convert raw similarity score (0–1) to a percentage integer."""
    return int(round(score * 100))
