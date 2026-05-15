"""
preprocess.py — NLP preprocessing pipeline.

Pipeline:
  1. Lowercase conversion
  2. Punctuation & special character removal
  3. Tokenization (spaCy)
  4. Stopword removal (NLTK + spaCy)
  5. Lemmatization (spaCy)
"""

import re
import spacy
import nltk
from nltk.corpus import stopwords

# Download required NLTK data (silent)
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

# Load spaCy English model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise RuntimeError(
        "spaCy model 'en_core_web_sm' not found. "
        "Run: python -m spacy download en_core_web_sm"
    )

# Build combined stopword set
NLTK_STOPWORDS = set(stopwords.words("english"))
SPACY_STOPWORDS = nlp.Defaults.stop_words
ALL_STOPWORDS = NLTK_STOPWORDS | SPACY_STOPWORDS


def clean_text(text: str) -> str:
    """Lowercase and remove punctuation / special characters."""
    text = text.lower()
    # Keep letters, digits, spaces, and dots (for e.g. Node.js)
    text = re.sub(r"[^a-z0-9\s\.\+#]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_text(text: str) -> dict:
    """
    Full NLP preprocessing pipeline.

    Args:
        text: Raw input text

    Returns:
        dict with keys:
            'cleaned'     — cleaned raw string
            'tokens'      — all tokens after cleaning
            'filtered'    — tokens after stopword removal
            'lemmas'      — lemmatized tokens (stopwords removed)
            'processed'   — final joined string for TF-IDF
    """
    cleaned = clean_text(text)
    doc = nlp(cleaned)

    tokens = [token.text for token in doc if token.text.strip()]
    filtered = [
        token.text for token in doc
        if token.text.strip() and token.text not in ALL_STOPWORDS
    ]
    lemmas = [
        token.lemma_ for token in doc
        if token.text.strip()
        and token.text not in ALL_STOPWORDS
        and not token.is_punct
        and len(token.text) > 1
    ]

    return {
        "cleaned": cleaned,
        "tokens": tokens,
        "filtered": filtered,
        "lemmas": lemmas,
        "processed": " ".join(lemmas),
    }


def get_top_keywords(lemmas: list, n: int = 20) -> list:
    """
    Return top-N most frequent tokens (excluding very short ones).

    Args:
        lemmas: List of lemmatized tokens
        n: Number of top keywords to return

    Returns:
        List of (word, count) tuples
    """
    from collections import Counter
    counts = Counter(
        w for w in lemmas if len(w) > 2
    )
    return counts.most_common(n)
