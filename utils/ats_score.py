"""
ats_score.py — ATS (Applicant Tracking System) score computation.

ATS Score Formula (0–100):
  ┌─────────────────────────────────┬────────┐
  │ Factor                          │ Weight │
  ├─────────────────────────────────┼────────┤
  │ TF-IDF text similarity          │  50%   │
  │ Skill overlap ratio             │  30%   │
  │ Resume completeness signals     │  20%   │
  └─────────────────────────────────┴────────┘

Completeness signals: presence of keywords like
'experience', 'education', 'project', 'skill', etc.
"""

import re


# Keywords that indicate a well-structured resume
_COMPLETENESS_KEYWORDS = [
    "experience", "education", "project", "skill", "certification",
    "achievement", "objective", "summary", "internship", "work",
    "university", "degree", "bachelor", "master", "gpa",
    "responsibility", "responsibility", "award", "publication",
]


def _compute_completeness(resume_text: str) -> float:
    """
    Score resume completeness by checking for structural section keywords.

    Returns:
        Float between 0.0 and 1.0
    """
    text_lower = resume_text.lower()
    found = sum(1 for kw in _COMPLETENESS_KEYWORDS if kw in text_lower)
    # Score scales up to 1.0 when ≥8 keywords are found
    return min(found / 8.0, 1.0)


def _compute_skill_overlap(matched: list, missing: list) -> float:
    """
    Skill overlap ratio = matched / (matched + missing).

    Returns:
        Float between 0.0 and 1.0
    """
    total = len(matched) + len(missing)
    if total == 0:
        return 0.0
    return round(len(matched) / total, 4)


def compute_ats_score(
    tfidf_score: float,
    matched_skills: list,
    missing_skills: list,
    resume_text: str,
) -> int:
    """
    Compute a weighted ATS compatibility score.

    Args:
        tfidf_score:    Raw TF-IDF cosine similarity (0.0 – 1.0)
        matched_skills: List of skills present in both resume and JD
        missing_skills: List of JD skills absent from resume
        resume_text:    Raw resume text for completeness check

    Returns:
        ATS score as integer 0 – 100
    """
    similarity_component = tfidf_score * 50          # Max 50 points
    skill_overlap_component = _compute_skill_overlap(matched_skills, missing_skills) * 30  # Max 30
    completeness_component = _compute_completeness(resume_text) * 20  # Max 20

    raw_score = similarity_component + skill_overlap_component + completeness_component
    return min(int(round(raw_score)), 100)


def get_ats_band(score: int) -> dict:
    """
    Categorise ATS score into a performance band with color and label.

    Args:
        score: Integer 0 – 100

    Returns:
        dict with 'band', 'color', 'emoji', 'advice'
    """
    if score >= 80:
        return {
            "band": "Excellent",
            "color": "#00d4aa",
            "emoji": "🏆",
            "advice": "Your resume is highly optimised for this role. Great work!",
        }
    elif score >= 60:
        return {
            "band": "Good",
            "color": "#4ecdc4",
            "emoji": "✅",
            "advice": "Solid match. A few targeted improvements can push you higher.",
        }
    elif score >= 40:
        return {
            "band": "Fair",
            "color": "#ffe66d",
            "emoji": "⚠️",
            "advice": "Moderate match. Consider adding missing skills and keywords.",
        }
    else:
        return {
            "band": "Needs Work",
            "color": "#ff6b6b",
            "emoji": "🔧",
            "advice": "Low match. Tailor your resume significantly to this job description.",
        }
