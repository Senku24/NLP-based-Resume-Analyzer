"""
suggestions.py — Rule-based recommendation engine.

Generates actionable improvement suggestions based on:
  - Missing skills from the JD
  - Overall match score band
  - Keyword density observations
"""


# Score-band messaging
_SCORE_MESSAGES = {
    "high": (
        "Your resume shows a strong alignment with this job description. "
        "Focus on quantifying achievements and polishing language."
    ),
    "medium": (
        "Good foundation — bridge the skill gaps below to significantly "
        "improve your match score."
    ),
    "low": (
        "Your resume needs significant tailoring for this role. "
        "Start by incorporating the missing technical skills listed below."
    ),
}

# Tips based on common resume weaknesses
_GENERAL_TIPS = [
    "Quantify your achievements (e.g., 'Improved performance by 30%') to stand out.",
    "Use industry-standard section headings: Summary, Experience, Education, Skills.",
    "Tailor your professional summary to mirror the job description language.",
    "Include a dedicated Technical Skills section for ATS parsing accuracy.",
    "Use bullet points to describe responsibilities — avoid dense paragraphs.",
    "Add measurable project outcomes in your Projects section.",
    "Ensure your contact information is up to date and professional.",
    "Include relevant certifications (AWS, Google, Coursera, etc.) if applicable.",
]


def generate_suggestions(
    missing_skills: list,
    match_pct: int,
    resume_text: str = "",
) -> dict:
    """
    Generate structured improvement suggestions.

    Args:
        missing_skills: Skills found in JD but absent from resume
        match_pct:      Overall match percentage (0–100)
        resume_text:    Raw resume text for additional heuristics

    Returns:
        dict with:
            'score_message'   — band-specific opening message
            'skill_tips'      — list of skill-specific suggestions
            'general_tips'    — list of general resume improvement tips
            'priority'        — 'High' / 'Medium' / 'Low'
    """
    # Determine score band
    if match_pct >= 70:
        band = "high"
        priority = "Low"
    elif match_pct >= 40:
        band = "medium"
        priority = "Medium"
    else:
        band = "low"
        priority = "High"

    # Build skill-specific suggestions
    skill_tips = []
    for skill in missing_skills[:10]:   # Cap at top 10 to keep it readable
        skill_tips.append(
            f"Add **{skill}** to your Skills section or demonstrate it in project descriptions."
        )

    # Extra heuristics on raw resume text
    extra_tips = []
    text_lower = resume_text.lower() if resume_text else ""

    if "github" not in text_lower and "portfolio" not in text_lower:
        extra_tips.append(
            "Include a link to your **GitHub profile** or portfolio to showcase real projects."
        )
    if "internship" not in text_lower and "experience" not in text_lower:
        extra_tips.append(
            "Add an **Experience** section, even if it is internship or freelance work."
        )
    if len(resume_text.split()) < 200:
        extra_tips.append(
            "Your resume appears short. Aim for 400–700 words to provide enough context for ATS systems."
        )

    # Pick 3–4 general tips based on band
    if band == "low":
        general_tips = _GENERAL_TIPS[:5] + extra_tips
    elif band == "medium":
        general_tips = _GENERAL_TIPS[:3] + extra_tips
    else:
        general_tips = _GENERAL_TIPS[:2] + extra_tips

    return {
        "score_message": _SCORE_MESSAGES[band],
        "skill_tips": skill_tips,
        "general_tips": general_tips,
        "priority": priority,
    }
