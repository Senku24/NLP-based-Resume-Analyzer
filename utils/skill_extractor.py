"""
skill_extractor.py — Skill extraction via predefined skill list matching.

Method:
  - Load skills from data/skills.csv
  - Case-insensitive phrase match against resume/JD text
  - Return matched, missing, and common skills
"""

import os
import re
import pandas as pd
from pathlib import Path


# Resolve skills.csv relative to project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SKILLS_PATH = _PROJECT_ROOT / "data" / "skills.csv"


def load_skills() -> pd.DataFrame:
    """Load the skill list from data/skills.csv."""
    if not _SKILLS_PATH.exists():
        raise FileNotFoundError(f"skills.csv not found at {_SKILLS_PATH}")
    return pd.read_csv(_SKILLS_PATH)


def extract_skills(text: str, skills_df: pd.DataFrame) -> dict:
    """
    Extract skills from text by matching against the skill list.

    Uses whole-word, case-insensitive matching to avoid false positives
    (e.g., 'R' matching inside other words).

    Args:
        text: Raw or preprocessed text
        skills_df: DataFrame with 'skill' and 'category' columns

    Returns:
        dict mapping skill_name → category for all matched skills
    """
    text_lower = text.lower()
    matched = {}

    for _, row in skills_df.iterrows():
        skill = row["skill"]
        category = row["category"]
        # Escape special regex chars (e.g. C++, Node.js)
        pattern = re.escape(skill.lower())
        # Word boundary — use lookaround to handle multi-word skills
        if re.search(r"(?<![a-zA-Z0-9])" + pattern + r"(?![a-zA-Z0-9])", text_lower):
            matched[skill] = category

    return matched


def get_skill_gap(resume_skills: dict, jd_skills: dict) -> dict:
    """
    Compare resume skills against JD skills to find matches and gaps.

    Args:
        resume_skills: {skill: category} from resume
        jd_skills:     {skill: category} from job description

    Returns:
        dict with:
            'matched'  — skills present in both resume and JD
            'missing'  — skills in JD but not in resume
            'extra'    — skills in resume not mentioned in JD
    """
    resume_set = set(resume_skills.keys())
    jd_set = set(jd_skills.keys())

    matched = resume_set & jd_set
    missing = jd_set - resume_set
    extra = resume_set - jd_set

    return {
        "matched": sorted(matched),
        "missing": sorted(missing),
        "extra": sorted(extra),
    }
