"""Local creative direction: account style + safe LUT candidate scoring.

Vision describes the content; this module makes the local, deterministic style
decision.  It never lets a remote model name an arbitrary LUT file.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config.paths import find_project_root, system_dir
from . import lut_engine


DEFAULT_ACCOUNT_STYLE = {
    "contrast": "medium",
    "saturation": "controlled",
    "skin_tone": "natural",
    "warmth": "slightly_warm",
    "cinematic_strength": 0.42,
    "preferred_variant": "A",
    "preferred_crop": "environment_preserving",
    "text_style": "minimal",
    "feedback_count": 0,
}


def account_style_path() -> Path:
    root = find_project_root(Path(__file__).resolve().parents[3])
    return system_dir(root) / "config" / "account_style_profile.json"


def load_account_style() -> Dict[str, Any]:
    """Load the account's creative defaults; invalid fields use safe defaults."""
    style = dict(DEFAULT_ACCOUNT_STYLE)
    path = account_style_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        if isinstance(data, dict):
            style.update(data)
    except (OSError, json.JSONDecodeError):
        pass

    try:
        style["cinematic_strength"] = max(0.0, min(1.0, float(style["cinematic_strength"])))
    except (TypeError, ValueError):
        style["cinematic_strength"] = DEFAULT_ACCOUNT_STYLE["cinematic_strength"]
    return style


def build_style_intent(scene_plan: Optional[Dict[str, Any]], account_style=None) -> Dict[str, Any]:
    """Convert content semantics to a local style contract.

    The remote vision result remains limited to scene/mood/subject facts.  The
    local profile and rules map those facts to style family, constraints, and
    a safe LUT strength.
    """
    account = account_style or load_account_style()
    scene = scene_plan or {}
    scene_type = str(scene.get("scene_type", "general")).lower()
    subject = str(scene.get("main_subject", "")).lower()
    text = f"{scene_type} {subject}"

    family = "documentary"
    preserve_skin = any(word in text for word in ("person", "portrait", "woman", "man", "wedding", "face"))
    preserve_sky = scene_type in ("sunset", "landscape") or any(word in text for word in ("beach", "ocean", "sky", "sunset"))

    if any(word in text for word in ("beach", "ocean", "pool", "sunset", "summer", "florida")):
        family = "warm_travel"
    elif scene_type == "night" or any(word in text for word in ("party", "club", "concert", "neon")):
        family = "night_cinematic"
    elif preserve_skin:
        family = "editorial_portrait"
    elif any(word in text for word in ("street", "city", "architecture", "travel")):
        family = "documentary_travel"
    elif any(word in text for word in ("nature", "forest", "mountain", "landscape")):
        family = "nature_rich"

    return {
        "family": family,
        "preserve_skin": preserve_skin,
        "preserve_sky": preserve_sky,
        "contrast_preference": account["contrast"],
        "saturation_preference": account["saturation"],
        "warmth_preference": account["warmth"],
        "lut_strength": account["cinematic_strength"],
        "preferred_variant": account["preferred_variant"],
        "crop_preference": account["preferred_crop"],
    }


def _name_score(name: str, style: Dict[str, Any]) -> float:
    name = name.lower()
    family = style["family"]
    score = 0.0
    if family == "warm_travel":
        score += 1.0 if any(k in name for k in ("velvia_srgb", "punch", "astia_srgb")) else 0.0
    elif family == "editorial_portrait":
        score += 1.0 if any(k in name for k in ("pro neg hi_srgb", "classic chrome_srgb", "fashion")) else 0.0
    elif family == "night_cinematic":
        score += 1.0 if any(k in name for k in ("eterna_srgb", "bleach bypass_srgb", "cinematic")) else 0.0
    elif family == "documentary_travel":
        score += 1.0 if any(k in name for k in ("classic neg_srgb", "nostalgic neg_srgb", "classic chrome_srgb")) else 0.0
    elif family == "nature_rich":
        score += 1.0 if any(k in name for k in ("velvia_srgb", "astia_srgb", "provia_srgb")) else 0.0
    else:
        score += 1.0 if "classic chrome_srgb" in name else 0.0

    # Avoid non-sRGB LUTs when equivalent sRGB LUT exists. Never auto-use a
    # technical conversion LUT for normal iPhone/Rec.709 input.
    if "displayp3" in name or "conversion" in name or "gyroflow" in name:
        score -= 2.0
    if style["preserve_skin"] and any(k in name for k in ("bleach", "cyanotype", "sepia")):
        score -= 1.0
    return score


def rank_lut_candidates(style_intent: Dict[str, Any], limit: int = 3) -> List[Dict[str, Any]]:
    """Return locally scored, compatible LUT candidates without applying them."""
    candidates = []
    for path in lut_engine.list_luts():
        score = _name_score(path.stem, style_intent)
        candidates.append({"path": path, "name": path.name, "score": round(score, 3)})
    candidates.sort(key=lambda item: (item["score"], item["name"]), reverse=True)
    return candidates[:max(1, int(limit))]


def choose_lut(style_intent: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Choose the best local LUT candidate, only when its local score is positive."""
    candidates = rank_lut_candidates(style_intent, limit=3)
    if not candidates or candidates[0]["score"] <= 0:
        return None
    selected = dict(candidates[0])
    selected["strength"] = style_intent["lut_strength"]
    selected["candidates"] = [{"name": c["name"], "score": c["score"]} for c in candidates]
    return selected


def blend_lut(original, transformed, strength: float):
    """Blend a LUT look with the graded base at a bounded style strength."""
    strength = max(0.0, min(1.0, float(strength)))
    return original * (1.0 - strength) + transformed * strength
