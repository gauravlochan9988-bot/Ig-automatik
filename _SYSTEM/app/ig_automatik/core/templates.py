"""Editing and style templates inspired by modern short-form video pacing."""

from typing import Dict, Any, Optional

DEFAULT_TEMPLATES = {
    "miami_vibes": {
        "id": "miami_vibes",
        "name": "Miami Summer Vibe",
        "description": "Vibrant, sun-kissed look with warm highlights and smooth crossfade transitions.",
        "pacing": "medium",
        "transitions": "xfade",
        "transition_type": "fade",
        "transition_duration": 0.5,
        "ken_burns": True,
        "zoom_speed": "0.0015",
        "grading": {
            "warmth": 0.35,
            "contrast": 1.2,
            "saturation": 1.3,
        },
        "tags": ["beach", "summer", "sunset", "sun", "vacation", "florida", "ocean", "pool", "boat"],
    },
    "cinematic_travel": {
        "id": "cinematic_travel",
        "name": "Cinematic Travel",
        "description": "Epic filmic pacing with subtle push-in zoom and smooth dissolves.",
        "pacing": "cinematic",
        "transitions": "xfade",
        "transition_type": "dissolve",
        "transition_duration": 0.6,
        "ken_burns": True,
        "zoom_speed": "0.001",
        "grading": {
            "warmth": 0.1,
            "contrast": 1.3,
            "saturation": 1.15,
        },
        "tags": ["travel", "landscape", "nature", "city", "architecture", "mountains", "street"],
    },
    "moody_night": {
        "id": "moody_night",
        "name": "Moody Night / Party",
        "description": "High contrast, rich shadows and punchy wipe cuts for nightlife & events.",
        "pacing": "dynamic",
        "transitions": "xfade",
        "transition_type": "wipeleft",
        "transition_duration": 0.4,
        "ken_burns": False,
        "zoom_speed": "0.0",
        "grading": {
            "warmth": -0.1,
            "contrast": 1.4,
            "saturation": 1.25,
        },
        "tags": ["night", "party", "club", "bar", "concert", "lights", "neon", "dark"],
    },
    "clean_creator": {
        "id": "clean_creator",
        "name": "Clean Creator / Vlog",
        "description": "Natural skin tones, true-to-life colors and crisp cuts for lifestyle.",
        "pacing": "snappy",
        "transitions": "none",
        "transition_type": "fade",
        "transition_duration": 0.3,
        "ken_burns": False,
        "zoom_speed": "0.0",
        "grading": {
            "warmth": 0.05,
            "contrast": 1.1,
            "saturation": 1.1,
        },
        "tags": ["portrait", "person", "food", "lifestyle", "vlog", "general"],
    },
}


def list_templates() -> Dict[str, Dict[str, Any]]:
    """Return all available templates."""
    return DEFAULT_TEMPLATES


def get_template(template_id: str) -> Dict[str, Any]:
    """Retrieve template by id, falling back to clean_creator."""
    return DEFAULT_TEMPLATES.get(template_id, DEFAULT_TEMPLATES["clean_creator"])


def select_template_for_scene(scene_plan: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Match the best template based on scene analysis tags and scene_type."""
    if not scene_plan:
        return DEFAULT_TEMPLATES["miami_vibes"]

    scene_type = str(scene_plan.get("scene_type", "")).lower()
    main_subject = str(scene_plan.get("main_subject", "")).lower()
    text = f"{scene_type} {main_subject}"

    for tpl_id, tpl in DEFAULT_TEMPLATES.items():
        for tag in tpl["tags"]:
            if tag in text:
                return tpl

    if scene_type in ("sunset",):
        return DEFAULT_TEMPLATES["miami_vibes"]
    if scene_type in ("night",):
        return DEFAULT_TEMPLATES["moody_night"]

    return DEFAULT_TEMPLATES["clean_creator"]
