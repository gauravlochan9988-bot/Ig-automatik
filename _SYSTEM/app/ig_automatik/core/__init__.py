"""Core processing modules for IG-AUTOMATIK."""

from .grading_engine import (
    process_photo,
    process_reel,
    run_on_folder,
    load_rgb,
    analyze_scene,
    grade_variant_a,
    grade_variant_b,
)
from .media import Media
from .pipeline import Pipeline
from . import vision
from . import templates
from . import lut_engine
from .video_tools import (
    probe_duration,
    detect_scenes,
    extract_segment_frame,
)

__all__ = [
    "process_photo",
    "process_reel",
    "run_on_folder",
    "load_rgb",
    "analyze_scene",
    "grade_variant_a",
    "grade_variant_b",
    "Media",
    "Pipeline",
    "vision",
    "probe_duration",
    "detect_scenes",
    "extract_segment_frame",
]