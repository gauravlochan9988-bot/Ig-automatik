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
from .media import Media, Photo, Video
from .pipeline import Pipeline, BatchProcessor
from . import vision
from .video_tools import (
    probe_duration,
    detect_scenes,
    extract_segment_frame,
    concat_segments,
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
    "Photo",
    "Video",
    "Pipeline",
    "BatchProcessor",
    "vision",
    "probe_duration",
    "detect_scenes",
    "extract_segment_frame",
    "concat_segments",
]
