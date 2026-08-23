"""Grading constants - all tunable parameters in one place."""


class GradingConstants:
    """All magic numbers extracted to named constants."""

    # ========== Exposure Normalization ==========
    EXPOSURE_TARGET_LUMINANCE = 0.5
    EXPOSURE_MIN_GAIN = 0.7
    EXPOSURE_MAX_GAIN = 1.4

    # ========== White Balance ==========
    WB_GAIN_MIN = 0.85
    WB_GAIN_MAX = 1.15

    # ========== Saturation Processing ==========
    SATURATION_DIVISOR = 22.0

    # ========== Highlight and Shadow Clipping ==========
    HIGHLIGHT_THRESHOLD = 0.97
    HIGHLIGHT_CLIP_THRESHOLD = 0.995
    SHADOW_CLIP_THRESHOLD = 0.005
    HIGHLIGHT_ROLLOFF_FACTOR = 0.45

    # ========== Scene Detection Thresholds (HSV) ==========
    SUNSET_WARM_INDEX = 0.12
    SUNSET_VALUE_MIN = 0.55
    NIGHT_VALUE_MAX = 0.28
    SATURATED_SAT_MIN = 110
    MUTED_SAT_MAX = 55

    # ========== Microcontrast Adjustment ==========
    MICROCONTRAST_FACTOR = 1.05

    # ========== Natural Grading (Variant A) ==========
    NATURAL_CONTRAST_BASE = 3
    NATURAL_SAT_BASE = 1

    # ========== Cinematic Grading (Variant B) - ENHANCED ==========
    CINEMATIC_CONTRAST_BASE = 6
    CINEMATIC_SAT_BASE = 4
    CINEMATIC_TEAL_ORANGE_ENHANCED = 2.2  # 110% stronger than original 1.04x

    # ========== Vision API ==========
    VISION_API_TIMEOUT = 120
    VISION_PREVIEW_MAX_SIZE = 1280
    VISION_PREVIEW_QUALITY = 92

    # ========== QA Thresholds ==========
    QA_HIGHLIGHT_CLIP_MAX_PCT = 5.0
    QA_RATIO_TOLERANCE = 0.02
    # Original-preserving color QA. These are deliberately conservative: a
    # creative look may change the tonal distribution, but it must not cause
    # an obvious technical or subject-level deterioration.
    QA_NATURAL_LUMA_DRIFT = 0.08
    QA_CREATIVE_LUMA_DRIFT = 0.12
    QA_MEDIAN_LUMA_DRIFT_MULTIPLIER = 1.25
    QA_SHADOW_DRIFT_MAX = 0.10
    QA_SATURATION_DRIFT_MAX = 0.18
    QA_NATURAL_SATURATION_DRIFT_MAX = 0.10
    QA_COLOR_CAST_DRIFT_MAX = 0.08
    QA_NATURAL_COLOR_CAST_DRIFT_MAX = 0.05
    QA_CLIP_DRIFT_MAX = 0.03
    QA_MIN_NATURAL_SCORE = 75.0
    QA_MIN_CREATIVE_SCORE = 65.0
    QA_MAX_RECOVERY_STEPS = 5
    LUT_MAX_INITIAL_STRENGTH = 0.55

    # ========== Video Processing ==========
    VIDEO_CROP_WIDTH = 1080
    VIDEO_CROP_HEIGHT = 1920
    VIDEO_FFMPEG_CRF = 18
    VIDEO_FFMPEG_PRESET = "fast"
    VIDEO_FFMPEG_BITRATE_AUDIO = "192k"
    VIDEO_FFMPEG_TIMEOUT = 600

    # ========== Color Preservation (HSV Ranges) ==========
    SKIN_TONE_HUE_MIN = 0
    SKIN_TONE_HUE_MAX = 20
    SKIN_TONE_SAT_MIN = 100
    SKIN_TONE_SAT_MAX = 255


# Quick reference
__all__ = ["GradingConstants"]
