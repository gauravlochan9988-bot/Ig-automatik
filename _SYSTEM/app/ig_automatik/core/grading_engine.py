"""IG-AUTOMATIK: AI Adaptive Professional Grading Engine (refactored)."""

import os
import json
import subprocess
import time
import functools
import hashlib
from pathlib import Path
from typing import Dict, Optional, Any
import numpy as np
import cv2
from PIL import Image

from ..config import Config, GradingConstants
from ..utils import get_logger, ExportManager
from .pipeline import Pipeline
from . import vision

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass

PHOTO_EXT = {".jpg", ".jpeg", ".png", ".gif", ".dng", ".tif", ".tiff", ".bmp", ".webp", ".heic", ".raw", ".nef", ".cr2", ".arw"}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".3gp"}


# ============================================================================
# Image Loading
# ============================================================================

def load_rgb(path):
    """Load image as float32 RGB (0..1)."""
    p = Path(path)
    if p.suffix.lower() in {".dng", ".nef", ".cr2", ".arw", ".tif", ".tiff", ".heic"}:
        with Image.open(path) as im:
            source = np.asarray(im)
            if np.issubdtype(source.dtype, np.integer) and source.dtype.itemsize > 1:
                if source.ndim == 2:
                    source = np.repeat(source[..., None], 3, axis=2)
                elif source.ndim == 3 and source.shape[2] == 1:
                    source = np.repeat(source, 3, axis=2)
                elif source.ndim != 3 or source.shape[2] < 3:
                    source = np.asarray(im.convert("RGB"))
                else:
                    source = source[..., :3]
            else:
                source = np.asarray(im.convert("RGB"))
            if np.issubdtype(source.dtype, np.integer):
                scale = float(np.iinfo(source.dtype).max)
            else:
                scale = 1.0 if source.max() <= 1.0 else float(source.max())
            arr = source.astype(np.float32) / scale
        return arr

    bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise ValueError(f"Cannot read image: {path}")

    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)

    if np.issubdtype(bgr.dtype, np.integer):
        # Preserve the source bit depth.  Dividing uint16 images by 255 would
        # produce values around 128 instead of the expected 0..1 range.
        scale = float(np.iinfo(bgr.dtype).max)
        bgr = bgr.astype(np.float32) / scale
    elif bgr.dtype != np.float32 or bgr.max() > 1.0:
        bgr = bgr.astype(np.float32)
        if bgr.max() > 1.0:
            bgr /= float(bgr.max())

    rgb = bgr[..., ::-1]
    if rgb.ndim == 3 and rgb.shape[2] == 4:
        rgb = rgb[..., :3]

    return rgb


# ============================================================================
# Technical Normalization
# ============================================================================

def normalize_technical(rgb):
    """Normalize exposure and white balance."""
    params = {}

    # Exposure normalization
    gray = cv2.cvtColor(
        (rgb * 255).astype(np.float32) if rgb.max() <= 1 else rgb,
        cv2.COLOR_RGB2GRAY,
    )
    lum = gray.mean() / 255.0
    exp = GradingConstants.EXPOSURE_TARGET_LUMINANCE / (lum + 1e-6)
    exp = np.clip(exp, GradingConstants.EXPOSURE_MIN_GAIN, GradingConstants.EXPOSURE_MAX_GAIN)
    out = np.clip(rgb * exp, 0, 1)
    params["exposure_gain"] = float(exp)

    # White balance
    mean_r = rgb[:, :, 0].mean()
    mean_g = rgb[:, :, 1].mean()
    mean_b = rgb[:, :, 2].mean()
    g = (mean_r + mean_b) / 2.0

    if mean_g > 0:
        gain_r = g / (mean_r + 1e-6)
        gain_b = g / (mean_b + 1e-6)
        gain_g = 1.0
    else:
        gain_r = gain_g = gain_b = 1.0

    gains = np.clip([gain_r, gain_g, gain_b], GradingConstants.WB_GAIN_MIN, GradingConstants.WB_GAIN_MAX)
    gains = gains / np.mean(gains)
    out = np.clip(out * gains.reshape(1, 1, 3), 0, 1)
    out = np.clip((out - 0.5) * GradingConstants.MICROCONTRAST_FACTOR + 0.5, 0, 1)
    params["wb_gains"] = [round(g, 3) for g in gains]

    return out, params


# ============================================================================
# Enhanced Scene Analysis
# ============================================================================

def analyze_scene(rgb):
    """Analyze scene with improved detection."""
    scene = {"tags": [], "cinematic": {}, "natural": {}}

    # HSV analysis
    hsv = cv2.cvtColor(np.clip(rgb * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    sat = float(hsv[:, :, 1].mean())
    val = float(hsv[:, :, 2].mean()) / 255.0

    # Color warmth
    mean_r = float(rgb[:, :, 0].mean())
    mean_b = float(rgb[:, :, 2].mean())
    warm_index = (mean_r - mean_b) / (mean_r + mean_b + 1e-6)

    # Scene detection
    tags = []
    if warm_index > GradingConstants.SUNSET_WARM_INDEX and val > GradingConstants.SUNSET_VALUE_MIN:
        tags.append("sunset_warm")
    elif val < GradingConstants.NIGHT_VALUE_MAX:
        tags.append("night")

    if sat > GradingConstants.SATURATED_SAT_MIN:
        tags.append("saturated")
    if sat < GradingConstants.MUTED_SAT_MAX:
        tags.append("muted")

    # Check for product/food (low saturation, even lighting)
    std_r = float(rgb[:, :, 0].std())
    std_b = float(rgb[:, :, 2].std())
    if sat < 80 and (std_r + std_b) / 2 < 0.15 and val > 0.4:
        tags.append("product")

    scene["tags"] = tags if tags else ["general"]
    scene["value"] = val
    scene["warm_index"] = warm_index
    scene["saturation"] = sat

    # Adaptive parameters
    if "night" in tags:
        scene["cinematic"] = {"shadow": -12, "highlights": -6, "contrast": 10, "sat": 6, "warm_mult": 2.0}
        scene["natural"] = {"shadow": 6, "highlights": -2, "contrast": 5, "sat": 3}
    elif "sunset_warm" in tags:
        scene["cinematic"] = {"contrast": 9, "sat": 5, "warm_mult": 2.2, "cool_shadows": 0.85}
        scene["natural"] = {"contrast": 4, "sat": 2}
    elif "muted" in tags:
        scene["cinematic"] = {"contrast": 8, "sat": 7, "warm_mult": 1.8}
        scene["natural"] = {"contrast": 6, "sat": 4}
    elif "product" in tags:
        scene["cinematic"] = {"contrast": 7, "sat": 4, "warm_mult": 1.5}
        scene["natural"] = {"contrast": 4, "sat": 2}
    else:  # general / saturated
        scene["cinematic"] = {"contrast": 8, "sat": 5, "warm_mult": 2.0}
        scene["natural"] = {"contrast": 4, "sat": 2}

    return scene


# ============================================================================
# Grading Functions (Enhanced)
# ============================================================================

def _hsv_float_sat(rgb_float, sat_gain):
    """Saturation adjustment without uint8 conversion."""
    rgb = np.clip(rgb_float.astype(np.float32), 0.0, 1.0)
    maximum = rgb.max(axis=2)
    minimum = rgb.min(axis=2)
    delta = maximum - minimum
    value = maximum
    saturation = np.divide(delta, maximum, out=np.zeros_like(delta), where=maximum > 1e-8)

    hue = np.zeros_like(maximum)
    nonzero = delta > 1e-8
    red = (maximum == rgb[..., 0]) & nonzero
    green = (maximum == rgb[..., 1]) & nonzero
    blue = (maximum == rgb[..., 2]) & nonzero
    hue[red] = ((rgb[..., 1][red] - rgb[..., 2][red]) / delta[red]) % 6.0
    hue[green] = (rgb[..., 2][green] - rgb[..., 0][green]) / delta[green] + 2.0
    hue[blue] = (rgb[..., 0][blue] - rgb[..., 1][blue]) / delta[blue] + 4.0
    hue /= 6.0

    saturation = np.clip(saturation * float(sat_gain), 0.0, 1.0)
    sector = np.floor(hue * 6.0).astype(np.int32)
    fraction = hue * 6.0 - sector
    p = value * (1.0 - saturation)
    q = value * (1.0 - saturation * fraction)
    t = value * (1.0 - saturation * (1.0 - fraction))
    choices = np.stack(
        [
            np.stack([value, t, p], axis=2),
            np.stack([q, value, p], axis=2),
            np.stack([p, value, t], axis=2),
            np.stack([p, q, value], axis=2),
            np.stack([t, p, value], axis=2),
            np.stack([value, p, q], axis=2),
        ],
        axis=0,
    )
    result = choices[sector % 6, np.arange(rgb.shape[0])[:, None], np.arange(rgb.shape[1])[None, :]]
    return np.clip(result.astype(np.float32), 0.0, 1.0)


def _adaptive_sat_gain(scene, base_sat):
    """Calculate adaptive saturation based on image complexity."""
    color_complexity = scene.get("saturation", 128) / 255.0
    if color_complexity > 150 / 255.0:
        return base_sat * 0.85
    return base_sat


def _protect_highlights(rgb, threshold=None, rolloff_factor=None):
    """Compress only the upper tonal range while leaving midtones intact."""
    threshold = threshold if threshold is not None else GradingConstants.HIGHLIGHT_THRESHOLD
    rolloff = rolloff_factor if rolloff_factor is not None else GradingConstants.HIGHLIGHT_ROLLOFF_FACTOR
    excess = np.clip(rgb - threshold, 0.0, 1.0)
    # A smooth rolloff avoids a visible hard boundary around the threshold.
    compressed = threshold + excess * rolloff
    return np.where(rgb > threshold, compressed, rgb).astype(np.float32)


def grade_variant_a(base, scene=None, intent=None):
    """Premium Natural grading (Variant A)."""
    scene = scene or analyze_scene(base)
    out = base.astype(np.float32)

    # Get intent or use scene defaults
    if intent:
        contrast = float(intent.get("contrast", 0)) / 2.0 + GradingConstants.NATURAL_CONTRAST_BASE
        sat = float(intent.get("saturation", 0)) / 2.0 + GradingConstants.NATURAL_SAT_BASE
    else:
        p = scene.get("natural", {})
        contrast = p.get("contrast", GradingConstants.NATURAL_CONTRAST_BASE)
        sat = p.get("sat", GradingConstants.NATURAL_SAT_BASE)

    # Contrast
    contrast_gain = contrast / 10.0 + 1.0
    out = np.clip((out - 0.5) * contrast_gain + 0.5, 0, 1)

    # Highlight protection
    out = _protect_highlights(out)

    # Saturation (adaptive)
    sat_gain = _adaptive_sat_gain(scene, 1.0 + sat / GradingConstants.SATURATION_DIVISOR)
    if sat > 0:
        out = _hsv_float_sat(out, sat_gain)

    # Apply the model's warmth nudge after the natural grade. Positive values
    # warm the image, negative values cool it.
    if intent:
        warmth = float(intent.get("warmth", 0.0))
        out[..., 0] = np.clip(out[..., 0] + 0.04 * warmth, 0, 1)
        out[..., 2] = np.clip(out[..., 2] - 0.04 * warmth, 0, 1)

    return np.clip(out, 0, 1)


def grade_variant_b(base, scene=None, intent=None, lut_path=None, lut_strength=1.0):
    """Premium Creative grade: local LUT blend with a safe algorithmic fallback."""
    # LUTs are selected locally from style intent, then blended rather than
    # blindly applied at 100%. This preserves skin/sky detail and makes B an
    # adaptive creative version instead of a fixed cinematic filter.
    if lut_path and Path(lut_path).is_file():
        try:
            from . import lut_engine, style_engine
            table = lut_engine.load_cube_file(Path(lut_path))
            transformed = lut_engine.apply_lut(base, table)
            graded = style_engine.blend_lut(base, transformed, lut_strength)
            return _protect_highlights(graded)
        except Exception as exc:
            get_logger().warn(f"Failed to apply LUT {lut_path}, falling back to algorithmic B grade: {exc}")

    scene = scene or analyze_scene(base)
    out = base.astype(np.float32)

    # Enhanced cinematic parameters
    if intent:
        contrast = max(1, float(intent.get("contrast", 0)) / 2.0) + 2
        sat = max(1, float(intent.get("saturation", 0)) / 2.0) + 2
    else:
        p = scene.get("cinematic", {})
        contrast = p.get("contrast", GradingConstants.CINEMATIC_CONTRAST_BASE)
        sat = p.get("sat", GradingConstants.CINEMATIC_SAT_BASE)

    # Enhanced contrast (S-curve)
    c = contrast / 12.0 + 1.0
    out = np.clip((out - 0.5) * c + 0.5, 0, 1)

    # Teal-orange look (enhanced)
    p = scene.get("cinematic", {})
    warm_mult = p.get("warm_mult", GradingConstants.CINEMATIC_TEAL_ORANGE_ENHANCED) / 100.0
    cool_mult = p.get("cool_shadows", 0.85)

    lum = out.mean(axis=2, keepdims=True)
    warm_mask = np.clip(lum / (0.8 + 1e-6), 0, 1)
    cool_mask = 1.0 - warm_mask

    # Enhanced warmth in highlights, cool in shadows
    out[..., 0] = np.clip(out[..., 0] * (1 + warm_mult * warm_mask[..., 0] - (1-cool_mult) * cool_mask[..., 0]), 0, 1)
    out[..., 2] = np.clip(out[..., 2] * (1 - warm_mult * warm_mask[..., 0] + (1-cool_mult) * cool_mask[..., 0]), 0, 1)

    # Saturation (adaptive)
    sat_gain = _adaptive_sat_gain(scene, 1.0 + sat / GradingConstants.SATURATION_DIVISOR)
    if sat > 0:
        out = _hsv_float_sat(out, sat_gain)

    if intent:
        warmth = float(intent.get("warmth", 0.0))
        out[..., 0] = np.clip(out[..., 0] + 0.04 * warmth, 0, 1)
        out[..., 2] = np.clip(out[..., 2] - 0.04 * warmth, 0, 1)

    # Teal-orange can push highlights back into clipping after the earlier
    # contrast step, so apply the same smooth rolloff at the end as well.
    out = _protect_highlights(out)

    return np.clip(out, 0, 1)


def _grade_variant_with_qa(
    variant,
    crop,
    scene=None,
    intent=None,
    ratio="4:5",
    max_retries=2,
    threshold_pct=None,
    lut_path=None,
    lut_strength=1.0,
):
    """Grade variant with automatic quality assurance feedback loop.

    If the technical QA check fails (e.g. highlight clipping exceeds the
    allowable threshold), this progressively tones down contrast, reduces
    exposure / intent aggressiveness, and strengthens highlight rolloff
    until the output passes or `max_retries` is reached.
    """
    if threshold_pct is None:
        threshold_pct = GradingConstants.QA_HIGHLIGHT_CLIP_MAX_PCT

    current_intent = dict(intent) if intent else {}

    best_out = None
    best_qa = None
    retries_used = 0

    for attempt in range(max_retries + 1):
        if attempt > 0:
            retries_used += 1
            # Progressive damping on each retry
            scale = 0.5 ** attempt
            current_intent = {
                "contrast": float(intent.get("contrast", 0.0)) * scale,
                "saturation": float(intent.get("saturation", 0.0)) * scale,
                "warmth": float(intent.get("warmth", 0.0)) * scale,
            } if intent else {}
        else:
            current_intent = dict(intent) if intent else {}

        if variant == "B":
            # On first attempt use LUT if available; on retries fallback to softer algorithmic B
            active_lut = lut_path if attempt == 0 else None
            out = grade_variant_b(
                crop,
                scene=scene,
                intent=current_intent if intent else None,
                lut_path=active_lut,
                lut_strength=lut_strength,
            )
        else:
            out = grade_variant_a(
                crop, scene=scene, intent=current_intent if intent else None
            )

        # On retries, add extra highlight damping directly to the result if still clipping
        if attempt > 0:
            out = _protect_highlights(out, threshold=0.96 - 0.02 * attempt)
            # Gentle EV pull-down on stubborn highlights
            out = np.clip(out * (0.98 ** attempt), 0, 1)

        qa = technical_qa(out, ratio)

        if best_out is None or qa["checks"]["clipped_high_pct"] < best_qa["checks"]["clipped_high_pct"]:
            best_out = out
            best_qa = qa

        if qa["pass"]:
            return out, qa, retries_used

    return best_out, best_qa, retries_used


def _save_caption_file(out_dir: Path, stem: str, caption_data: Dict[str, Any]) -> Optional[Path]:
    """Save an easy-to-copy Instagram caption and hashtag text file."""
    if not caption_data:
        return None
    try:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        hook = str(caption_data.get("hook", "")).strip()
        caption = str(caption_data.get("caption", "")).strip()
        tags = caption_data.get("hashtags", [])
        tag_str = " ".join(tags) if isinstance(tags, list) else str(tags)

        lines = []
        if hook:
            lines.append(hook)
            lines.append("")
        if caption:
            lines.append(caption)
            lines.append("")
        if tag_str:
            lines.append(".")
            lines.append(".")
            lines.append(tag_str)

        content = "\n".join(lines).strip() + "\n"
        caption_path = out_dir / f"{stem}_caption.txt"
        caption_path.write_text(content, encoding="utf-8")
        return caption_path
    except Exception:
        return None


# ============================================================================
# Cropping & QA
# ============================================================================

def _subject_anchor(rgb, plan):
    """Return a normalized (x, y) anchor for the crop window, or None.

    Resolution order:
      1. ``subject_box`` supplied by the vision model wins.
      2. Local face detection (frontal then profile cascade).
      3. A person-aware fallback: when vision says the subject is a person but
         gave no box, bias the anchor toward the upper third — heads live at
         the top, and a mid-frame crop is what decapitates people.
      4. Saliency (detail centroid) as a last resort.
    """
    plan = plan or {}

    box = plan.get("subject_box")
    if isinstance(box, (list, tuple)) and len(box) == 4:
        try:
            x, y, w, h = [float(c) for c in box]
            w = max(w, 1e-3)
            h = max(h, 1e-3)
            return (min(0.98, max(0.02, x + w / 2.0)),
                    min(0.98, max(0.02, y + h / 2.0)))
        except (TypeError, ValueError):
            pass

    face = _face_anchor(rgb)
    if face is not None:
        return face

    if _is_person_subject(plan):
        # Faces sit in the upper part of a person; combined with the x from
        # the saliency map when available, otherwise centred.
        y = 0.30
        sal = _saliency_anchor(rgb)
        x = sal[0] if sal else 0.5
        return (min(0.9, max(0.1, x)), y)

    # Last resort before the legacy blind crop: find the visually most
    # detailed region (faces sit in the upper half, hats/party light don't
    # confuse a gradient map).
    return _saliency_anchor(rgb)


_PERSON_WORDS = (
    "person", "people", "woman", "man", "girl", "boy", "face", "portrait",
    "selfie", "couple", "family", "model", "human",
)


def _is_person_subject(plan):
    """Best-effort guess that the main subject is a person (faces matter)."""
    scene = str(plan.get("scene_type", "")).lower()
    if scene in ("portrait",):
        return True
    subject = str(plan.get("main_subject", "")).lower()
    return any(word in subject for word in _PERSON_WORDS)


def _face_anchor(rgb):
    """Detect faces and return the centroid of the group, or None.

    Tries the frontal cascade first, then the profile cascade — profile shots
    and party/hat photos regularly defeat the frontal detector. Both are
    local OpenCV models, no network needed.
    """
    gray = None
    try:
        gray = cv2.cvtColor(
            np.clip(rgb, 0, 1).astype(np.float32) * 255.0, cv2.COLOR_RGB2GRAY
        ).astype(np.uint8)
    except Exception:
        return None

    for cascade_name in (
        "haarcascade_frontalface_default.xml",
        "haarcascade_profileface.xml",
    ):
        try:
            cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_name)
            faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(24, 24))
            if len(faces) == 0:
                continue
            h, w = rgb.shape[:2]
            # Larger faces are more reliable; prefer the biggest one.
            x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            return ((x + fw / 2.0) / w, (y + fh / 2.0) / h)
        except Exception:
            continue
    return None


def _saliency_anchor(rgb):
    """Estimate where the visual interest is without any ML model.

    Faces are usually in the upper half, so weight the detail (gradient) map
    toward the top; the anchor is the intensity-weighted centroid. This keeps
    subjects in frame when vision has no box and the Haar cascade misses
    (hats, side profiles, dim party light).
    """
    try:
        gray = cv2.cvtColor(
            np.clip(rgb, 0, 1).astype(np.float32) * 255.0, cv2.COLOR_RGB2GRAY
        ).astype(np.uint8)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        detail = cv2.magnitude(gx, gy)

        h, w = detail.shape
        # Vertical bias: subjects (and especially faces) sit high in the frame.
        rows = np.linspace(0.7, 1.0, h, dtype=np.float32)[:, None]
        weighted = detail * rows

        total = float(weighted.sum())
        if total < 1e-6:
            return None
        ys, xs = np.mgrid[0:h, 0:w]
        cx = float((weighted * xs).sum() / total) / w
        cy = float((weighted * ys).sum() / total) / h
        return (min(0.95, max(0.05, cx)), min(0.95, max(0.05, cy)))
    except Exception:
        return None


def _crop_to_ratio(rgb_float, crop_target, anchor=None):
    """Crop to target ratio, keeping an optional subject anchor in frame."""
    ratios = {
        "4:5": (4, 5), "4/5": (4, 5),
        "9:16": (9, 16), "9/16": (9, 16),
        "1:1": (1, 1)
    }
    num, den = ratios.get(crop_target, (4, 5))
    target = num / den
    h, w = rgb_float.shape[:2]
    cur = w / h

    if anchor is None:
        ax, ay = 0.5, 0.45  # legacy behaviour
    else:
        ax, ay = float(anchor[0]), float(anchor[1])

    if cur > target:
        nw = int(h * target)
        nw = max(1, min(nw, w))
        cx = float(np.clip(ax, 0.05, 0.95)) * w
        x0 = int(np.clip(cx - nw / 2.0, 0, w - nw))
        cropped = rgb_float[:, x0:x0 + nw]
    else:
        nh = int(w / target)
        nh = max(1, min(nh, h))
        cy = float(np.clip(ay, 0.05, 0.95)) * h
        y0 = int(np.clip(cy - nh / 2.0, 0, h - nh))
        cropped = rgb_float[y0:y0 + nh]

    return np.ascontiguousarray(cropped)


def technical_qa(rgb_float, crop_target):
    """Quality assurance check."""
    h, w = rgb_float.shape[:2]
    ratios = {
        "4:5": (4, 5), "4/5": (4, 5),
        "9:16": (9, 16), "9/16": (9, 16),
        "1:1": (1, 1)
    }
    num, den = ratios.get(crop_target, (4, 5))

    # Report the percentage of pixels with at least one clipped channel.  Using
    # ``sum()`` over RGB channels makes the result exceed 100% on solid black or
    # white images and does not describe pixel clipping.
    high_clip = float(
        (rgb_float > GradingConstants.HIGHLIGHT_CLIP_THRESHOLD).any(axis=2).mean()
    )
    low_clip = float(
        (rgb_float < GradingConstants.SHADOW_CLIP_THRESHOLD).any(axis=2).mean()
    )

    checks = {
        "size": f"{w}x{h}",
        "ratio_ok": bool(abs(w / h - num / den) < GradingConstants.QA_RATIO_TOLERANCE),
        "clipped_high_pct": round(high_clip * 100, 2),
        "clipped_low_pct": round(low_clip * 100, 2),
        "brightness_mean": round(float(rgb_float.mean()), 3),
    }

    ok = checks["ratio_ok"] and high_clip < (GradingConstants.QA_HIGHLIGHT_CLIP_MAX_PCT / 100.0)
    return {"checks": checks, "pass": bool(ok)}


# ============================================================================
# Main Processing Functions (Simplified)
# ============================================================================

def build_editing_plan(rgb, crop_target="4:5", src=None, scene_override=None):
    """Build editing plan from scene analysis.

    When `scene_override` is supplied (a plan from the vision model) its
    semantic judgement wins; the local heuristic below is the fallback for when
    vision is disabled or unreachable.
    """
    crop_target = crop_target.replace("/", ":")
    if crop_target not in ("4:5", "9:16", "1:1"):
        crop_target = "4:5"

    if scene_override:
        plan = dict(scene_override)
        plan.setdefault("preserve_region", {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0})
        plan["style_a"] = "premium_natural"
        plan["style_b"] = "premium_cinematic"
        plan["crop_target"] = crop_target
        return plan

    scene = analyze_scene(rgb)
    tags = scene["tags"]

    # Determine scene type
    scene_type = "general"
    main_subject = "subject"
    subject_imp = 0.8
    env_imp = 0.7
    sky_imp = 0.3

    if "night" in tags:
        scene_type, main_subject, subject_imp, env_imp, sky_imp = "night", "subject", 0.85, 0.5, 0.2
    elif "sunset_warm" in tags:
        scene_type = "sunset"
        main_subject = "landscape"
        subject_imp, env_imp, sky_imp = 0.7, 0.9, 0.9
    elif "product" in tags:
        scene_type = "product"
        subject_imp = 0.9

    return {
        "scene_type": scene_type,
        "main_subject": main_subject,
        "subject_importance": subject_imp,
        "environment_importance": env_imp,
        "sky_importance": sky_imp,
        "preserve_colors": [],
        "preserve_region": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0},
        "style_a": "premium_natural",
        "style_b": "premium_cinematic",
        "crop_target": crop_target,
        "provider": "heuristic",
    }


def _unique_output_stem(out_root, stem):
    """Return a stem that will not overwrite existing media outputs."""
    candidate = stem
    index = 2
    suffixes = ("_A.jpg", "_B.jpg", "_A.mp4", "_B.mp4")
    while any(
        (Path(out_root) / folder / f"{candidate}{suffix}").exists()
        for folder in ("POSTS", "STORIES", "REELS")
        for suffix in suffixes
    ):
        candidate = f"{stem}_{index}"
        index += 1
    return candidate


def process_photo(cfg, src, out_root, output_stem=None):
    """Process single photo."""
    logger = get_logger()
    export_mgr = ExportManager(cfg)

    stem = output_stem or Path(src).stem
    rgb = load_rgb(src)
    results = []

    # Apply the technical baseline before semantic analysis and creative
    # grading, so exposure and white balance are consistent across variants.
    rgb, technical_params = normalize_technical(rgb)

    # Analyse once per photo, not once per format: the scene does not change
    # between POSTS and STORIES, and each call costs a request.
    scene_plan = vision.analyze(Path(src)) if vision.is_enabled() else None
    if scene_plan:
        logger.info(
            f"Vision: {scene_plan['scene_type']} | {scene_plan['main_subject']}"
        )

    # The subject anchor is image-level (vision box or detected face), so it is
    # computed once and reused across formats.
    anchor = _subject_anchor(rgb, scene_plan or {})

    for fmt in cfg.get("produce_formats", ["POSTS", "STORIES"]):
        ratio = "9:16" if fmt == "STORIES" else "4:5"
        out_w = cfg.get(
            "output_width_story" if fmt == "STORIES" else "output_width_post", 1080
        )

        # Get editing plan
        plan = build_editing_plan(
            rgb, crop_target=ratio, src=src, scene_override=scene_plan
        )

        # Crop and grade. Keep the scene's subject in frame when the vision
        # model supplied a box or a local face was detected.
        crop = _crop_to_ratio(rgb, ratio, anchor=anchor)
        # The vision model supplies a bounded grading nudge.  Keep the local
        # scene defaults as the base and apply the model intent on top. The
        # scene analysis is shared by both variants instead of being recomputed.
        intent = plan.get("grading_intent") if scene_plan else None
        scene = analyze_scene(crop)
        max_qa_retries = int(cfg.get("max_qa_retries", 2))

        # Vision supplies semantic facts only. The local style engine combines
        # those facts with the account profile, scores compatible LUT candidates,
        # and selects a controlled look strength for B.
        from . import style_engine
        style_intent = style_engine.build_style_intent(scene_plan or plan)
        selected_lut = style_engine.choose_lut(style_intent)
        matched_lut = selected_lut["path"] if selected_lut else None
        lut_strength = selected_lut["strength"] if selected_lut else 0.0
        plan["style_intent"] = style_intent
        plan["style_a"] = "premium_natural"
        plan["style_b"] = f"premium_creative:{style_intent['family']}"
        if selected_lut:
            plan["lut_b"] = selected_lut["name"]
            plan["lut_strength_b"] = lut_strength
            plan["lut_candidates"] = selected_lut["candidates"]

        a, qa_a, retries_a = _grade_variant_with_qa(
            "A", crop, scene=scene, intent=intent, ratio=ratio, max_retries=max_qa_retries
        )
        b, qa_b, retries_b = _grade_variant_with_qa(
            "B",
            crop,
            scene=scene,
            intent=intent,
            ratio=ratio,
            max_retries=max_qa_retries,
            lut_path=matched_lut,
            lut_strength=lut_strength,
        )

        if retries_a > 0 or retries_b > 0:
            logger.info(
                f"[{fmt}] QA auto-recovery applied (A: {retries_a} retries, B: {retries_b} retries)"
            )

        # Export
        out_dir = out_root / fmt
        out_dir.mkdir(parents=True, exist_ok=True)

        files = {
            "A": export_mgr.save_variant(a, out_dir, stem, "A", output_width=out_w),
            "B": export_mgr.save_variant(b, out_dir, stem, "B", output_width=out_w),
        }

        if not all(export_mgr.verify_exports(exported) for exported in files.values()):
            raise RuntimeError(f"Export verification failed for {stem} ({fmt})")

        qa = {"A": qa_a, "B": qa_b}
        m = export_mgr.save_manifest(out_dir, stem, plan, files, qa)

        # Save Instagram caption & hashtags text file alongside the media
        caption_info = plan.get("instagram") or (scene_plan.get("instagram") if scene_plan else None)
        if caption_info:
            _save_caption_file(out_dir, stem, caption_info)

        logger.info(f"[{fmt}] A/B exported | provider={plan.get('provider')}")
        results.append({
            "fmt": fmt,
            "files": files,
            "manifest": str(m),
            "plan": plan,
            "technical": technical_params,
        })

    return results


@functools.lru_cache(maxsize=1)
def _nvenc_available():
    """True only when NVENC actually works end-to-end.

    An encoder being listed in `ffmpeg -encoders` is not enough: a too-old
    NVIDIA driver still lists h264_nvenc but fails to open it at runtime. So
    do a one-frame micro-encode probe and trust the exit code.
    """
    try:
        r = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "color=c=black:s=1280x720:d=0.1:r=1",
                "-frames:v", "1", "-c:v", "h264_nvenc", "-f", "null", "-",
            ],
            capture_output=True, text=True, timeout=60,
        )
        return r.returncode == 0
    except Exception:
        return False


def _build_reel_command(
    cfg,
    src,
    out,
    variant,
    selected_segments=None,
    include_audio=True,
    use_gpu=False,
    template=None,
    lut_path=None,
    lut_strength=1.0,
    output_fps=None,
    output_width=None,
    output_height=None,
):
    """Build an Instagram/mobile-compatible ffmpeg export command."""
    # Keep paths relative to the project working directory.  The project lives
    # on a mapped S: drive, which is visible to Python but not reliably visible
    # to child native processes when passed as an absolute mapped-drive path.
    src = str(Path(src))
    out = str(Path(out))
    output_width = int(output_width or cfg.get("reel_width", 1080))
    output_height = int(output_height or cfg.get("reel_height", 1920))
    output_fps = int(output_fps or cfg.get("reel_output_fps", 30))
    base = (
        f"scale={output_width}:{output_height}:force_original_aspect_ratio=increase,"
        f"crop={output_width}:{output_height}"
    )
    
    # Use 3D LUT filter if Variant B has a LUT file, otherwise fallback to template/eq
    if variant == "B" and lut_path and Path(lut_path).is_file():
        # Video LUT application stays a single temporal FFmpeg filter.  Photo
        # masters use pixel-accurate strength blending; for video, avoiding a
        # second split/blend graph here prevents temporal/filter-graph errors
        # across Best-Cut transitions. The candidate is therefore only used
        # when the account deliberately asks for a strong creative look.
        clean_lut_path = str(Path(lut_path).resolve()).replace("\\", "/").replace(":", "\\:")
        lut_strength = max(0.0, min(1.0, float(lut_strength)))
        color_filter = f"lut3d='{clean_lut_path}'" if lut_strength >= 0.60 else "eq=contrast=1.12:saturation=1.10"
    elif template and "grading" in template:
        t_grading = template["grading"]
        contrast = float(t_grading.get("contrast", 1.2 if variant == "B" else 1.1))
        saturation = float(t_grading.get("saturation", 1.25))
        if variant == "B":
            contrast *= 1.15
            saturation *= 1.1
        color_filter = f"eq=contrast={contrast:.2f}:saturation={saturation:.2f}"
    else:
        contrast = 2.5 if variant == "B" else 1.5
        color_filter = f"eq=contrast={1 + contrast / 10.0}:saturation=1.3"

    ken_burns = bool(
        template.get("ken_burns", False)
        if template
        else cfg.get("video_kenburns", False)
    )
    zoom_speed = template.get("zoom_speed", "0.0015") if template else "0.0015"
    transition = template.get("transition_type", "none") if template else "none"
    transition_duration = float(template.get("transition_duration", 0.5)) if template else 0.5

    command = ["ffmpeg", "-y", "-i", str(src)]
    if selected_segments:
        from .video_tools import build_segment_filter

        filter_graph = build_segment_filter(
            selected_segments,
            base + "," + color_filter,
            include_audio=include_audio,
            transition=transition,
            transition_duration=transition_duration,
            ken_burns=ken_burns,
            zoom_speed=zoom_speed,
            output_width=output_width,
            output_height=output_height,
            output_fps=output_fps,
        )
        command += ["-filter_complex", filter_graph, "-map", "[outv]"]
        if include_audio:
            command += ["-map", "[outa]"]
    else:
        command += ["-map", "0:v:0"]
        if include_audio:
            command += ["-map", "0:a:0?"]
        vf_chain = [base, color_filter]
        if ken_burns:
            # `d=1` preserves source timing; any larger value repeats frames
            # and turns a short clip into minutes of video.
            vf_chain.insert(
                0,
                f"zoompan=z='min(zoom+{zoom_speed},1.15)':d=1:s={output_width}x{output_height}:fps={output_fps}",
            )
        vf_chain.extend([f"fps={output_fps}", "setsar=1", "format=yuv420p"])
        command += ["-vf", ",".join(vf_chain)]

    # 1080x1920 social delivery is H.264 Level 4.1. The higher-resolution
    # working master needs Level 5.1; forcing 4.1 there makes NVENC reject it.
    h264_level = "5.1" if output_width > 1080 or output_height > 1920 else "4.1"
    if use_gpu:
        codec = ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "18", "-b:v", "0", "-profile:v", "main", "-level", h264_level]
    else:
        codec = ["-c:v", "libx264", "-profile:v", "main", "-level", h264_level, "-preset", "fast", "-crf", "18"]

    return command + [
        "-t", str(cfg.get("reel_max_duration", 30)),
        "-r", str(output_fps),
        *codec,
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-ar", "48000", "-ac", "2", "-b:a", "192k",
        "-movflags", "+faststart", "-shortest", str(out),
    ]


def _build_social_derivative_command(
    cfg,
    master_src,
    out,
    include_audio=True,
    use_gpu=False,
    output_fps=30,
):
    """Derive the Instagram delivery file from a single rendered master.

    This is deliberately a separate pass: colour, crop, cuts and transitions
    are baked exactly once in the master.  The final 1080x1920 file only
    resizes/re-encodes that master for Instagram compatibility.
    """
    master_src, out = str(Path(master_src)), str(Path(out))
    scale = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920"
    command = ["ffmpeg", "-y", "-i", master_src, "-map", "0:v:0", "-vf", scale]
    if include_audio:
        command += ["-map", "0:a:0?"]
    codec = (
        ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "18", "-b:v", "0"]
        if use_gpu
        else ["-c:v", "libx264", "-preset", "fast", "-crf", "18"]
    )
    return command + [
        "-r", str(int(output_fps)), *codec,
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", "48000", "-ac", "2", "-b:a", "192k",
        "-movflags", "+faststart", "-shortest", out,
    ]


def _reel_part_path(final_path):
    """Return the hidden/incomplete path used during reel export."""
    final_path = Path(final_path)
    return final_path.with_name(f"{final_path.stem}.part{final_path.suffix}")


def _save_reel_manifest(
    manifest_dir,
    stem,
    source_duration,
    selected_segments,
    provider,
    outputs,
    masters=None,
    template=None,
    lut_name=None,
    source_info=None,
    fps_plan=None,
):
    """Save the editing decision next to the other machine-readable manifests."""
    manifest_dir = Path(manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_duration": round(float(source_duration), 3),
        "output_duration": round(
            sum(float(segment.get("take", 0.0)) for segment in selected_segments), 3
        ) if selected_segments else None,
        "selected_segments": selected_segments or [],
        "editing_provider": provider,
        "template": template.get("name") if template else "Default A/B",
        "lut_b": lut_name,
        "source_technical": source_info or {},
        "fps_plan": fps_plan or {},
        "grading_variants": ["A", "B"],
        "masters": {key: str(value) for key, value in (masters or {}).items()},
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    out_path = manifest_dir / f"{stem}_REELS_manifest.json"
    out_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return out_path


def _select_reel_segments(cfg, segments, scores):
    """Apply the Best-Cut setting and select clips when it is enabled."""
    if not cfg.get("video_best_clips", True):
        return None
    from .video_tools import select_best_segments

    return select_best_segments(
        segments,
        scores,
        max_duration=float(cfg.get("reel_max_duration", 30)),
        max_segments=cfg.get("best_clips_max_segments", 15),
        max_clip_duration=float(cfg.get("reel_max_clip_duration", 8)),
    )


def process_reel(cfg, src, out_root, output_stem=None):
    """Process video reel."""
    from .video_tools import (
        detect_scenes,
        extract_segment_frame,
        frame_quality_score,
        has_audio_stream,
        probe_duration,
        probe_video_info,
        plan_social_fps,
        select_best_segments,
    )

    logger = get_logger()

    out_dir = out_root / "REELS"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem or Path(src).stem

    # Source technical analysis happens before editing: preserve normal source
    # cadence (24/25/30 fps), while 50/60 fps is explicitly marked as a future
    # slow-motion candidate and delivered at social-safe 30 fps by default.
    source_info = probe_video_info(str(src))
    fps_plan = plan_social_fps(source_info)
    duration = source_info["duration"] or probe_duration(str(src)) or 0.0
    segs = detect_scenes(str(src), max_segments=cfg.get("best_clips_max_segments", 15))
    selected_segments = None

    # Get plan from representative frame
    frame = extract_segment_frame(str(src), 0.5) if segs else None
    plan = None
    scene_plan = None

    if frame:
        try:
            rgb = load_rgb(frame)
            # The extracted frame stands in for the video, so vision analyses it.
            scene_plan = vision.analyze(Path(frame)) if vision.is_enabled() else None
            if scene_plan:
                logger.info(
                    f"Vision: {scene_plan['scene_type']} | {scene_plan['main_subject']}"
                )
            plan = build_editing_plan(
                rgb, crop_target="9:16", src=frame, scene_override=scene_plan
            )
        except Exception as e:
            logger.warn(f"Frame analysis failed, using defaults: {e}")
            plan = {"scene_type": "general", "provider": "heuristic"}
        finally:
            try:
                os.unlink(frame)
            except OSError:
                pass

    if not plan:
        plan = {"scene_type": "general", "provider": "heuristic"}

    if (
        cfg.get("video_best_clips", True)
        and duration > float(cfg.get("reel_max_duration", 30))
        and segs
    ):
        samples = []
        temp_frames = []
        try:
            for number, (start, end) in enumerate(segs, 1):
                frame_path = extract_segment_frame(str(src), (start + end) / 2.0)
                if frame_path:
                    temp_frames.append(frame_path)
                    samples.append((number, frame_path, start, end))

            sampled_segments = [
                (start, end) for _, _, start, end in samples
            ]
            sample_numbers = [number for number, _, _, _ in samples]
            local_scores = [frame_quality_score(frame_path) for _, frame_path, _, _ in samples]
            vision_scores = vision.rank_video_segments(samples) if vision.is_enabled() else None
            scores = []
            for number, local_score in zip(sample_numbers, local_scores):
                model_score = (vision_scores or {}).get(number, {}).get("score", local_score)
                scores.append(0.35 * local_score + 0.65 * float(model_score))

            selected_segments = _select_reel_segments(cfg, sampled_segments, scores)
            if selected_segments:
                plan["selected_segments"] = selected_segments
                plan["segment_scores"] = scores
                plan["segment_reasons"] = [
                    (vision_scores or {}).get(number, {}).get("reason", "local quality")
                    for number in sample_numbers
                ]
                logger.info(f"Best-Cut selected {len(selected_segments)} segments")
        finally:
            for frame_path in temp_frames:
                try:
                    os.unlink(frame_path)
                except OSError:
                    pass

    logger.info(f"[Reel] {plan.get('scene_type')} | processing...")

    # Vision describes the scene; local account style chooses compatible LUTs.
    from . import templates, style_engine
    template = templates.select_template_for_scene(scene_plan or plan)
    style_intent = style_engine.build_style_intent(scene_plan or plan)
    selected_lut = style_engine.choose_lut(style_intent)
    matched_lut = selected_lut["path"] if selected_lut else None
    lut_strength = selected_lut["strength"] if selected_lut else 0.0
    lut_name = matched_lut.name if matched_lut else "None"
    plan["style_intent"] = style_intent
    if selected_lut:
        plan["lut_candidates"] = selected_lut["candidates"]
    logger.info(f"[Reel] Applied Template: {template.get('name')} | LUT (B): {lut_name} @ {lut_strength:.2f}")

    # Constant per source — probe once, reuse for both variants.
    has_audio = has_audio_stream(str(src))
    use_gpu = _nvenc_available()

    # Master-first video delivery: the creative edit is rendered once at a
    # higher working resolution. Instagram output is then a simple derivative
    # from that master, so color/crop/transitions are never recomputed.
    masters_dir = Path(
        cfg.get("masters_folder")
        or (Path(cfg.get("output_folder", out_root)).parent / "3_ARCHIV" / "MASTERS")
    ) / "REELS"
    masters_dir.mkdir(parents=True, exist_ok=True)
    reel_outputs = {}
    reel_masters = {}
    for variant in ("A", "B"):
        master = masters_dir / f"{stem}_REEL_{variant}_master.mp4"
        master_part = _reel_part_path(master)
        master_part.unlink(missing_ok=True)
        master_cmd = _build_reel_command(
            cfg,
            src,
            master_part,
            variant,
            selected_segments=selected_segments,
            include_audio=has_audio,
            use_gpu=use_gpu,
            template=template,
            lut_path=matched_lut if variant == "B" else None,
            lut_strength=lut_strength,
            output_fps=fps_plan["output_fps"],
            output_width=int(cfg.get("reel_master_width", 1440)),
            output_height=int(cfg.get("reel_master_height", 2560)),
        )
        r = subprocess.run(master_cmd, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            logger.error(f"Reel master {variant} failed", error=Exception(r.stderr[-200:]))
            master_part.unlink(missing_ok=True)
            raise RuntimeError(f"Reel master {variant} export failed for {src.name}")
        master_part.replace(master)
        reel_masters[variant] = master

        out = out_dir / f"{stem}_{variant}.mp4"
        part = _reel_part_path(out)
        part.unlink(missing_ok=True)
        delivery_cmd = _build_social_derivative_command(
            cfg, master, part, include_audio=has_audio, use_gpu=use_gpu,
            output_fps=fps_plan["output_fps"],
        )
        r = subprocess.run(delivery_cmd, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            logger.error(f"Reel delivery {variant} failed", error=Exception(r.stderr[-200:]))
            part.unlink(missing_ok=True)
            raise RuntimeError(f"Reel delivery {variant} export failed for {src.name}")
        part.replace(out)
        reel_outputs[variant] = out
        logger.info(f"[{variant}] Master -> {master.name}; Instagram -> {out.name}")

    manifest_dir = cfg.get(
        "manifests_folder", out_root.parent / "_SYSTEM" / "manifests"
    )
    _save_reel_manifest(
        manifest_dir,
        stem,
        source_duration=duration,
        selected_segments=selected_segments or [],
        provider="best_cut" if selected_segments else "full_video",
        outputs={
            variant: reel_outputs[variant]
            for variant in ("A", "B")
        },
        masters={variant: str(reel_masters[variant]) for variant in ("A", "B")},
        template=template,
        lut_name=lut_name if matched_lut else None,
        source_info=source_info,
        fps_plan=fps_plan,
    )

    # Save ready-to-copy caption alongside Reels
    caption_info = plan.get("instagram") or (scene_plan.get("instagram") if scene_plan else None)
    if caption_info:
        _save_caption_file(out_dir, stem, caption_info)

    logger.info("Reels complete (A/B variants)")


def run_on_folder(cfg=None, batch_limit=None):
    """Main batch processing."""
    logger = get_logger()

    cfg = cfg or Config.load()
    inp = Path(cfg["input_folder"])
    out = Path(cfg["output_folder"])

    for sub in cfg.get("produce_formats", ["POSTS", "STORIES"]) + ["REELS"]:
        (out / sub).mkdir(parents=True, exist_ok=True)

    photo_ext = PHOTO_EXT
    video_ext = VIDEO_EXT
    all_ext = photo_ext | video_ext

    files = sorted([p for p in inp.iterdir() if p.is_file() and p.suffix.lower() in all_ext])
    if batch_limit:
        files = files[:batch_limit]

    if not files:
        logger.info("No files in input folder")
        return {"ok": [], "failed": [], "report": None}

    pipeline = Pipeline(cfg)

    ok = []
    failed = []
    skipped = []

    # Build the archive hash index once per batch instead of re-scanning the
    # archive (and re-hashing every archived file) for each input file.
    archive_index = _build_archive_index(cfg)

    for src in files:
        kind = "VIDEO" if src.suffix.lower() in video_ext else "PHOTO"

        try:
            # Identical bytes already archived: skip the expensive grading +
            # vision call and drop the extra copy instead of accumulating
            # _2/_3 files. If the check itself raises (permissions, concurrent
            # delete), fall back to processing rather than aborting the batch.
            try:
                is_duplicate = _already_archived(cfg, src, index=archive_index)
            except Exception:
                is_duplicate = False

            if is_duplicate:
                skipped.append(src.name)
                logger.info(f"Skipping duplicate (already archived): {src.name}")
                src.unlink(missing_ok=True)
                continue

            logger.info(f"Processing: {src.name} ({kind})")
            output_stem = _unique_output_stem(out, src.stem)

            if kind == "PHOTO":
                process_photo(cfg, src, out, output_stem=output_stem)
            else:
                process_reel(cfg, src, out, output_stem=output_stem)

            # Archive original
            if cfg.get("auto_move_sources", True):
                if not pipeline.archive_source(src):
                    raise RuntimeError(f"Archiving failed for {src.name}; original kept in input")

            ok.append(src.name)
            logger.success(f"Completed: {src.name}")
        except Exception as e:
            failed.append((src.name, str(e)))
            logger.error(f"Failed to process: {src.name}", error=e)

    report = _write_batch_report(cfg, ok, failed, skipped)
    if failed:
        logger.warn(f"Batch finished: {len(ok)} OK, {len(failed)} failed, {len(skipped)} skipped")
    else:
        logger.success(f"Batch finished: {len(ok)} OK, 0 failed, {len(skipped)} skipped")

    return {"ok": ok, "failed": failed, "skipped": skipped, "report": str(report)}


def _build_batch_report_text(ok, failed, skipped=None):
    """Render a human-readable batch summary."""
    skipped = skipped or []
    lines = [
        "IG-AUTOMATIK Batch-Report",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"OK: {len(ok)}",
        f"Failed: {len(failed)}",
        f"Skipped (duplicate): {len(skipped)}",
        "",
    ]
    if ok:
        lines.append("Erfolgreich:")
        for name in ok:
            lines.append(f"  - {name}")
    if skipped:
        lines.append("")
        lines.append("Duplikate übersprungen:")
        for name in skipped:
            lines.append(f"  - {name}")
    if failed:
        lines.append("")
        lines.append("Fehler:")
        for name, err in failed:
            lines.append(f"  - {name}: {err}")
    lines.append("")
    return "\n".join(lines)


def _write_batch_report(cfg, ok, failed, skipped=None):
    """Persist the batch report next to the manifests."""
    manifests = Path(
        cfg.get("manifests_folder")
        or (Path(cfg.get("output_folder", ".")).parent / "_SYSTEM" / "manifests")
    )
    reports_dir = manifests.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / ("batch_report_" + time.strftime("%Y%m%d_%H%M%S") + ".txt")
    path.write_text(_build_batch_report_text(ok, failed, skipped), encoding="utf-8")
    return path


def _file_hash(path, chunk=1024 * 1024):
    """SHA-256 of a file, streamed so large videos stay cheap on memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _already_archived(cfg, src, index=None):
    """True when a byte-identical copy already lives in the archive folder.

    Pass a pre-built ``index`` (from :func:`_build_archive_index`) to avoid
    re-scanning the archive for every file in a batch — with a large archive
    that would be O(N·A) stat round-trips over the network share. Without an
    index, this falls back to a single scan of the archive.
    """
    if index is None:
        index = _build_archive_index(cfg)
    try:
        size = src.stat().st_size
    except OSError:
        return False

    same_size = index.get(size)
    if not same_size:
        return False

    src_hash = _file_hash(src)
    return src_hash in same_size


def _build_archive_index(cfg):
    """Return {size: set(sha256)} of every file in the archive folder."""
    archive = Path(
        cfg.get("processed_folder")
        or (Path(cfg.get("output_folder", ".")).parent / "3_ARCHIV")
    )
    index = {}
    if not archive.is_dir():
        return index
    for p in archive.iterdir():
        if not p.is_file():
            continue
        try:
            size = p.stat().st_size
            index.setdefault(size, set()).add(_file_hash(p))
        except OSError:
            continue
    return index


if __name__ == "__main__":
    cfg = Config.load()
    run_on_folder(cfg)
