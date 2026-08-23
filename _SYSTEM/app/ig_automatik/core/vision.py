"""Vision-based scene analysis via OpenRouter.

The grading engine asks this module what a photo *is* (sunset, night, portrait,
food) so it can pick crop and grading intent. When no key is configured, or the
call fails for any reason, callers fall back to the local heuristic -- vision is
an enhancement, never a hard dependency of the pipeline.
"""

import base64
import functools
import json
import ssl
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional

from PIL import Image, ImageDraw

from ..config import Config, GradingConstants
from ..utils import get_logger

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass

API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemini-2.5-flash-lite"

SCENE_TYPES = {"sunset", "night", "landscape", "portrait", "food_product", "general"}

PROMPT = """You are an image analysis and colour-grading expert. Look at the image and \
describe it for an automated grading pipeline.

Reply with JSON only, no prose:
{
  "scene_type": "sunset|night|landscape|portrait|food_product|general",
  "main_subject": "short description of the main subject",
  "subject_importance": 0.0-1.0,
  "subject_box": [x, y, width, height] as normalized 0..1 box around the main subject,
  "composition_plan": {
    "subject_type": "full_body_person|upper_body_person|portrait|group|object|scene",
    "protected_regions": [
      {"name": "head|body|feet|main_subject|important_object", "box": [x, y, width, height], "required": true}
    ],
    "preferred_position": "center|slightly_upper_center|environment_preserving",
    "allow_zoom": true,
    "preserve_environment": true
  },
  "environment_importance": 0.0-1.0,
  "sky_importance": 0.0-1.0,
  "preserve_colors": ["colours that must not shift, e.g. skin tones"],
  "grading_intent": {"warmth": -1.0-1.0, "contrast": -1.0-1.0, "saturation": -1.0-1.0},
  "creative_direction": {
    "preserve": ["important details to retain: face, skin tones, sky, environment"],
    "light_mood": "golden_hour|soft_daylight|harsh_sun|blue_hour|night_neon|indoor_warm|party_flash|general",
    "style_family": "warm_travel|editorial_portrait|documentary_travel|night_cinematic|nature_rich|clean_creator",
    "composition": {
      "post_4_5": {"subject_priority": 0.0-1.0, "environment_priority": 0.0-1.0},
      "story_9_16": {"subject_priority": 0.0-1.0, "environment_priority": 0.0-1.0}
    }
  }
}

subject_box is REQUIRED. Always return a bounding box around the complete
visible main subject, not just the face. For a visible full-body person, also
return separate protected head, body, and feet regions in composition_plan.
Approximate coordinates are fine: [x, y, width, height] all in 0..1 where
0,0 is the top-left of the image. If feet are not visible, do not invent a
feet box. subject_box is a fallback; composition_plan is more important for
safe cropping. Never claim that a region is visible when it is not.
subject_importance says how tightly a crop should hold the subject; \
sky_importance how much sky is worth keeping. grading_intent is a nudge \
relative to a neutral grade, where 0 means "leave as is"."""


@functools.lru_cache(maxsize=1)
def _ssl_context():
    """Build an SSL context with a usable CA bundle (cached per process).

    A framework Python without "Install Certificates.command" run has no trust
    store, so every HTTPS call raises CERTIFICATE_VERIFY_FAILED. certifi ships
    one; fall back to the system default when it is unavailable.
    """
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def is_enabled() -> bool:
    """True when an API key is configured."""
    return bool(Config.load_env().get("OPENROUTER_API_KEY", "").strip())


def _encode_preview(src: Path) -> str:
    """Downscale to a preview JPEG and base64-encode it.

    Full-resolution uploads cost tokens and add latency without improving the
    scene call, so send a bounded preview.
    """
    with Image.open(str(src)) as im:
        im = im.convert("RGB")
        im.thumbnail((
            GradingConstants.VISION_PREVIEW_MAX_SIZE,
            GradingConstants.VISION_PREVIEW_MAX_SIZE,
        ))
        buf = BytesIO()
        im.save(buf, "JPEG", quality=GradingConstants.VISION_PREVIEW_QUALITY)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _clamp(value, low, high, default):
    """Coerce a model-supplied number into range, tolerating junk."""
    try:
        return max(low, min(high, float(value)))
    except (TypeError, ValueError):
        return default


def analyze(src: Path) -> Optional[Dict]:
    """Ask the vision model to describe `src`.

    Returns a plan dict, or None if vision is unavailable or the call failed --
    the caller is expected to fall back to the heuristic plan.
    """
    logger = get_logger()
    env = Config.load_env()
    api_key = env.get("OPENROUTER_API_KEY", "").strip()

    if not api_key:
        return None

    model = env.get("OPENROUTER_MODEL", "").strip() or DEFAULT_MODEL

    try:
        b64 = _encode_preview(Path(src))
    except Exception as e:
        logger.warn(f"Vision preview failed for {Path(src).name}: {e}")
        return None

    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }],
    }

    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://github.com/ig-automatik",
            "X-Title": "IG-AUTOMATIK",
        },
    )

    try:
        with urllib.request.urlopen(
            req, timeout=GradingConstants.VISION_API_TIMEOUT, context=_ssl_context()
        ) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", "replace")[:300]
        except Exception:
            pass
        logger.warn(f"Vision API HTTP {e.code} for {Path(src).name}: {detail}")
        return None
    except Exception as e:
        logger.warn(f"Vision API unreachable for {Path(src).name}: {e}")
        return None

    try:
        text = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        logger.warn(f"Vision API returned no content for {Path(src).name}")
        return None

    # Models often wrap JSON in prose or a ```json fence; take the outermost
    # brace pair rather than trusting the response to be bare JSON.
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        logger.warn(f"Vision reply had no JSON for {Path(src).name}")
        return None

    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError as e:
        logger.warn(f"Vision reply was not valid JSON for {Path(src).name}: {e}")
        return None

    scene_type = str(data.get("scene_type", "general")).strip().lower()
    if scene_type not in SCENE_TYPES:
        scene_type = "general"

    intent_raw = data.get("grading_intent") or {}
    if not isinstance(intent_raw, dict):
        intent_raw = {}

    preserve = data.get("preserve_colors") or []
    if not isinstance(preserve, list):
        preserve = []

    box = data.get("subject_box")
    subject_box = None
    if isinstance(box, (list, tuple)) and len(box) == 4:
        try:
            subject_box = [
                _clamp(box[0], 0.0, 1.0, 0.0),
                _clamp(box[1], 0.0, 1.0, 0.0),
                _clamp(box[2], 0.0, 1.0, 1.0),
                _clamp(box[3], 0.0, 1.0, 1.0),
            ]
        except (TypeError, ValueError):
            subject_box = None

    usage = body.get("usage") or {}

    creative_raw = data.get("creative_direction") or {}
    if not isinstance(creative_raw, dict):
        creative_raw = {}
    preserve_items = creative_raw.get("preserve") or []
    if not isinstance(preserve_items, list):
        preserve_items = []
    composition_raw = creative_raw.get("composition") or {}
    if not isinstance(composition_raw, dict):
        composition_raw = {}

    def _format_composition(format_key, fallback_subject, fallback_environment):
        values = composition_raw.get(format_key) or {}
        if not isinstance(values, dict):
            values = {}
        return {
            "subject_priority": _clamp(values.get("subject_priority"), 0.0, 1.0, fallback_subject),
            "environment_priority": _clamp(values.get("environment_priority"), 0.0, 1.0, fallback_environment),
        }

    creative_direction = {
        "preserve": [str(item)[:60] for item in preserve_items[:10]],
        "light_mood": str(creative_raw.get("light_mood", "general"))[:40].lower(),
        "style_family": str(creative_raw.get("style_family", ""))[:50].lower(),
        "composition": {
            "post_4_5": _format_composition("post_4_5", 0.80, 0.70),
            "story_9_16": _format_composition("story_9_16", 0.90, 0.45),
        },
    }

    composition_plan_raw = data.get("composition_plan") or {}
    if not isinstance(composition_plan_raw, dict):
        composition_plan_raw = {}
    protected_raw = composition_plan_raw.get("protected_regions") or []
    if not isinstance(protected_raw, list):
        protected_raw = []
    protected_regions = []
    for item in protected_raw[:12]:
        if not isinstance(item, dict):
            continue
        raw_box = item.get("box")
        if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
            continue
        protected_regions.append({
            "name": str(item.get("name", "main_subject"))[:40].lower(),
            "box": [
                _clamp(raw_box[0], 0.0, 1.0, 0.0),
                _clamp(raw_box[1], 0.0, 1.0, 0.0),
                _clamp(raw_box[2], 0.0, 1.0, 1.0),
                _clamp(raw_box[3], 0.0, 1.0, 1.0),
            ],
            "required": bool(item.get("required", True)),
        })
    subject_type = str(
        composition_plan_raw.get("subject_type", "")
    )[:40].lower()
    if subject_type not in {
        "full_body_person", "upper_body_person", "portrait", "group", "object", "scene"
    }:
        subject_type = ""
    composition_plan = {
        "subject_type": subject_type,
        "protected_regions": protected_regions,
        "preferred_position": str(
            composition_plan_raw.get("preferred_position", "center")
        )[:40].lower(),
        "allow_zoom": bool(composition_plan_raw.get("allow_zoom", True)),
        "preserve_environment": bool(composition_plan_raw.get("preserve_environment", True)),
    }

    return {
        "scene_type": scene_type,
        "main_subject": str(data.get("main_subject", "subject"))[:120],
        "subject_importance": _clamp(data.get("subject_importance"), 0.0, 1.0, 0.8),
        "environment_importance": _clamp(data.get("environment_importance"), 0.0, 1.0, 0.7),
        "sky_importance": _clamp(data.get("sky_importance"), 0.0, 1.0, 0.3),
        "subject_box": subject_box,
        "composition_plan": composition_plan,
        "preserve_colors": [str(c)[:40] for c in preserve[:8]],
        "grading_intent": {
            "warmth": _clamp(intent_raw.get("warmth"), -1.0, 1.0, 0.0),
            "contrast": _clamp(intent_raw.get("contrast"), -1.0, 1.0, 0.0),
            "saturation": _clamp(intent_raw.get("saturation"), -1.0, 1.0, 0.0),
        },
        "creative_direction": creative_direction,
        "provider": "openrouter",
        "model": model,
        "tokens": usage.get("total_tokens"),
    }


def rank_video_segments(frames) -> Optional[Dict[int, Dict]]:
    """Rank representative video frames in one vision request.

    ``frames`` contains ``(segment_number, frame_path, start, end)`` tuples.
    The contact sheet lets the model compare the candidates instead of judging
    every frame in isolation. Returns a mapping of segment number to score and
    reason, or None when vision is unavailable.
    """
    logger = get_logger()
    env = Config.load_env()
    api_key = env.get("OPENROUTER_API_KEY", "").strip()
    if not api_key or not frames:
        return None

    try:
        thumb_w, thumb_h = 320, 220
        columns = 3
        rows = (len(frames) + columns - 1) // columns
        sheet = Image.new("RGB", (columns * thumb_w, rows * thumb_h), "white")
        draw = ImageDraw.Draw(sheet)
        for position, (number, frame_path, start, end) in enumerate(frames):
            with Image.open(str(frame_path)) as image:
                image = image.convert("RGB")
                image.thumbnail((thumb_w - 8, thumb_h - 34))
                x = (position % columns) * thumb_w + (thumb_w - image.width) // 2
                y = (position // columns) * thumb_h + 26
                sheet.paste(image, (x, y))
            draw.text(
                ((position % columns) * thumb_w + 6, (position // columns) * thumb_h + 6),
                f"#{number} {start:.1f}s-{end:.1f}s",
                fill="black",
            )

        buf = BytesIO()
        sheet.save(buf, "JPEG", quality=88)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        prompt = """Compare the numbered video segments in this contact sheet.
Return JSON only as an array. Score each segment from 0 to 1 for suitability
for a short Instagram Reel: clear subject, interesting action, good framing,
and useful story value. Penalize blur, empty frames, duplicates, and bad
composition. Do not score color grading.
Format: [{"segment": 1, "score": 0.0, "reason": "short reason"}]"""
        payload = {
            "model": env.get("OPENROUTER_MODEL", "").strip() or DEFAULT_MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }],
        }
        req = urllib.request.Request(
            API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        with urllib.request.urlopen(
            req, timeout=GradingConstants.VISION_API_TIMEOUT, context=_ssl_context()
        ) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        text = body["choices"][0]["message"]["content"]
        start, end = text.find("["), text.rfind("]")
        if start == -1 or end <= start:
            return None
        raw = json.loads(text[start:end + 1])
        result = {}
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                number = int(item["segment"])
            except (KeyError, TypeError, ValueError):
                continue
            result[number] = {
                "score": _clamp(item.get("score"), 0.0, 1.0, 0.5),
                "reason": str(item.get("reason", ""))[:160],
            }
        return result or None
    except Exception as exc:
        logger.warn(f"Video segment ranking failed: {exc}")
        return None
