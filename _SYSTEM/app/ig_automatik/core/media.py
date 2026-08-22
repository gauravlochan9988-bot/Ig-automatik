"""Media abstraction for unified photo/video processing."""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional
import base64
from io import BytesIO
import urllib.request

import numpy as np
import cv2
from PIL import Image

from ..config import Config, GradingConstants
from ..utils import get_logger

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass


class Media:
    """Base class for media processing."""

    PHOTO_EXT = {".jpg", ".jpeg", ".png", ".gif", ".dng", ".tif", ".tiff", ".bmp", ".webp", ".heic", ".raw", ".nef", ".cr2", ".arw"}
    VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".3gp"}

    def __init__(self, src_path: Path):
        self.src = Path(src_path)
        self.logger = get_logger()

    @staticmethod
    def is_photo(path: Path) -> bool:
        return path.suffix.lower() in Media.PHOTO_EXT

    @staticmethod
    def is_video(path: Path) -> bool:
        return path.suffix.lower() in Media.VIDEO_EXT

    @classmethod
    def factory(cls, src_path: Path) -> "Media":
        """Create appropriate media type."""
        src = Path(src_path)
        if cls.is_photo(src):
            return Photo(src)
        elif cls.is_video(src):
            return Video(src)
        else:
            raise ValueError(f"Unsupported media type: {src.suffix}")

    def load(self) -> Optional[np.ndarray]:
        """Load media. Returns RGB float32 array or None."""
        raise NotImplementedError

    def analyze(self, cfg: Dict) -> Optional[Dict]:
        """Analyze and return editing plan."""
        raise NotImplementedError

    def export(self, cfg: Dict, results: Dict) -> Dict:
        """Export processed results."""
        raise NotImplementedError


class Photo(Media):
    """Photo media processing."""

    def load(self) -> Optional[np.ndarray]:
        """Load photo as float32 RGB (0..1)."""
        try:
            p = self.src
            if p.suffix.lower() in {".dng", ".nef", ".cr2", ".arw", ".tif", ".tiff", ".heic"}:
                with Image.open(str(p)) as im:
                    arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
                return arr

            bgr = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if bgr is None:
                raise ValueError(f"Cannot read image: {p}")

            if bgr.ndim == 2:
                bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)

            if bgr.dtype != np.float32 or bgr.max() > 1.0:
                bgr = bgr.astype(np.float32) / 255.0

            rgb = bgr[..., ::-1]  # BGR->RGB

            if rgb.ndim == 3 and rgb.shape[2] == 4:
                rgb = rgb[..., :3]

            return rgb
        except Exception as e:
            self.logger.error("Failed to load photo", error=e, file=str(self.src))
            return None

    def analyze(self, cfg: Dict) -> Optional[Dict]:
        """Get editing plan from vision API or heuristic."""
        rgb = self.load()
        if rgb is None:
            return None

        env = Config.load_env()
        api_key = env.get("OPENROUTER_API_KEY", "")

        if api_key:
            try:
                return self._get_vision_plan(api_key, env.get("OPENROUTER_MODEL"))
            except Exception as e:
                self.logger.warn("Vision analysis failed", error=e)

        # Fallback to heuristic
        return self._get_heuristic_plan(rgb)

    def _get_vision_plan(self, api_key: str, model: str) -> Optional[Dict]:
        """Call OpenRouter Vision API."""
        try:
            with Image.open(str(self.src)) as im:
                im = im.convert("RGB")
                im.thumbnail((GradingConstants.VISION_PREVIEW_MAX_SIZE, GradingConstants.VISION_PREVIEW_MAX_SIZE))
                buf = BytesIO()
                im.save(buf, "JPEG", quality=GradingConstants.VISION_PREVIEW_QUALITY)
                b64 = base64.b64encode(buf.getvalue()).decode("ascii")

            prompt = """Du bist ein Instagram-Content-Experte fuer Farbkorrektur.
Antworte NUR mit JSON (kein Text):
{"scene_type": "sunset|night|landscape|portrait|food_product|general", "subject_importance": 0-1, "preserve_colors": [...]}"""

            payload = {
                "model": model or "google/gemini-2.5-flash-lite",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ],
                }],
            }

            req = urllib.request.Request(
                "https://openrouter.ai/api/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
            )

            with urllib.request.urlopen(req, timeout=GradingConstants.VISION_API_TIMEOUT) as resp:
                out = json.loads(resp.read().decode("utf-8"))

            response_text = out["choices"][0]["message"]["content"]
            start, end = response_text.find("{"), response_text.rfind("}")
            if start == -1 or end == -1:
                return None

            data = json.loads(response_text[start:end+1])
            return {
                "scene_type": data.get("scene_type", "general"),
                "subject_importance": float(data.get("subject_importance", 0.8)),
                "preserve_colors": data.get("preserve_colors", []),
                "provider": "openrouter",
                "grading_intent": data.get("grading_intent", {}),
            }
        except Exception as e:
            self.logger.error("Vision API failed", error=e)
            return None

    def _get_heuristic_plan(self, rgb: np.ndarray) -> Dict:
        """Generate editing plan from heuristic scene analysis."""
        return {
            "scene_type": "general",
            "subject_importance": 0.8,
            "preserve_colors": [],
            "provider": "heuristic",
            "grading_intent": {},
        }

    def export(self, cfg: Dict, results: Dict) -> Dict:
        """Export photo not implemented in this module."""
        raise NotImplementedError("Use ExportManager for photo exports")


class Video(Media):
    """Video media processing."""

    def load(self) -> Optional[str]:
        """Video returns path since it's too large to load entirely."""
        return str(self.src) if self.src.exists() else None

    def analyze(self, cfg: Dict) -> Optional[Dict]:
        """Get editing plan from video's representative frame."""
        try:
            frame_path = self._extract_frame(0.5)
            if not frame_path:
                return {"scene_type": "general", "provider": "heuristic"}

            photo = Photo(Path(frame_path))
            plan = photo.analyze(cfg) or {"scene_type": "general"}

            try:
                os.unlink(frame_path)
            except Exception:
                pass

            return plan
        except Exception as e:
            self.logger.error("Failed to analyze video", error=e)
            return {"scene_type": "general"}

    def _extract_frame(self, time_offset: float) -> Optional[str]:
        """Extract frame at time offset."""
        try:
            fd, fp = tempfile.mkstemp(suffix=".jpg")
            os.close(fd)
            r = subprocess.run(
                ["ffmpeg", "-y", "-ss", str(time_offset), "-i", str(self.src), "-frames:v", "1", "-q:v", "2", fp],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if os.path.exists(fp) and os.path.getsize(fp) > 0:
                return fp
            return None
        except Exception:
            return None

    def export(self, cfg: Dict, results: Dict) -> Dict:
        """Export video not implemented in this module."""
        raise NotImplementedError("Use grading_engine.process_reel for video exports")
