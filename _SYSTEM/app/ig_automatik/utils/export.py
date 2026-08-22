"""Export manager for unified media export handling."""

import json
import time
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import cv2

from ..config import GradingConstants
from .logging_utils import get_logger


class ExportManager:
    """Manages all export operations."""

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.logger = get_logger()

    def save_jpg_ig(
        self,
        rgb8: np.ndarray,
        out_dir: Path,
        filename: str,
        output_width: Optional[int] = None,
    ) -> Optional[Path]:
        """Save JPG for IG (1080px width)."""
        try:
            h, w = rgb8.shape[:2]
            target_width = int(output_width or self.cfg.get("output_width_post", 1080))
            scale = target_width / w
            if scale != 1:
                resized = cv2.resize(
                    rgb8,
                    (target_width, int(round(h * scale))),
                    interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LANCZOS4,
                )
            else:
                resized = rgb8

            out_path = out_dir / f"{filename}.jpg"
            bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
            cv2.imwrite(
                str(out_path),
                bgr,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.cfg["export_quality"]],
            )
            return out_path
        except Exception as e:
            self.logger.error("Failed to save JPG", error=e, filename=filename)
            return None

    def save_png_archive(self, rgb8: np.ndarray, out_dir: Path, filename: str) -> Optional[Path]:
        """Save PNG for archive (full resolution)."""
        try:
            out_path = out_dir / f"{filename}_archiv.png"
            bgr = cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_path), bgr)
            return out_path
        except Exception as e:
            self.logger.error("Failed to save PNG archive", error=e, filename=filename)
            return None

    def save_variant(
        self,
        rgb_float: np.ndarray,
        out_dir: Path,
        stem: str,
        variant: str,
        output_width: Optional[int] = None,
    ) -> Dict[str, Path]:
        """Export one grading variant (jpg ig + png archive if enabled)."""
        rgb8 = np.clip(rgb_float * 255, 0, 255).astype(np.uint8)
        out_files = {}

        if self.cfg.get("produce_ig", True):
            jpg_path = self.save_jpg_ig(
                rgb8, out_dir, f"{stem}_{variant}", output_width=output_width
            )
            if jpg_path:
                out_files["ig"] = jpg_path

        if self.cfg.get("produce_archives", True):
            png_path = self.save_png_archive(rgb8, out_dir, f"{stem}_{variant}")
            if png_path:
                out_files["archive"] = png_path

        return out_files

    def save_manifest(
        self,
        out_dir: Path,
        stem: str,
        plan: Dict,
        files: Dict[str, Dict[str, Path]],
        qa: Dict[str, Dict],
    ) -> Optional[Path]:
        """Save processing manifest JSON to the manifests folder (not the output folder)."""
        try:
            manifest = {
                "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
                "original_untouched": True,
                "provider": plan.get("provider"),
                "scene_type": plan.get("scene_type"),
                "main_subject": plan.get("main_subject"),
                "subject_importance": plan.get("subject_importance"),
                "environment_importance": plan.get("environment_importance"),
                "sky_importance": plan.get("sky_importance"),
                "preserve_colors": plan.get("preserve_colors"),
                "grading_intent": plan.get("grading_intent"),
                "crop_target": plan.get("crop_target"),
                "style_a": plan.get("style_a"),
                "style_b": plan.get("style_b"),
                "files": {k: {kk: str(vv) for kk, vv in v.items()} for k, v in files.items()},
                "qa": qa,
            }

            # Manifests live in _SYSTEM/manifests/, not in the output folders,
            # so 2_FERTIG only ever contains the processed media.
            manifests_dir = Path(
                self.cfg.get("manifests_folder", out_dir.parent.parent / "_SYSTEM" / "manifests")
            )
            manifests_dir.mkdir(parents=True, exist_ok=True)
            # Keep one manifest per asset and format.  A fixed filename such as
            # ``POSTS_manifest.json`` would overwrite the previous asset.
            out_path = manifests_dir / f"{stem}_{Path(out_dir).name}_manifest.json"
            out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
            return out_path
        except Exception as e:
            self.logger.error("Failed to save manifest", error=e, stem=stem)
            return None

    def verify_exports(self, out_files: Dict[str, Path]) -> bool:
        """Verify exported files exist and have content."""
        if not out_files:
            return False
        for ftype, path in out_files.items():
            if not path.exists() or path.stat().st_size == 0:
                self.logger.warn(f"Export verification failed: {ftype} at {path}")
                return False
        return True
