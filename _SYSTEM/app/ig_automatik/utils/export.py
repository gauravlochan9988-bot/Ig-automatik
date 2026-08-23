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

    def save_master_png(
        self,
        rgb_float: np.ndarray,
        stem: str,
        variant: str,
        format_name: str,
    ) -> Optional[Path]:
        """Save a full-resolution, 16-bit lossless master before any IG resize.

        The master is a derived edit, never a replacement for the untouched
        source.  It preserves the post-crop working resolution, whereas the
        social JPG is only a final delivery derivative.
        """
        try:
            masters_dir = Path(
                self.cfg.get("masters_folder")
                or Path(self.cfg.get("processed_folder", "3_ARCHIV")) / "MASTERS"
            )
            masters_dir.mkdir(parents=True, exist_ok=True)
            master16 = np.clip(rgb_float * 65535.0, 0, 65535).astype(np.uint16)
            out_path = masters_dir / f"{stem}_{format_name}_{variant}_master.png"
            bgr16 = cv2.cvtColor(master16, cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(str(out_path), bgr16):
                return None
            return out_path
        except Exception as e:
            self.logger.error("Failed to save lossless master", error=e, stem=stem)
            return None

    def save_variant(
        self,
        rgb_float: np.ndarray,
        out_dir: Path,
        stem: str,
        variant: str,
        output_width: Optional[int] = None,
        format_name: Optional[str] = None,
    ) -> Dict[str, Path]:
        """Export one variant: full-resolution master first, then social JPG."""
        out_files = {}
        format_name = format_name or Path(out_dir).name

        if self.cfg.get("produce_masters", True):
            master_path = self.save_master_png(rgb_float, stem, variant, format_name)
            # A master-first workflow is transactional: never deliver a social
            # derivative when the required lossless master did not land.
            if not master_path:
                self.logger.error("Required master export failed; social derivative withheld", stem=stem)
                return {}
            out_files["master"] = master_path

        if self.cfg.get("produce_ig", True):
            rgb8 = np.clip(rgb_float * 255, 0, 255).astype(np.uint8)
            jpg_path = self.save_jpg_ig(
                rgb8, out_dir, f"{stem}_{variant}", output_width=output_width
            )
            if jpg_path:
                out_files["ig"] = jpg_path

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
                "crop_decision": plan.get("crop_decision"),
                "composition_plan": plan.get("composition_plan"),
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
