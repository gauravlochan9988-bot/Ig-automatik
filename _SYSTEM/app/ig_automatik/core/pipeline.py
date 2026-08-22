"""Pipeline orchestration for unified batch processing."""

import shutil
from pathlib import Path
from typing import Dict, List, Optional

from ..config import Config
from ..utils import get_logger
from .media import Media


class Pipeline:
    """Main pipeline for batch processing media."""

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.logger = get_logger()
        self._temp_files = []

    def process_batch(self, media_list: List[Path]) -> Dict[str, Dict]:
        """Process batch of media files."""
        results = {}

        for i, src_path in enumerate(media_list, 1):
            self.logger.info(f"Processing {i}/{len(media_list)}: {src_path.name}")
            try:
                result = self.process_item(src_path)
                results[str(src_path)] = result
            except Exception as e:
                self.logger.error(f"Batch item failed: {src_path.name}", error=e)
                results[str(src_path)] = {"status": "error", "error": str(e)}

        self._cleanup_temp_files()
        return results

    def process_item(self, src_path: Path) -> Dict:
        """Process single media item."""
        src = Path(src_path)

        if not src.exists():
            raise FileNotFoundError(f"Media file not found: {src}")

        try:
            media = Media.factory(src)
            kind = "video" if Media.is_video(src) else "photo"
            self.logger.info(f"Processing {kind}: {src.name}")

            return {
                "status": "pending",
                "type": kind,
                "src": str(src),
                "size_mb": round(src.stat().st_size / (1024 * 1024), 2),
            }
        except Exception as e:
            self.logger.error(f"Failed to process item", error=e, file=str(src))
            raise

    def archive_source(self, src_path: Path) -> bool:
        """Move processed source to archive (original untouched)."""
        try:
            src = Path(src_path)
            arch = Path(self.cfg.get("processed_folder", Path(__file__).parent.parent.parent / "3_ARCHIV"))
            arch.mkdir(parents=True, exist_ok=True)

            dest = arch / src.name
            if dest.exists():
                # Never overwrite an older original. Keep the newly processed
                # source by choosing the next free archive name instead.
                stem, suffix = src.stem, src.suffix
                index = 2
                while dest.exists():
                    dest = arch / f"{stem}_{index}{suffix}"
                    index += 1
                self.logger.warn(
                    f"Archive destination exists, using unique name: {dest.name}"
                )

            shutil.move(str(src), str(dest))
            self.logger.info(f"Archived: {src.name} -> {arch.name}")
            return True
        except Exception as e:
            self.logger.error("Failed to archive source", error=e, file=str(src_path))
            return False

    def register_temp_file(self, path: Path):
        """Register temporary file for cleanup."""
        self._temp_files.append(path)

    def _cleanup_temp_files(self):
        """Clean up registered temporary files."""
        for path in self._temp_files:
            try:
                if isinstance(path, str):
                    path = Path(path)
                if path.exists():
                    if path.is_dir():
                        shutil.rmtree(path)
                    else:
                        path.unlink()
            except Exception as e:
                self.logger.warn(f"Failed to cleanup temp file: {path}", error=e)

        self._temp_files.clear()


class BatchProcessor:
    """High-level batch processing coordinator."""

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.logger = get_logger()
        self.pipeline = Pipeline(cfg)

    def run(self, batch_limit: Optional[int] = None) -> Dict:
        """Run full processing pipeline."""
        try:
            with self.logger.operation("Full batch processing"):
                media_files = self._discover_media()

                if batch_limit:
                    media_files = media_files[:batch_limit]

                if not media_files:
                    self.logger.info("No media files found in input folder")
                    return {"status": "empty", "count": 0}

                self.logger.info(f"Found {len(media_files)} media files to process")

                results = self.pipeline.process_batch(media_files)

                summary = {
                    "status": "complete",
                    "total": len(media_files),
                    "success": sum(1 for r in results.values() if r.get("status") != "error"),
                    "errors": sum(1 for r in results.values() if r.get("status") == "error"),
                }

                self.logger.success(f"Batch complete: {summary['success']}/{summary['total']} succeeded")
                return summary
        except Exception as e:
            self.logger.error("Batch processing failed", error=e)
            raise

    def _discover_media(self) -> List[Path]:
        """Discover all media files in input folder."""
        inp = Path(self.cfg["input_folder"])
        if not inp.exists():
            return []

        photo_ext = Media.PHOTO_EXT | Media.VIDEO_EXT
        files = sorted([
            p for p in inp.iterdir()
            if p.is_file() and p.suffix.lower() in photo_ext
        ])

        return files
