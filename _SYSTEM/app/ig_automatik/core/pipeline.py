"""Pipeline orchestration (source archiving only).

The live batch flow drives grading through the free functions in
``grading_engine`` (``process_photo`` / ``process_reel`` / ``run_on_folder``);
this class only owns moving a processed original safely into the archive.
"""

import shutil
from pathlib import Path
from typing import Dict

from ..utils import get_logger


class Pipeline:
    """Main pipeline for batch processing media."""

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.logger = get_logger()

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