"""Media type helpers: extension sets and photo/video classification."""

from pathlib import Path


class Media:
    """Supported media extensions and file-type classification.

    The actual grading/cropping/exporting lives in ``grading_engine``; this
    class only centralizes what counts as a photo or a video, so every part
    of the pipeline (watchdog, batch runner, tests) agrees on the same sets.
    """

    PHOTO_EXT = {".jpg", ".jpeg", ".png", ".gif", ".dng", ".tif", ".tiff", ".bmp", ".webp", ".heic", ".raw", ".nef", ".cr2", ".arw"}
    VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".3gp"}

    @staticmethod
    def is_photo(path: Path) -> bool:
        return path.suffix.lower() in Media.PHOTO_EXT

    @staticmethod
    def is_video(path: Path) -> bool:
        return path.suffix.lower() in Media.VIDEO_EXT

    @staticmethod
    def is_media(path: Path) -> bool:
        """True for any supported photo or video extension."""
        return path.suffix.lower() in (Media.PHOTO_EXT | Media.VIDEO_EXT)