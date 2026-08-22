"""Configuration loader with validation."""

import json
from pathlib import Path
from typing import Dict, Any

from .paths import find_project_root, system_dir

# The package may sit in the project root or inside _SYSTEM; discover the root
# from the user-facing working folders rather than a fixed number of parents.
PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent.parent.parent)
SYSTEM_DIR = system_dir(PROJECT_ROOT)


class Config:
    """Configuration loader with validation."""

    CONFIG_FILE = SYSTEM_DIR / "config" / "config.json"
    ENV_FILE = PROJECT_ROOT / ".env"

    DEFAULTS = {
        "export_format": "jpg",
        "export_quality": 98,
        "output_width_post": 1080,
        "output_width_story": 1080,
        "ratio_post": "4/5",
        "ratio_story": "9/16",
        "produce_archives": True,
        "produce_ig": True,
        "produce_formats": ["POSTS", "STORIES"],
        "reel_max_duration": 30,
        "reel_max_clip_duration": 8,
        "cost_limit_per_batch": 1.0,
        "max_qa_retries": 2,
        "safe_edit_only": True,
        "auto_move_sources": True,
        "video_kenburns": False,
        "video_best_clips": True,
        "best_clips_max_segments": 15,
    }

    SUPPORTED_RATIOS = {"4:5", "9:16", "1:1"}
    SUPPORTED_FORMATS = {"POSTS", "STORIES", "REELS"}

    @classmethod
    def load(cls) -> Dict[str, Any]:
        """Load and validate configuration."""
        cls.CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)

        cfg = {}
        if cls.CONFIG_FILE.exists():
            try:
                cfg = json.loads(cls.CONFIG_FILE.read_text(encoding="utf-8"))
            except Exception:
                cfg = {}

        # Merge with defaults
        cfg = {**cls.DEFAULTS, **cfg}

        # Resolve folder paths relative to the project root.
        # The config file is shared between machines (Windows PC and macOS both
        # reach this project over the NAS), so a stored path is only honoured
        # when it is usable on the current platform.
        cls._resolve_folder(cfg, "input_folder", PROJECT_ROOT / "1_EINGANG")
        cls._resolve_folder(cfg, "output_folder", PROJECT_ROOT / "2_FERTIG")
        cls._resolve_folder(cfg, "processed_folder", PROJECT_ROOT / "3_ARCHIV")
        cls._resolve_folder(cfg, "manifests_folder", SYSTEM_DIR / "manifests")

        # Drop keys from older versions that pointed at folders the pipeline
        # never reads, so they are not recreated on save.
        for stale in ("captions_folder", "style_folder", "trends_folder"):
            cfg.pop(stale, None)

        # Validate and save back
        cfg = cls._validate(cfg)
        cls.CONFIG_FILE.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

        return cfg

    @staticmethod
    def _resolve_folder(cfg: Dict[str, Any], key: str, fallback: Path):
        """Point `key` at a usable absolute path, falling back to the project root.

        A stored value is discarded when it was written on another platform --
        e.g. a Windows drive path like "S:\\...", which Path() on macOS treats as
        a relative name and silently creates as a junk folder in the cwd.
        """
        raw = str(cfg.get(key) or "").strip()
        # is_absolute() uses the running platform's convention, which is exactly
        # the test we want: "S:\..." is absolute on Windows but not on macOS,
        # and "/Volumes/..." is absolute on macOS but not on Windows.
        if raw and Path(raw).is_absolute():
            cfg[key] = str(Path(raw))
        else:
            cfg[key] = str(fallback)

    @classmethod
    def _validate(cls, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Validate config values."""
        # Validate export quality (0-100)
        cfg["export_quality"] = max(1, min(100, int(cfg.get("export_quality", 98))))

        # Validate output widths (min 640)
        for key in ["output_width_post", "output_width_story"]:
            cfg[key] = max(640, int(cfg.get(key, 1080)))

        # Validate produce formats
        formats = cfg.get("produce_formats", ["POSTS", "STORIES"])
        cfg["produce_formats"] = [f for f in formats if f in cls.SUPPORTED_FORMATS]
        if not cfg["produce_formats"]:
            cfg["produce_formats"] = ["POSTS", "STORIES"]

        # Validate booleans
        for key in ["produce_archives", "produce_ig", "safe_edit_only", "auto_move_sources"]:
            cfg[key] = bool(cfg.get(key, cls.DEFAULTS.get(key, True)))

        # Validate reel settings
        cfg["reel_max_duration"] = max(5, min(120, int(cfg.get("reel_max_duration", 30))))
        cfg["best_clips_max_segments"] = max(2, min(30, int(cfg.get("best_clips_max_segments", 15))))

        return cfg

    @classmethod
    def load_env(cls) -> Dict[str, str]:
        """Load environment variables from .env file."""
        env = {}
        if cls.ENV_FILE.exists():
            for line in cls.ENV_FILE.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
        return env
