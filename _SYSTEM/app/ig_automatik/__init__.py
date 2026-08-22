"""IG-AUTOMATIK - Instagram Content Grading Pipeline"""

__version__ = "2.0.0"
__author__ = "IG-AUTOMATIK"
__description__ = "Professional photo and video grading for Instagram"

from .config import Config, GradingConstants
from .utils import Logger, get_logger
from .core import (
    process_photo,
    process_reel,
    run_on_folder,
)

__all__ = [
    "Config",
    "GradingConstants",
    "Logger",
    "get_logger",
    "process_photo",
    "process_reel",
    "run_on_folder",
]
