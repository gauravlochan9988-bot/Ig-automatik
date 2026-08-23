"""3D LUT (.cube) Parser, Trilinear Interpolator, and LUT Library Manager."""

import functools
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from ..config.paths import find_project_root, system_dir

LUT_DIR_NAME = "luts"


@functools.lru_cache(maxsize=16)
def load_cube_file(cube_path: Path) -> np.ndarray:
    """Parse an Adobe .cube file into a 3D NumPy array of shape (size, size, size, 3)."""
    cube_path = Path(cube_path)
    if not cube_path.is_file():
        raise FileNotFoundError(f"LUT file not found: {cube_path}")

    size = None
    data_lines = []

    with open(cube_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if parts[0] == "LUT_3D_SIZE":
                size = int(parts[1])
                continue
            if parts[0] in ("TITLE", "LUT_1D_SIZE", "DOMAIN_MIN", "DOMAIN_MAX"):
                continue

            if len(parts) == 3:
                try:
                    data_lines.append([float(parts[0]), float(parts[1]), float(parts[2])])
                except ValueError:
                    continue

    if size is None:
        raise ValueError(f"Invalid .cube file (no LUT_3D_SIZE): {cube_path}")

    expected_count = size * size * size
    if len(data_lines) < expected_count:
        raise ValueError(
            f"Corrupt .cube file {cube_path}: expected {expected_count} entries, got {len(data_lines)}"
        )

    # In .cube format: Red changes fastest, then Green, then Blue.
    # We reshape to (size_b, size_g, size_r, 3)
    raw_array = np.array(data_lines[:expected_count], dtype=np.float32)
    table = raw_array.reshape((size, size, size, 3))
    return table


def apply_lut(rgb_float: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Apply a 3D LUT to an RGB float32 image (0.0 .. 1.0) using trilinear interpolation."""
    orig_shape = rgb_float.shape
    size = table.shape[0]

    # Flatten pixels to (N, 3)
    pixels = np.clip(rgb_float.reshape(-1, 3), 0.0, 1.0).astype(np.float32)

    # Scale coordinates to LUT index space [0, size - 1]
    coords = pixels * (size - 1)
    
    # R, G, B channels
    r_coords = coords[:, 0]
    g_coords = coords[:, 1]
    b_coords = coords[:, 2]

    r0 = np.floor(r_coords).astype(np.int32)
    r1 = np.clip(r0 + 1, 0, size - 1)
    dr = (r_coords - r0)[:, None]

    g0 = np.floor(g_coords).astype(np.int32)
    g1 = np.clip(g0 + 1, 0, size - 1)
    dg = (g_coords - g0)[:, None]

    b0 = np.floor(b_coords).astype(np.int32)
    b1 = np.clip(b0 + 1, 0, size - 1)
    db = (b_coords - b0)[:, None]

    # Trilinear 8 corner samples: table index is [B, G, R]
    c000 = table[b0, g0, r0]
    c001 = table[b0, g0, r1]
    c010 = table[b0, g1, r0]
    c011 = table[b0, g1, r1]
    c100 = table[b1, g0, r0]
    c101 = table[b1, g0, r1]
    c110 = table[b1, g1, r0]
    c111 = table[b1, g1, r1]

    # Interpolate along R
    c00 = c000 * (1.0 - dr) + c001 * dr
    c01 = c010 * (1.0 - dr) + c011 * dr
    c10 = c100 * (1.0 - dr) + c101 * dr
    c11 = c110 * (1.0 - dr) + c111 * dr

    # Interpolate along G
    c0 = c00 * (1.0 - dg) + c01 * dg
    c1 = c10 * (1.0 - dg) + c11 * dg

    # Interpolate along B
    result = c0 * (1.0 - db) + c1 * db

    return np.clip(result.reshape(orig_shape), 0.0, 1.0).astype(np.float32)


def get_luts_directory() -> Path:
    """Return the central LUTs directory path (_SYSTEM/luts)."""
    # This module lives in _SYSTEM/app/ig_automatik/core/. Start discovery at
    # _SYSTEM itself so fresh/nested layouts resolve to the visible project root
    # rather than accidentally creating _SYSTEM/app/_SYSTEM/luts.
    root = find_project_root(Path(__file__).resolve().parents[3])
    lut_dir = system_dir(root) / LUT_DIR_NAME
    lut_dir.mkdir(parents=True, exist_ok=True)
    return lut_dir


def list_luts() -> List[Path]:
    """List all .cube LUT files in the LUTs directory."""
    lut_dir = get_luts_directory()
    return sorted(list(lut_dir.glob("*.cube")))


def match_lut_for_scene(scene_plan: Optional[Dict[str, Any]] = None) -> Optional[Path]:
    """Smart auto-selection of a LUT based on scene description or default cinematic."""
    available = list_luts()
    if not available:
        return None

    if not scene_plan:
        return available[0]

    scene_type = str(scene_plan.get("scene_type", "")).lower()
    main_subject = str(scene_plan.get("main_subject", "")).lower()
    text = f"{scene_type} {main_subject}".lower()

    # Match rules prioritizing high quality sRGB / standard profiles:
    # 1. Beach/Sunset/Florida/Vacation -> Velvia (punchy colors) or Universal Punch
    # 2. Fashion/Portrait/Skin/Wedding -> Classic Chrome / Pro Neg / Colorify Fashion
    # 3. Cinematic/Film/Street -> Classic Neg / True Cinematic
    # 4. Night/Party/Moody -> Eterna / Bleach Bypass / D-Cinelike Blockbuster
    # 5. Monochrome/B&W -> MonoPhotoRedux
    
    for p in available:
        name = p.stem.lower()
        if any(w in text for w in ("sunset", "beach", "summer", "florida", "ocean", "pool", "sun")):
            if "velvia_srgb" in name or "punch" in name:
                return p
        if any(w in text for w in ("fashion", "portrait", "woman", "person", "wedding", "model", "face")):
            if "classic chrome_srgb" in name or "pro neg hi_srgb" in name or "fashion" in name:
                return p
        if any(w in text for w in ("travel", "street", "city", "architecture", "landscape")):
            if "classic neg_srgb" in name or "nostalgic neg_srgb" in name or "cinematic" in name:
                return p
        if any(w in text for w in ("night", "party", "club", "dark", "concert")):
            if "eterna_srgb" in name or "bleach bypass_srgb" in name or "blockbuster" in name or "cinelike" in name:
                return p
        if any(w in text for w in ("bw", "black and white", "monochrome", "vintage")):
            if "monophotoredux" in name or "sepia" in name:
                return p

    # Fallback to high quality sRGB Classic Chrome or True Cinematic
    for p in available:
        if "classic chrome_srgb" in p.stem.lower():
            return p
    for p in available:
        if "cinematic" in p.stem.lower():
            return p

    return available[0]
