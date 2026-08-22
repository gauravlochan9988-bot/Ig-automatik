"""Project root discovery.

The code may live either directly in the project root (flat layout) or tucked
away inside the ``_SYSTEM`` folder so the user only sees the four working
folders. Both layouts must resolve ``1_EINGANG`` / ``2_FERTIG`` / ``3_ARCHIV``
to the same place, so the root is discovered by looking for those folders
instead of counting parent directories.
"""

from pathlib import Path

# Folders that mark the user-facing project root.
MARKER_FOLDERS = ("1_EINGANG", "2_FERTIG", "3_ARCHIV")

# Name of the folder that hides the machinery from the user.
SYSTEM_FOLDER = "_SYSTEM"


def find_project_root(start: Path) -> Path:
    """Return the project root for a path inside the project.

    ``start`` is any directory belonging to the project (typically the folder
    that contains the ``ig_automatik`` package).
    """
    start = Path(start).resolve()

    for candidate in (start, *start.parents):
        if any((candidate / marker).is_dir() for marker in MARKER_FOLDERS):
            return candidate

    # Nothing created yet (fresh checkout): a package inside _SYSTEM still means
    # the root is one level up, otherwise the working folders would be created
    # inside _SYSTEM.
    if start.name == SYSTEM_FOLDER:
        return start.parent

    return start


def system_dir(project_root: Path) -> Path:
    """Return the _SYSTEM folder for a given project root."""
    return Path(project_root) / SYSTEM_FOLDER
