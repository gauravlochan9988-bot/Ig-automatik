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
        # A valid project root has the whole user-facing layout. Matching only
        # one marker (for example a test-created 3_ARCHIV/MASTERS) can wrongly
        # turn _SYSTEM/app into a second project root.
        if all((candidate / marker).is_dir() for marker in MARKER_FOLDERS):
            return candidate

    # Nothing created yet (fresh checkout): a package inside _SYSTEM still means
    # the root is one level above _SYSTEM, otherwise working folders would be
    # created inside _SYSTEM/app. Handle both the _SYSTEM folder itself and any
    # descendant such as _SYSTEM/app/ig_automatik.
    for candidate in (start, *start.parents):
        if candidate.name == SYSTEM_FOLDER:
            return candidate.parent

    return start


def system_dir(project_root: Path) -> Path:
    """Return the _SYSTEM folder for a given project root."""
    return Path(project_root) / SYSTEM_FOLDER
