#!/usr/bin/env python3
"""IG-AUTOMATIK Auto-Watchdog - Monitor folder and auto-process files.

Usage:
    python watch.py                 # Start watching 1_EINGANG folder
    python watch.py --help          # Show help
"""

import time
import signal
import socket
import subprocess
import sys
import os
from pathlib import Path
from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler

# Add package to path. Works whether this file sits in the project root or in
# a subfolder such as _SYSTEM/, so the entry point keeps working if it is moved.
_here = Path(__file__).resolve().parent
PACKAGE_DIR = next(
    (c for c in (_here, *_here.parents) if (c / "ig_automatik").is_dir()),
    _here,
)
sys.path.insert(0, str(PACKAGE_DIR))

from ig_automatik.config.paths import find_project_root, system_dir
from ig_automatik.utils import get_logger

# The working folders live in the user-facing project root, which is *not* the
# package folder when the code is hidden inside _SYSTEM.
# PACKAGE_DIR is _SYSTEM/app in the nested layout. Start root discovery at its
# enclosing _SYSTEM folder to avoid treating _SYSTEM/app as a fresh project.
ROOT = find_project_root(PACKAGE_DIR.parent)
INPUT = ROOT / "1_EINGANG"
MAIN = PACKAGE_DIR / "main.py"
# Runtime state belongs with the machinery, not in the user's clean folder view.
LOCK = system_dir(ROOT) / "watchdog.lock"

logger = get_logger()


def _pid_alive(pid):
    """Check whether a process is still running (Windows and POSIX)."""
    if os.name == "nt":
        import ctypes
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        h = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not h:
            return False
        try:
            exitcode = ctypes.c_ulong(0)
            ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(exitcode))
            return exitcode.value == STILL_ACTIVE
        finally:
            ctypes.windll.kernel32.CloseHandle(h)

    try:
        os.kill(pid, 0)  # signal 0 only probes for existence
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by another user
    return True


def _acquire_single_instance():
    """Lock file - prevent duplicate watchers.

    The lock lives next to the project on the NAS, so it is visible from every
    machine. A PID is only meaningful on the host that wrote it, hence the
    hostname: two watchers on one folder would process each file twice.
    """
    lock = LOCK
    lock.parent.mkdir(parents=True, exist_ok=True)
    me = socket.gethostname()
    contents = f"{os.getpid()} {me}"

    # Create the lock atomically. A check followed by write_text() has a race:
    # two startup triggers can both observe a missing/stale lock and then both
    # start watching the same folder.
    for _ in range(3):
        try:
            with lock.open("x", encoding="utf-8") as handle:
                handle.write(contents)
            return
        except FileExistsError:
            try:
                parts = lock.read_text(encoding="utf-8").strip().split(None, 1)
                pid = int(parts[0])
                host = parts[1] if len(parts) > 1 else me
            except (FileNotFoundError, ValueError, OSError):
                # Another starter may be replacing a stale lock. Retry the
                # atomic create instead of allowing both processes through.
                continue

            if host != me:
                print(f"[Watchdog] A watcher is registered on '{host}' (PID {pid}).")
                print("[Watchdog] Two watchers on one folder double-process files.")
                print("[Watchdog] Stop it there, or delete watchdog.lock to override.")
                sys.exit(0)
            if _pid_alive(pid):
                print(f"[Watchdog] Already running (PID {pid}). Exiting.")
                sys.exit(0)

            logger.info(f"Clearing stale lock from dead PID {pid}")
            try:
                lock.unlink()
            except FileNotFoundError:
                pass
        except SystemExit:
            raise
        except Exception as e:
            logger.warn(f"Single-instance check failed: {e}")
            return

    logger.info("Another watcher acquired the lock while startup was in progress")
    sys.exit(0)


def _release_single_instance():
    """Remove lock file."""
    try:
        lock = LOCK
        if lock.exists() and lock.read_text().strip().split(None, 1)[0] == str(os.getpid()):
            lock.unlink()
    except Exception:
        pass


PHOTO_EXT = {".jpg", ".jpeg", ".png", ".gif", ".dng", ".tif", ".tiff", ".bmp", ".webp", ".heic", ".raw", ".nef", ".cr2", ".arw"}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".3gp"}

# Seconds between directory scans when polling a network share. Low enough to
# feel instant, high enough to stay cheap over SMB.
POLL_INTERVAL = 2


def is_media(p):
    """Check if file is photo or video."""
    if p.name.startswith((".", "~$")):
        return False  # .DS_Store, resource forks, partial downloads
    return p.suffix.lower() in (PHOTO_EXT | VIDEO_EXT)


class Handler(FileSystemEventHandler):
    """File system event handler."""

    def __init__(self):
        self._pending_paths = set()

    def on_created(self, event):
        self._handle(event)

    def on_modified(self, event):
        self._handle(event)

    def on_moved(self, event):
        # A rename INTO the folder (copy completes via rename) is new work.
        # A move OUT (archiving to 3_ARCHIV) targets a path outside INPUT and
        # must not re-trigger a scan.
        if event.is_directory:
            return
        dest = Path(event.dest_path)
        if is_media(dest) and dest.is_relative_to(INPUT):
            self._note(dest)

    def _handle(self, event):
        if event.is_directory:
            return

        p = Path(event.src_path)
        if is_media(p):
            self._note(p)

    def _note(self, p):
        """Record that work is waiting; the main loop drains it."""
        key = str(p.resolve())
        if key in self._pending_paths:
            return
        self._pending_paths.add(key)
        kind = "VIDEO" if p.suffix.lower() in VIDEO_EXT else "PHOTO"
        logger.info(f"Detected: {p.name} ({kind})")

    def take_pending(self):
        """Return whether files are waiting to be processed."""
        if self._pending_paths:
            return True
        return False

    def clear_pending(self):
        """Forget the current batch after the folder has been scanned."""
        self._pending_paths.clear()


def wait_for_stable(p, timeout=180, settle=2.0):
    """Wait for a file to stop growing, so we never grade a half-copied file.

    Large videos copied over a network share can take minutes, so the timeout is
    generous; the loop exits as soon as the size holds steady for `settle`.
    """
    size = -1
    stable_since = None
    t0 = time.time()

    while time.time() - t0 < timeout:
        try:
            current = p.stat().st_size
        except (FileNotFoundError, OSError):
            return False

        if current == size and current > 0:
            if stable_since is None:
                stable_since = time.time()
            elif time.time() - stable_since >= settle:
                return True
        else:
            stable_since = None

        size = current
        time.sleep(0.5)

    logger.warn(f"File still changing after {timeout}s, processing anyway: {p.name}")
    return True


def run_processing():
    """Run grading engine on new files."""
    if not MAIN.exists():
        logger.error(f"Cannot process: entry point not found at {MAIN}")
        return

    logger.info("Starting processing...")
    run_kwargs = {"cwd": str(ROOT)}
    # On Windows, launching python.exe from the watchdog creates a new console
    # window for every batch. Keep batch processing attached to the watchdog
    # without opening a popup for each arriving file/burst.
    if os.name == "nt":
        run_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

    proc = subprocess.run([sys.executable, str(MAIN)], **run_kwargs)

    if proc.returncode != 0:
        logger.warn(f"Processing ended with code {proc.returncode}")
    else:
        logger.success("Processing complete")


def _is_network_path(path):
    """Best-effort check for whether `path` lives on a network share.

    OS change notifications (FSEvents on macOS, inotify on Linux) are generated
    by the kernel that owns the filesystem. For an SMB/NFS mount that kernel is
    the NAS, so a file written by *another* machine -- or by the NAS itself --
    produces no local event and the watcher never wakes up. Polling compares
    directory listings instead, which works regardless of who wrote the file.
    """
    p = str(path)

    if os.name == "nt":
        # UNC path, or a drive letter mapped to a remote share.
        if p.startswith("\\\\"):
            return True
        try:
            import ctypes
            DRIVE_REMOTE = 4
            drive = os.path.splitdrive(os.path.abspath(p))[0]
            if drive:
                return ctypes.windll.kernel32.GetDriveTypeW(drive + "\\") == DRIVE_REMOTE
        except Exception:
            pass
        return False

    # POSIX: ask the mount table which filesystem type backs this path.
    network_fs = {
        "smbfs", "cifs", "nfs", "nfs4", "afpfs", "webdav", "fuse.sshfs", "afs",
    }
    try:
        out = subprocess.run(
            ["df", "-P", p], capture_output=True, text=True, timeout=15
        ).stdout.splitlines()
        if len(out) >= 2:
            source = out[1].split()[0]
            # A remote source is "//user@host/share" or "host:/export".
            if source.startswith("//") or (":" in source and not source.startswith("/")):
                return True
    except Exception:
        pass

    try:
        for line in subprocess.run(
            ["mount"], capture_output=True, text=True, timeout=15
        ).stdout.splitlines():
            # "//user@host/share on /Volumes/X (smbfs, ...)"
            if " on " not in line or "(" not in line:
                continue
            mount_point = line.split(" on ", 1)[1].split(" (")[0]
            fstype = line.rsplit("(", 1)[1].split(",")[0].strip(") ")
            if fstype in network_fs and os.path.abspath(p).startswith(mount_point):
                return True
    except Exception:
        pass

    return False


def _make_observer():
    """Pick an observer that actually sees changes for this folder."""
    force = os.environ.get("IG_WATCH_MODE", "").strip().lower()

    if force == "poll":
        logger.info("Using polling observer (forced by IG_WATCH_MODE=poll)")
        return PollingObserver(timeout=POLL_INTERVAL)
    if force == "native":
        logger.info("Using native observer (forced by IG_WATCH_MODE=native)")
        return Observer()

    if _is_network_path(INPUT):
        logger.info(
            f"Network folder detected - polling every {POLL_INTERVAL}s "
            "(OS file events are not delivered across network shares)"
        )
        return PollingObserver(timeout=POLL_INTERVAL)

    logger.info("Local folder detected - using native OS file events")
    return Observer()


def main():
    """Main watchdog loop."""
    # pythonw.exe has no console on Windows. Use UTF-8 with replacement so
    # the informational banner cannot abort startup on a legacy code page.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    print("""
    ╔════════════════════════════════════╗
    ║  IG-AUTOMATIK Watchdog (Auto)      ║
    ║  Monitoring 1_EINGANG folder...    ║
    ╚════════════════════════════════════╝
    """)

    INPUT.mkdir(parents=True, exist_ok=True)

    _acquire_single_instance()

    # Turn a service-manager stop (SIGTERM) into the same clean shutdown as
    # Ctrl+C, so the lock file is always released.
    def _on_term(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _on_term)

    observer = None
    try:
        # Process existing files at startup
        existing = [p for p in INPUT.iterdir() if p.is_file() and is_media(p)]
        if existing:
            logger.info(f"Found {len(existing)} files at startup")
            for p in existing:
                logger.info(f"  • {p.name}")
            run_processing()

        # A file can land while the startup batch runs (the observer is not
        # scheduled yet), so sweep once more so nothing is stranded.
        late = [p for p in INPUT.iterdir() if p.is_file() and is_media(p)]
        if late:
            logger.info(f"Found {len(late)} additional files after startup sweep")
            run_processing()

        # Start watching
        event_handler = Handler()
        observer = _make_observer()
        observer.schedule(event_handler, str(INPUT), recursive=False)
        observer.start()

        logger.info(f"Watching: {INPUT}")
        print("Press Ctrl+C to stop\n")

        # Processing runs here rather than in the event callback: grading can take
        # minutes, and blocking the observer thread would drop events meanwhile.
        # A whole burst of arrivals collapses into one batch run.
        while True:
            if event_handler.take_pending():
                # Drain the trigger BEFORE the (long) batch run. A file that
                # arrives while run_processing() runs lands in a fresh set and
                # is handled on the next loop pass instead of being wiped by a
                # clear_pending() executed after the run.
                event_handler.clear_pending()
                for p in sorted(INPUT.iterdir()):
                    if p.is_file() and is_media(p):
                        wait_for_stable(p)

                # Re-check: batch.py archives what it processes, so only act if
                # something is genuinely still waiting.
                if any(p.is_file() and is_media(p) for p in INPUT.iterdir()):
                    run_processing()

            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Stopping watchdog...")
        print("\n✓ Watchdog stopped")
    finally:
        if observer is not None:
            observer.stop()
            observer.join()
        _release_single_instance()


if __name__ == "__main__":
    main()
