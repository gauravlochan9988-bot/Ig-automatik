#!/usr/bin/env python3
"""Small local bridge between the iPhone PWA and IG-AUTOMATIK.

This is intentionally a separate program. It does not import or modify the
existing grading engine: uploads are atomically placed in 1_EINGANG, where the
existing watchdog continues to do the real work.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import socket
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


DEFAULT_PROJECT_ROOT = r"S:\all my projects\IG-AUTOMATIK"
PHOTO_EXT = {".jpg", ".jpeg", ".png", ".gif", ".dng", ".tif", ".tiff", ".bmp", ".webp", ".heic", ".raw", ".nef", ".cr2", ".arw"}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".3gp"}
FORMATS = ("POSTS", "STORIES", "REELS")
MAX_UPLOAD_BYTES = 4 * 1024 * 1024 * 1024


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_part(value: str, fallback: str = "upload") -> str:
    value = Path(value or "").name
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return value[:80] or fallback


class MobileBridge:
    def __init__(self, project_root: Path, data_dir: Path):
        self.project_root = project_root.expanduser().resolve()
        self.input_dir = self.project_root / "1_EINGANG"
        self.output_dir = self.project_root / "2_FERTIG"
        self.masters_dir = self.project_root / "3_ARCHIV" / "MASTERS"
        self.jobs_dir = data_dir.resolve() / "jobs"
        self.posters_dir = data_dir.resolve() / "posters"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.posters_dir.mkdir(parents=True, exist_ok=True)

    def create_job(self, original_name: str) -> dict:
        original_name = Path(unquote(original_name or "")).name
        extension = Path(original_name).suffix.lower()
        if extension not in PHOTO_EXT | VIDEO_EXT:
            raise ValueError("Dieses Foto- oder Videoformat wird nicht unterstützt.")

        job_id = uuid.uuid4().hex
        original_stem = safe_part(Path(original_name).stem)
        stem = f"iphone_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{job_id[:8]}_{original_stem}"
        source_name = stem + extension
        job = {
            "id": job_id,
            "original_name": original_name,
            "source_name": source_name,
            "stem": stem,
            "created": utc_now(),
        }
        self._write_job(job)
        return job

    def _job_path(self, job_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{32}", job_id):
            raise FileNotFoundError(job_id)
        return self.jobs_dir / f"{job_id}.json"

    def _write_job(self, job: dict) -> None:
        path = self._job_path(job["id"])
        temporary = path.with_suffix(".part")
        temporary.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)

    def get_job(self, job_id: str) -> dict:
        path = self._job_path(job_id)
        if not path.is_file():
            raise FileNotFoundError(job_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def delete_job(self, job_id: str) -> None:
        """Remove a job record when its upload did not complete."""
        self._job_path(job_id).unlink(missing_ok=True)

    def save_upload(self, job: dict, source_stream, content_length: int | None) -> None:
        if content_length is not None and content_length < 0:
            raise ValueError("Ungültige Upload-Größe.")
        if content_length is not None and content_length > MAX_UPLOAD_BYTES:
            raise ValueError("Die Datei ist größer als 4 GB.")

        self.input_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.input_dir / f".igmobile-{job['id']}.uploading"
        destination = self.input_dir / job["source_name"]
        written = 0
        try:
            with temporary.open("wb") as output:
                while True:
                    remaining = None if content_length is None else content_length - written
                    if remaining == 0:
                        break
                    chunk = source_stream.read(1024 * 1024 if remaining is None else min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > MAX_UPLOAD_BYTES:
                        raise ValueError("Die Datei ist größer als 4 GB.")
                    output.write(chunk)

            if content_length is not None and written != content_length:
                raise ValueError("Der Upload wurde unvollständig übertragen.")
            temporary.replace(destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

    def _output_files(self, job: dict) -> dict[str, list[dict]]:
        result = {}
        for format_name in FORMATS:
            directory = self.output_dir / format_name
            files = []
            for variant in ("A", "B"):
                matches = sorted(directory.glob(f"{job['stem']}_{variant}.*")) if directory.is_dir() else []
                for path in matches:
                    if path.is_file() and path.suffix.lower() not in {".part", ".json"}:
                        item = {
                            "variant": variant,
                            "name": path.name,
                            "url": f"/api/download/{job['id']}/{format_name}/{path.name}",
                            "preview_url": f"/api/preview/{job['id']}/{format_name}/{path.name}",
                        }
                        master = self.master_file(job, format_name, variant, missing_ok=True)
                        item["master_available"] = master is not None
                        if master is not None:
                            item["master_url"] = f"/api/master/{job['id']}/{format_name}/{variant}"
                            item["master_name"] = master.name
                        if path.suffix.lower() in VIDEO_EXT:
                            item["poster_url"] = f"/api/poster/{job['id']}/{format_name}/{path.name}"
                        files.append(item)
            if files:
                result[format_name] = files
        return result

    def master_file(self, job: dict, format_name: str, variant: str, missing_ok=False) -> Path | None:
        """Return the edited master belonging to one visible delivery file."""
        format_name = str(format_name).upper()
        variant = str(variant).upper()
        if format_name not in FORMATS or variant not in {"A", "B"}:
            raise FileNotFoundError("master")
        if format_name == "REELS":
            path = self.masters_dir / "REELS" / f"{job['stem']}_REEL_{variant}_master.mp4"
        else:
            path = self.masters_dir / f"{job['stem']}_{format_name}_{variant}_master.png"
        if path.is_file() and path.stat().st_size > 0:
            return path
        if missing_ok:
            return None
        raise FileNotFoundError(path.name)

    def poster_path(self, job_id: str, format_name: str, filename: str) -> Path:
        """Create/cache a small JPEG poster for a video preview."""
        video_path = self.download_path(job_id, format_name, filename)
        if video_path.suffix.lower() not in VIDEO_EXT:
            raise FileNotFoundError(filename)

        poster = self.posters_dir / f"{job_id}_{safe_part(filename, 'poster')}.jpg"
        if poster.is_file() and poster.stat().st_size > 0:
            return poster

        temporary = poster.with_suffix(".part.jpg")
        temporary.unlink(missing_ok=True)
        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-ss", "0.5", "-i", str(video_path), "-frames:v", "1",
                    "-vf", "scale=480:-2", "-q:v", "5", str(temporary),
                ],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0 or not temporary.is_file() or temporary.stat().st_size == 0:
                raise FileNotFoundError(filename)
            temporary.replace(poster)
            return poster
        except (OSError, subprocess.SubprocessError):
            temporary.unlink(missing_ok=True)
            raise FileNotFoundError(filename)

    def snapshot(self, job: dict) -> dict:
        outputs = self._output_files(job)
        source = self.input_dir / job["source_name"]
        if outputs:
            status = "done"
        elif source.exists():
            status = "processing"
        else:
            status = "waiting"
        return {**job, "status": status, "outputs": outputs}

    def list_jobs(self) -> list[dict]:
        jobs = []
        for path in self.jobs_dir.glob("*.json"):
            try:
                jobs.append(self.snapshot(json.loads(path.read_text(encoding="utf-8"))))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        # Keep the complete history available. The web UI can filter/search it
        # without making older uploads disappear.
        return sorted(jobs, key=lambda item: item.get("created", ""), reverse=True)

    def download_path(self, job_id: str, format_name: str, filename: str) -> Path:
        job = self.get_job(job_id)
        format_name = format_name.upper()
        if format_name not in FORMATS or Path(filename).name != filename:
            raise FileNotFoundError(filename)
        allowed = {item["name"] for item in self._output_files(job).get(format_name, [])}
        if filename not in allowed:
            raise FileNotFoundError(filename)
        path = (self.output_dir / format_name / filename).resolve()
        if not path.is_relative_to((self.output_dir / format_name).resolve()) or not path.is_file():
            raise FileNotFoundError(filename)
        return path


class Handler(BaseHTTPRequestHandler):
    bridge: MobileBridge
    web_dir: Path

    def log_message(self, fmt, *args):
        # Windows consoles may use cp1252; malformed HTTPS probes can contain
        # arbitrary bytes that must never crash a request handler.
        message = "[Mobile] " + (fmt % args) + "\n"
        sys.stdout.write(message.encode("ascii", "backslashreplace").decode("ascii"))

    def send_json(self, payload: dict, status=HTTPStatus.OK):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            if path == "/api/health":
                self.send_json({
                    "ok": True,
                    "project_exists": self.bridge.project_root.is_dir(),
                    "input_exists": self.bridge.input_dir.is_dir(),
                    "output_exists": self.bridge.output_dir.is_dir(),
                })
                return
            if path == "/api/jobs":
                self.send_json({"jobs": self.bridge.list_jobs()})
                return
            if path.startswith("/api/download/"):
                parts = path.split("/")
                if len(parts) != 6:
                    raise FileNotFoundError(path)
                file_path = self.bridge.download_path(parts[3], parts[4], parts[5])
                inline = parse_qs(parsed.query).get("inline", ["0"])[0] == "1"
                self._send_file(file_path, attachment=not inline)
                return
            if path.startswith("/api/preview/"):
                parts = path.split("/")
                if len(parts) != 6:
                    raise FileNotFoundError(path)
                file_path = self.bridge.download_path(parts[3], parts[4], parts[5])
                self._send_file(file_path, attachment=False)
                return
            if path.startswith("/api/poster/"):
                parts = path.split("/")
                if len(parts) != 6:
                    raise FileNotFoundError(path)
                poster = self.bridge.poster_path(parts[3], parts[4], parts[5])
                self._send_file(poster, attachment=False)
                return
            if path.startswith("/api/master/"):
                parts = path.split("/")
                if len(parts) != 6:
                    raise FileNotFoundError(path)
                job = self.bridge.get_job(parts[3])
                master = self.bridge.master_file(job, parts[4], parts[5])
                inline = parse_qs(parsed.query).get("inline", ["0"])[0] == "1"
                self._send_file(master, attachment=not inline)
                return
            self._send_static(path)
        except FileNotFoundError:
            self.send_json({"error": "Nicht gefunden."}, HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/upload":
            self.send_json({"error": "Nicht gefunden."}, HTTPStatus.NOT_FOUND)
            return

        filename = self.headers.get("X-Filename", "")
        length_header = self.headers.get("Content-Length")
        try:
            content_length = int(length_header) if length_header else None
            job = self.bridge.create_job(filename)
            try:
                self.bridge.save_upload(job, self.rfile, content_length)
            except Exception:
                # A failed or interrupted upload must not remain as a false
                # waiting job in the mobile history.
                self.bridge.delete_job(job["id"])
                raise
            self.send_json({"job": self.bridge.snapshot(job)}, HTTPStatus.CREATED)
        except ValueError as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def _send_static(self, request_path: str):
        relative = "index.html" if request_path in ("", "/") else request_path.lstrip("/")
        candidate = (self.web_dir / relative).resolve()
        if not candidate.is_relative_to(self.web_dir.resolve()) or not candidate.is_file():
            self.send_json({"error": "Nicht gefunden."}, HTTPStatus.NOT_FOUND)
            return
        self._send_file(candidate)

    def _send_file(self, path: Path, attachment=False):
        size = path.stat().st_size
        start, end = 0, size - 1
        partial = False
        range_header = self.headers.get("Range")
        if range_header and size > 0:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
            if match:
                raw_start, raw_end = match.groups()
                if raw_start:
                    start = int(raw_start)
                    end = int(raw_end) if raw_end else end
                elif raw_end:
                    length = int(raw_end)
                    start = max(0, size - length)
                if start <= end < size:
                    partial = True
                else:
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return

        self.send_response(HTTPStatus.PARTIAL_CONTENT if partial else HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Cache-Control", "no-store" if attachment else "no-cache")
        if attachment:
            safe_name = path.name.replace('"', "")
            self.send_header("Content-Disposition", f'attachment; filename="{safe_name}"')
        self.end_headers()
        try:
            with path.open("rb") as source:
                source.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    chunk = source.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            # Safari/iOS may cancel and immediately restart a range request
            # while loading video metadata or seeking. This is not a server
            # error and must not trigger a second, misleading HTTP 500 reply.
            return


def main() -> int:
    parser = argparse.ArgumentParser(description="Lokale iPhone-Oberfläche für IG-AUTOMATIK")
    parser.add_argument("--project-root", default=os.environ.get("IG_AUTOMATIK_ROOT", DEFAULT_PROJECT_ROOT))
    parser.add_argument("--host", default="0.0.0.0", help="0.0.0.0 für WLAN/Tailscale")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--data-dir", default=str(Path(__file__).parent / "data"))
    args = parser.parse_args()

    bridge = MobileBridge(Path(args.project_root), Path(args.data_dir))
    Handler.bridge = bridge
    Handler.web_dir = Path(__file__).parent / "web"
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        print(f"IG-AUTOMATIK Mobile läuft auf Port {args.port}")
        print(f"Projekt: {bridge.project_root}")
        print("Öffne auf dem iPhone: http://<PC-IP>:8787")
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nIG-AUTOMATIK Mobile beendet.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
