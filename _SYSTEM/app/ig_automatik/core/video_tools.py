"""Video utility functions for scene detection and frame extraction."""

import os
import re
import subprocess
import tempfile
from pathlib import Path

import cv2


def probe_duration(path):
    """Get video duration in seconds."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=60
        )
        return float(r.stdout.strip())
    except Exception:
        return None


def has_audio_stream(path):
    """Return whether the input contains at least one audio stream."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=60
        )
        return r.returncode == 0 and bool(r.stdout.strip())
    except Exception:
        return False


def detect_scenes(src, max_segments=15):
    """Detect scene changes via ffmpeg."""
    try:
        dur = probe_duration(src) or 0
        if dur <= 0:
            return []

        r = subprocess.run(
            ["ffmpeg", "-i", str(src),
             "-vf", "select='gt(scene,0.3)',showinfo",
             "-f", "null", "-"],
            capture_output=True, text=True, timeout=120
        )

        times = []
        for m in re.finditer(r"pts_time:([0-9.]+)", r.stderr):
            t = float(m.group(1))
            if t > 0.2 and t < dur - 0.2:
                times.append(t)

        times = sorted(set(round(t, 1) for t in times))
        segments = []
        prev = 0.0

        for t in times:
            segments.append((prev, t))
            prev = t

        segments.append((prev, dur))

        # Consolidate if too many
        if len(segments) > max_segments:
            step = len(segments) / max_segments
            segs = []
            for i in range(max_segments):
                s = segments[int(i*step)][0]
                e = segments[min(int((i+1)*step)-1, len(segments)-1)][1]
                segs.append((s, e))
            segments = segs

        segments = [(s, e) for (s, e) in segments if e-s >= 0.5]

        # Fallback: uniform split
        if len(segments) < 2:
            segments = []
            dur = probe_duration(src) or 0
            if dur > 0:
                step = dur / max_segments
                for i in range(max_segments):
                    a = i*step
                    b = min((i+1)*step, dur)
                    if b - a >= 0.5:
                        segments.append((a, b))

        return segments
    except Exception:
        return []


def extract_segment_frame(src, t):
    """Extract frame at time t."""
    try:
        fd, fp = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        r = subprocess.run(
            ["ffmpeg", "-y", "-ss", str(t), "-i", str(src), "-frames:v", "1", "-q:v", "2", fp],
            capture_output=True, text=True, timeout=60
        )
        if os.path.exists(fp) and os.path.getsize(fp) > 0:
            return fp
        return None
    except Exception:
        return None


def frame_quality_score(path):
    """Score a representative frame using cheap local quality signals."""
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        return 0.0

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sharpness = min(float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 500.0, 1.0)
    brightness = float(gray.mean()) / 255.0
    exposure = max(0.0, 1.0 - abs(brightness - 0.5) / 0.5)
    clipped = float(((gray <= 2) | (gray >= 253)).mean())
    return max(0.0, min(1.0, 0.45 * sharpness + 0.45 * exposure + 0.10 * (1.0 - clipped)))


def select_best_segments(
    segments,
    scores,
    max_duration=30.0,
    max_segments=5,
    max_clip_duration=8.0,
):
    """Pick highest-scoring non-overlapping clips and return them chronologically."""
    candidates = []
    for (start, end), score in zip(segments, scores):
        start, end = float(start), float(end)
        duration = max(0.0, end - start)
        if duration <= 0:
            continue
        take = min(duration, float(max_clip_duration))
        clip_start = start + max(0.0, (duration - take) / 2.0)
        candidates.append({
            "start": round(clip_start, 3),
            "end": round(clip_start + take, 3),
            "take": round(take, 3),
            "score": round(float(score), 4),
        })

    chosen = []
    used_duration = 0.0
    for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True):
        if len(chosen) >= int(max_segments):
            break
        if used_duration + candidate["take"] > float(max_duration) + 1e-6:
            continue
        if any(
            candidate["start"] < other["end"] and other["start"] < candidate["end"]
            for other in chosen
        ):
            continue
        chosen.append(candidate)
        used_duration += candidate["take"]

    return sorted(chosen, key=lambda item: item["start"])


def build_segment_filter(segments, video_filter, include_audio=True):
    """Build an ffmpeg filter graph that trims and concatenates selected clips."""
    parts = []
    inputs = []
    video_labels = [f"vin{index}" for index in range(len(segments))]
    audio_labels = [f"ain{index}" for index in range(len(segments))]

    if len(segments) > 1:
        parts.append(f"[0:v]split={len(segments)}" + "".join(f"[{label}]" for label in video_labels))
        if include_audio:
            parts.append(f"[0:a]asplit={len(segments)}" + "".join(f"[{label}]" for label in audio_labels))

    for index, segment in enumerate(segments):
        start = float(segment["start"])
        end = float(segment["end"])
        parts.append(
            f"[{video_labels[index] if len(segments) > 1 else '0:v'}]"
            f"trim=start={start:.3f}:end={end:.3f},setpts=PTS-STARTPTS,"
            f"{video_filter}[v{index}]"
        )
        inputs.append(f"[v{index}]")
        if include_audio:
            parts.append(
                f"[{audio_labels[index] if len(segments) > 1 else '0:a'}]"
                f"atrim=start={start:.3f}:end={end:.3f},"
                f"asetpts=PTS-STARTPTS[a{index}]"
            )
            inputs.append(f"[a{index}]")

    audio_flag = 1 if include_audio else 0
    parts.append(
        "".join(inputs)
        + f"concat=n={len(segments)}:v=1:a={audio_flag}[outv]"
        + ("[outa]" if include_audio else "")
    )
    return ";".join(parts)


def concat_segments(src, chosen, w=1080, h=1920):
    """Concatenate selected segments into one video."""
    try:
        tmp_files = []
        for i, seg in enumerate(chosen):
            fd, fp = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)
            cmd = [
                "ffmpeg", "-y",
                "-ss", f"{seg['start']:.2f}", "-i", str(src),
                "-t", f"{seg['take']:.2f}",
                "-vf", f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},fps=30",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                "-an", fp
            ]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if r.returncode == 0 and os.path.exists(fp) and os.path.getsize(fp) > 0:
                tmp_files.append(fp)
            else:
                try:
                    os.unlink(fp)
                except Exception:
                    pass

        if not tmp_files:
            return None

        listfile = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8")
        for f in tmp_files:
            listfile.write(f"file '{f.replace(chr(92), '/')}'\n")
        listfile.close()

        out_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", listfile.name,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p",
            "-an", out_path
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)

        for f in tmp_files:
            try:
                os.unlink(f)
            except Exception:
                pass

        try:
            os.unlink(listfile.name)
        except Exception:
            pass

        if r.returncode != 0 or not os.path.exists(out_path) or os.path.getsize(out_path) <= 0:
            try:
                os.unlink(out_path)
            except Exception:
                pass
            return None

        return out_path
    except Exception:
        try:
            os.unlink(out_path)
        except Exception:
            pass
        return None


if __name__ == "__main__":
    import sys
    print("video_tools: Utility module for video processing.")
    if len(sys.argv) > 1:
        print("Scenes:", detect_scenes(sys.argv[1]))
