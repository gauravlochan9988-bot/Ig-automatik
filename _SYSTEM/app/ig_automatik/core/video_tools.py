"""Video utility functions for scene detection and frame extraction."""

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

import cv2


def _parse_fps(value):
    """Turn ffprobe's rational frame rate (e.g. 60000/1001) into a float."""
    try:
        text = str(value or "0")
        if "/" in text:
            numerator, denominator = text.split("/", 1)
            denominator = float(denominator)
            return float(numerator) / denominator if denominator else 0.0
        return float(text)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def probe_video_info(path):
    """Read source duration, FPS, geometry, codecs and audio in one ffprobe call."""
    fallback = {
        "duration": 0.0, "fps": 0.0, "width": 0, "height": 0,
        "video_codec": None, "audio_codec": None, "has_audio": False,
    }
    try:
        r = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries",
                "format=duration:stream=codec_type,codec_name,width,height,avg_frame_rate,r_frame_rate",
                "-of", "json", str(path),
            ],
            capture_output=True, text=True, timeout=60,
        )
        if r.returncode != 0:
            return fallback
        data = json.loads(r.stdout)
        streams = data.get("streams") or []
        video = next((s for s in streams if s.get("codec_type") == "video"), {})
        audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
        fps = _parse_fps(video.get("avg_frame_rate")) or _parse_fps(video.get("r_frame_rate"))
        return {
            "duration": float((data.get("format") or {}).get("duration") or 0.0),
            "fps": round(fps, 3),
            "width": int(video.get("width") or 0),
            "height": int(video.get("height") or 0),
            "video_codec": video.get("codec_name"),
            "audio_codec": audio.get("codec_name") if audio else None,
            "has_audio": audio is not None,
        }
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return fallback


def plan_social_fps(video_info):
    """Choose output FPS deliberately; preserve normal source cadence.

    24/25/30 fps remain unchanged. 50/60 fps is normalized to 30 for regular
    Reel delivery but is marked as a slow-motion candidate for a future edit
    plan, rather than silently pretending its source cadence never existed.
    """
    fps = float((video_info or {}).get("fps") or 0.0)
    if fps <= 0:
        return {"source_fps": 0.0, "output_fps": 30, "slow_motion_candidate": False}
    if fps >= 45:
        return {"source_fps": fps, "output_fps": 30, "slow_motion_candidate": True}
    standard = min((24, 25, 30), key=lambda target: abs(fps - target))
    return {"source_fps": fps, "output_fps": standard, "slow_motion_candidate": False}


def probe_duration(path):
    """Get video duration in seconds."""
    duration = probe_video_info(path).get("duration", 0.0)
    return duration if duration > 0 else None


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


def build_segment_filter(
    segments,
    video_filter,
    include_audio=True,
    transition="none",
    transition_duration=0.5,
    ken_burns=False,
    zoom_speed="0.0015",
    output_width=1080,
    output_height=1920,
    output_fps=30,
):
    """Build an ffmpeg filter graph that trims, animates, transitions, and concatenates clips."""
    parts = []
    video_labels = [f"vin{index}" for index in range(len(segments))]
    audio_labels = [f"ain{index}" for index in range(len(segments))]

    if len(segments) > 1:
        parts.append(f"[0:v]split={len(segments)}" + "".join(f"[{label}]" for label in video_labels))
        if include_audio:
            parts.append(f"[0:a]asplit={len(segments)}" + "".join(f"[{label}]" for label in audio_labels))

    # Process each segment
    for index, segment in enumerate(segments):
        start = float(segment["start"])
        end = float(segment["end"])
        duration = end - start
        
        # Build segment video filter
        v_chain = [f"trim=start={start:.3f}:end={end:.3f}", "setpts=PTS-STARTPTS"]
        
        if ken_burns:
            # zoompan's `d` is output frames PER input frame — it must be 1
            # for temporal-duration preservation. The zoom evolves per frame.
            v_chain.append(
                f"zoompan=z='min(zoom+{zoom_speed},1.15)':d=1:s={int(output_width)}x{int(output_height)}:fps={int(output_fps)}"
            )

        v_chain.extend([f"fps={int(output_fps)}", "setsar=1", "format=yuv420p"])
        v_chain.append(video_filter)
        parts.append(f"[{video_labels[index] if len(segments) > 1 else '0:v'}]{','.join(v_chain)}[v{index}]")

        if include_audio:
            parts.append(
                f"[{audio_labels[index] if len(segments) > 1 else '0:a'}]"
                f"atrim=start={start:.3f}:end={end:.3f},"
                f"asetpts=PTS-STARTPTS[a{index}]"
            )

    # Combine segments: using xfade/acrossfade if transitions requested, else simple concat
    use_xfade = transition not in ("none", "", None) and len(segments) > 1 and transition != "concat"
    
    if use_xfade:
        t_dur = float(transition_duration)
        # Chain video with xfade
        prev_v = "[v0]"
        current_offset = float(segments[0]["end"]) - float(segments[0]["start"]) - t_dur
        for idx in range(1, len(segments)):
            seg_dur = float(segments[idx]["end"]) - float(segments[idx]["start"])
            out_v = f"[vx{idx}]" if idx < len(segments) - 1 else "[outv]"
            parts.append(
                f"{prev_v}[v{idx}]xfade=transition={transition}:duration={t_dur:.2f}:offset={max(0.0, current_offset):.3f}{out_v}"
            )
            prev_v = out_v
            current_offset += seg_dur - t_dur

        if include_audio:
            prev_a = "[a0]"
            for idx in range(1, len(segments)):
                out_a = f"[ax{idx}]" if idx < len(segments) - 1 else "[outa]"
                parts.append(
                    f"{prev_a}[a{idx}]acrossfade=d={t_dur:.2f}{out_a}"
                )
                prev_a = out_a
    else:
        inputs = []
        for index in range(len(segments)):
            inputs.append(f"[v{index}]")
            if include_audio:
                inputs.append(f"[a{index}]")
        audio_flag = 1 if include_audio else 0
        parts.append(
            "".join(inputs)
            + f"concat=n={len(segments)}:v=1:a={audio_flag}[outv]"
            + ("[outa]" if include_audio else "")
        )

    return ";".join(parts)


if __name__ == "__main__":
    import sys
    print("video_tools: Utility module for video processing.")
    if len(sys.argv) > 1:
        print("Scenes:", detect_scenes(sys.argv[1]))
