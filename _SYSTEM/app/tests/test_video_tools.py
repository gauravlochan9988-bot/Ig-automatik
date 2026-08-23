import json
import unittest
from unittest.mock import patch

from ig_automatik.core import video_tools


class VideoToolsTests(unittest.TestCase):
    def test_probe_video_info_extracts_source_fps_audio_and_geometry(self):
        payload = {
            "format": {"duration": "22.15"},
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "hevc",
                    "width": 3840,
                    "height": 2160,
                    "avg_frame_rate": "60000/1001",
                    "r_frame_rate": "60000/1001",
                },
                {"codec_type": "audio", "codec_name": "aac"},
            ],
        }
        result = type("R", (), {"returncode": 0, "stdout": json.dumps(payload)})()
        with patch.object(video_tools.subprocess, "run", return_value=result):
            info = video_tools.probe_video_info("input.mov")

        self.assertEqual(info["width"], 3840)
        self.assertEqual(info["height"], 2160)
        self.assertAlmostEqual(info["fps"], 59.94, places=2)
        self.assertTrue(info["has_audio"])

    def test_social_fps_plan_preserves_standard_rates_and_normalizes_high_fps(self):
        self.assertEqual(video_tools.plan_social_fps({"fps": 24.0})["output_fps"], 24)
        self.assertEqual(video_tools.plan_social_fps({"fps": 25.0})["output_fps"], 25)
        self.assertEqual(video_tools.plan_social_fps({"fps": 30.0})["output_fps"], 30)
        self.assertEqual(video_tools.plan_social_fps({"fps": 60.0})["output_fps"], 30)
        self.assertTrue(video_tools.plan_social_fps({"fps": 60.0})["slow_motion_candidate"])

    def test_segment_filter_accepts_master_output_geometry(self):
        graph = video_tools.build_segment_filter(
            [{"start": 0.0, "end": 4.0}],
            "scale=1440:2560,crop=1440:2560",
            include_audio=False,
            ken_burns=True,
            output_width=1440,
            output_height=2560,
            output_fps=24,
        )
        self.assertIn("s=1440x2560", graph)
        self.assertIn("fps=24", graph)

    def test_social_derivative_command_uses_master_as_only_input(self):
        from ig_automatik.core import grading_engine as engine
        cmd = engine._build_social_derivative_command(
            {"reel_max_duration": 30},
            "master_A.mp4", "post_A.part.mp4", include_audio=True,
            use_gpu=False, output_fps=24,
        )
        self.assertEqual(cmd[cmd.index("-i") + 1], "master_A.mp4")
        self.assertIn("scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920", cmd)
        self.assertEqual(cmd[cmd.index("-r") + 1], "24")
        self.assertIn("-c:a", cmd)
        self.assertIn("aac", cmd)

    def test_ken_burns_emits_one_output_frame_per_source_frame(self):
        graph = video_tools.build_segment_filter(
            [{"start": 0.0, "end": 4.0}],
            "scale=1080:1920,crop=1080:1920",
            include_audio=False,
            ken_burns=True,
            output_width=1080,
            output_height=1920,
            output_fps=24,
        )
        # zoompan's d is output frames PER input frame, never total duration.
        self.assertIn("zoompan=z='min(zoom+0.0015,1.15)':d=1:s=1080x1920:fps=24", graph)
        self.assertIn("fps=24,setsar=1,format=yuv420p", graph)


if __name__ == "__main__":
    unittest.main()
