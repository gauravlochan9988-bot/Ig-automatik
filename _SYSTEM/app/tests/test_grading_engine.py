import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from ig_automatik.core import grading_engine as engine
from ig_automatik.core.media import Media
from ig_automatik.core import video_tools
from ig_automatik.config import paths as project_paths
from ig_automatik.config import Config
import watch


class GradingEngineTests(unittest.TestCase):
    def test_project_root_is_found_from_a_nested_system_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for marker in ("1_EINGANG", "2_FERTIG", "3_ARCHIV"):
                (root / marker).mkdir()
            nested = root / "_SYSTEM" / "app"
            nested.mkdir(parents=True)

            self.assertEqual(
                project_paths.find_project_root(nested).resolve(), root.resolve()
            )

    def test_project_root_is_found_from_a_flat_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for marker in ("1_EINGANG", "2_FERTIG", "3_ARCHIV"):
                (root / marker).mkdir()

            self.assertEqual(
                project_paths.find_project_root(root).resolve(), root.resolve()
            )

    def test_single_archive_folder_does_not_become_a_false_project_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / "_SYSTEM" / "app"
            (nested / "3_ARCHIV" / "MASTERS").mkdir(parents=True)

            self.assertNotEqual(project_paths.find_project_root(nested), nested.resolve())

    def test_fresh_system_folder_resolves_to_its_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            system = Path(tmp) / "_SYSTEM"
            system.mkdir()

            self.assertEqual(
                project_paths.find_project_root(system).resolve(),
                Path(tmp).resolve(),
            )

    def test_lock_file_lives_inside_the_real_project_system_folder(self):
        self.assertEqual(watch.ROOT.name, "IG-AUTOMATIK")
        self.assertEqual(watch.INPUT, watch.ROOT / "1_EINGANG")
        self.assertEqual(watch.LOCK.parent.name, "_SYSTEM")
        self.assertEqual(watch.LOCK.name, "watchdog.lock")
        self.assertEqual(watch.LOCK.parent.parent, watch.ROOT)

    def test_logger_uses_project_system_log_dir_not_nested_app_system_dir(self):
        from ig_automatik.utils import logging_utils
        self.assertEqual(logging_utils.PROJECT_ROOT.name, "IG-AUTOMATIK")
        log_system = logging_utils.system_dir(logging_utils.PROJECT_ROOT)
        self.assertEqual(log_system.name, "_SYSTEM")
        self.assertNotIn("_SYSTEM/app/_SYSTEM", str(log_system).replace("\\", "/"))

    def test_best_segments_are_selected_by_score_and_returned_chronologically(self):
        segments = [(0.0, 4.0), (10.0, 14.0), (20.0, 24.0), (30.0, 34.0)]
        scores = [0.2, 0.95, 0.8, 0.9]

        chosen = video_tools.select_best_segments(
            segments, scores, max_duration=8.0, max_segments=2
        )

        self.assertEqual(chosen, [
            {"start": 10.0, "end": 14.0, "take": 4.0, "score": 0.95},
            {"start": 30.0, "end": 34.0, "take": 4.0, "score": 0.9},
        ])

    def test_long_segment_is_trimmed_to_clip_limit(self):
        chosen = video_tools.select_best_segments(
            [(0.0, 20.0)], [0.8], max_duration=6.0, max_segments=1, max_clip_duration=6.0
        )

        self.assertEqual(chosen[0]["take"], 6.0)
        self.assertEqual(chosen[0]["start"], 7.0)
        self.assertEqual(chosen[0]["end"], 13.0)

    def test_segment_filter_contains_video_and_audio_concat(self):
        filter_graph = video_tools.build_segment_filter(
            [{"start": 1.0, "end": 3.5}, {"start": 8.0, "end": 10.0}],
            "scale=1080:1920",
            include_audio=True,
        )

        self.assertIn("trim=start=1.000:end=3.500", filter_graph)
        self.assertIn("atrim=start=1.000:end=3.500", filter_graph)
        self.assertIn("concat=n=2:v=1:a=1", filter_graph)

    def test_reel_command_uses_selected_segments_when_provided(self):
        command = engine._build_reel_command(
            {"reel_max_duration": 30},
            Path("input.mov"),
            Path("output.mp4"),
            "A",
            selected_segments=[{"start": 2.0, "end": 5.0, "take": 3.0}],
            include_audio=True,
        )

        self.assertIn("-filter_complex", command)
        self.assertIn("[outv]", command)
        self.assertIn("[outa]", command)

    def test_reel_manifest_records_editing_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from ig_automatik.core import templates
            tpl = templates.get_template("miami_vibes")
            manifest = engine._save_reel_manifest(
                root,
                "clip",
                source_duration=84.8,
                selected_segments=[{"start": 2.0, "end": 6.5}],
                provider="openrouter",
                outputs={"A": "out_A.mp4", "B": "out_B.mp4"},
                template=tpl,
                lut_name="True Cinematic.cube",
            )

            data = __import__("json").loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(data["source_duration"], 84.8)
            self.assertEqual(data["selected_segments"][0]["start"], 2.0)
            self.assertEqual(data["editing_provider"], "openrouter")
            self.assertEqual(data["template"], "Miami Summer Vibe")

    def test_video_best_clips_flag_controls_selection(self):
        self.assertEqual(
            engine._select_reel_segments(
                {"video_best_clips": False, "reel_max_duration": 30},
                [(0.0, 10.0), (20.0, 30.0)],
                [0.9, 0.1],
            ),
            None,
        )

    def test_watchdog_deduplicates_created_and_modified_events(self):
        handler = watch.Handler()
        path = Path("same_video.mp4")
        event = type("Event", (), {"is_directory": False, "src_path": str(path)})()

        with patch.object(watch.logger, "info") as log_info:
            handler.on_created(event)
            handler.on_modified(event)

        self.assertTrue(handler.take_pending())
        handler.clear_pending()
        self.assertFalse(handler.take_pending())
        self.assertEqual(log_info.call_count, 1)

    def test_output_stem_is_unique_when_existing_outputs_are_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "POSTS").mkdir()
            (root / "POSTS" / "clip_A.jpg").write_bytes(b"old")

            self.assertEqual(engine._unique_output_stem(root, "clip"), "clip_2")

    def test_saturation_adjustment_preserves_float_precision(self):
        base = np.array([[[0.21, 0.43, 0.67], [0.35, 0.52, 0.78]]], dtype=np.float32)
        result = engine._hsv_float_sat(base, 1.15)

        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(np.all(result >= 0.0))
        self.assertTrue(np.all(result <= 1.0))
        self.assertGreater(float(np.abs(result - base).max()), 0.001)

        close_colors = np.array(
            [[[0.4000, 0.5000, 0.6000], [0.4015, 0.5015, 0.6015]]],
            dtype=np.float32,
        )
        close_result = engine._hsv_float_sat(close_colors, 1.15)
        self.assertGreater(
            float(np.abs(close_result[0, 1] - close_result[0, 0]).max()), 0.0005
        )

    def test_highlight_protection_reduces_clipping_without_flattening_midtones(self):
        base = np.array([[[0.50, 0.50, 0.50], [0.98, 0.98, 0.98]]], dtype=np.float32)
        result = engine._protect_highlights(base)

        self.assertLess(float(result[0, 1].max()), 0.98)
        self.assertAlmostEqual(float(result[0, 0].mean()), 0.50, places=3)
        self.assertTrue(np.all(result >= 0.0))
        self.assertTrue(np.all(result <= 1.0))

    def test_cinematic_grade_also_protects_highlights(self):
        base = np.array(
            [[[0.98, 0.96, 0.94], [0.92, 0.88, 0.84], [0.45, 0.40, 0.35]]],
            dtype=np.float32,
        )
        result = engine.grade_variant_b(
            base,
            scene={"cinematic": {"contrast": 8, "sat": 5}, "saturation": 80},
        )

        self.assertLess(float(result[:, :2].max()), 0.995)
        self.assertTrue(np.all(result >= 0.0))
        self.assertTrue(np.all(result <= 1.0))

    def test_reel_output_uses_a_temporary_part_path(self):
        final_path = Path("output.mp4")
        temporary_path = engine._reel_part_path(final_path)

        self.assertEqual(temporary_path.name, "output.part.mp4")
        self.assertNotEqual(temporary_path, final_path)

    def test_natural_grade_keeps_skin_like_colors_reasonably_balanced(self):
        skin = np.full((20, 20, 3), [0.72, 0.48, 0.36], dtype=np.float32)
        result = engine.grade_variant_a(
            skin,
            scene={"natural": {"contrast": 3, "sat": 1}, "saturation": 40},
        )

        red_blue_before = float(skin[..., 0].mean() - skin[..., 2].mean())
        red_blue_after = float(result[..., 0].mean() - result[..., 2].mean())
        self.assertGreater(red_blue_after, 0.0)
        self.assertLess(abs(red_blue_after - red_blue_before), 0.15)

    def test_crop_anchors_to_a_subject_box_on_the_left(self):
        rgb = np.zeros((100, 300, 3), dtype=np.float32)
        plan = {"subject_box": [0.05, 0.5, 0.2, 0.2]}
        ax, ay = engine._subject_anchor(rgb, plan)
        self.assertLess(ax, 0.25)
        self.assertAlmostEqual(ay, 0.6, places=1)

        crop = engine._crop_to_ratio(rgb, "9:16", anchor=(ax, ay))
        self.assertEqual(crop.shape[0], 100)
        self.assertLess(crop.shape[1], 100)

    def test_crop_without_anchor_keeps_legacy_behaviour(self):
        rgb = np.zeros((200, 100, 3), dtype=np.float32)
        crop = engine._crop_to_ratio(rgb, "1:1")
        self.assertEqual(crop.shape, (100, 100, 3))

        anchor = engine._subject_anchor(rgb, {"scene_type": "general"})
        self.assertIsNone(anchor)

    def test_saliency_anchor_finds_subject_in_upper_left(self):
        # A bright detailed blob top-left, uniform elsewhere. The saliency
        # anchor must point there (upper half bias + detail centroid).
        rgb = np.full((100, 200, 3), 0.1, dtype=np.float32)
        rgb[15:45, 20:60] = 0.9  # detailed bright region, upper-left
        rgb[15:45, 20:60] += np.random.default_rng(0).random((30, 40, 3)).astype(np.float32) * 0.05
        ax, ay = engine._saliency_anchor(rgb)
        self.assertIsNotNone(ax)
        self.assertLess(ax, 0.4)
        self.assertLess(ay, 0.6)

    def test_saliency_anchor_returns_none_for_uniform_image(self):
        rgb = np.full((50, 50, 3), 0.5, dtype=np.float32)
        self.assertIsNone(engine._saliency_anchor(rgb))

    def test_person_subject_biases_anchor_into_upper_third(self):
        # Person detected (vision) but no subject_box and no Haar face match:
        # the anchor must sit in the upper third so the head stays in frame.
        rgb = np.full((100, 80, 3), 0.3, dtype=np.float32)
        plan = {"scene_type": "general", "main_subject": "woman in sparkly outfit with cowboy hat"}
        ax, ay = engine._subject_anchor(rgb, plan)
        self.assertIsNotNone(ax)
        self.assertLess(ay, 0.4)  # upper third, not mid-frame

        # Non-person scenes without vision box/face fall to saliency when
        # there is detectable detail; a perfectly uniform image has no anchor.
        detailed = np.full((100, 80, 3), 0.3, dtype=np.float32)
        detailed[10:50, 10:40] = 0.9
        anchor = engine._subject_anchor(detailed, {"scene_type": "landscape"})
        self.assertIsNotNone(anchor)
        self.assertIsNone(engine._subject_anchor(rgb, {"scene_type": "landscape"}))

    def test_subject_preserving_crop_keeps_full_body_regions(self):
        rgb = np.zeros((150, 100, 3), dtype=np.float32)
        plan = {
            "scene_type": "portrait",
            "main_subject": "full body person",
            "composition_plan": {
                "subject_type": "full_body_person",
                "protected_regions": [
                    {"name": "head", "box": [0.25, 0.10, 0.50, 0.12], "required": True},
                    {"name": "body", "box": [0.20, 0.22, 0.60, 0.50], "required": True},
                    {"name": "feet", "box": [0.22, 0.72, 0.56, 0.12], "required": True},
                ],
                "preferred_position": "slightly_upper_center",
                "allow_zoom": False,
                "preserve_environment": True,
            },
        }

        result = engine.select_safe_crop(rgb, "4:5", plan=plan)

        self.assertTrue(result["safe"])
        self.assertEqual(result["missing"], [])
        self.assertEqual(result["crop"].shape[:2], (125, 100))

    def test_subject_preserving_crop_uses_safe_padding_when_ratio_is_impossible(self):
        rgb = np.zeros((150, 100, 3), dtype=np.float32)
        plan = {
            "scene_type": "portrait",
            "main_subject": "full body person",
            "composition_plan": {
                "subject_type": "full_body_person",
                "protected_regions": [
                    {"name": "head", "box": [0.20, 0.0, 0.60, 0.10], "required": True},
                    {"name": "feet", "box": [0.20, 0.90, 0.60, 0.10], "required": True},
                ],
                "allow_zoom": False,
                "preserve_environment": True,
            },
        }

        result = engine.select_safe_crop(rgb, "4:5", plan=plan)

        self.assertTrue(result["safe"])
        self.assertEqual(result["mode"], "padded_safe")
        self.assertEqual(result["missing"], [])
        self.assertFalse(result["feasible"])
        self.assertEqual(result["crop"].shape[:2], (150, 120))

    def test_person_words_detection(self):
        self.assertTrue(engine._is_person_subject({"scene_type": "portrait"}))
        self.assertTrue(engine._is_person_subject({"main_subject": "woman with hat"}))
        self.assertFalse(engine._is_person_subject({"scene_type": "landscape", "main_subject": "mountains"}))

    def test_batch_report_text_lists_failures(self):
        text = engine._build_batch_report_text(["a.jpg", "b.jpg"], [("c.jpg", "boom")])
        self.assertIn("OK: 2", text)
        self.assertIn("Failed: 1", text)
        self.assertIn("c.jpg", text)
        self.assertIn("boom", text)

    def test_batch_report_writes_a_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manifests").mkdir()
            cfg = {"manifests_folder": str(root / "manifests")}

            path = engine._write_batch_report(cfg, ["x.jpg"], [])
            self.assertTrue(path.exists())
            self.assertIn("batch_report_", path.name)
            self.assertIn("OK: 1", path.read_text(encoding="utf-8"))

    def test_identical_file_is_detected_as_already_archived(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "photo.jpg"
            arc = root / "archive"
            arc.mkdir()
            src.write_bytes(b"same-bytes-123")
            (arc / "photo.jpg").write_bytes(b"same-bytes-123")

            self.assertTrue(engine._already_archived({"processed_folder": str(arc)}, src))

            (arc / "photo.jpg").write_bytes(b"different-bytes")
            self.assertFalse(engine._already_archived({"processed_folder": str(arc)}, src))

    def test_archive_index_finds_duplicate_by_size_and_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "video.mp4"
            arc = root / "archive"
            arc.mkdir()
            src.write_bytes(b"abc")
            (arc / "copy1.mp4").write_bytes(b"abc")
            (arc / "other.jpg").write_bytes(b"xyz")

            index = engine._build_archive_index({"processed_folder": str(arc)})
            self.assertTrue(engine._already_archived({"processed_folder": str(arc)}, src, index=index))

            src.write_bytes(b"different!")
            self.assertFalse(engine._already_archived({"processed_folder": str(arc)}, src, index=index))

    def test_auto_regrade_on_qa_failure_recovers_blown_highlights(self):
        # Create an image that easily clips if graded aggressively
        # (bright highlights near 0.95-0.99 with high contrast intent)
        bright_crop = np.full((100, 100, 3), 0.92, dtype=np.float32)
        bright_crop[20:80, 20:80] = 0.98

        scene = {"natural": {"contrast": 4.0, "sat": 2.0}, "tags": []}
        intent = {"contrast": 1.0, "saturation": 0.8, "warmth": 0.5}

        graded_a, qa_a, retries_a = engine._grade_variant_with_qa(
            "A", bright_crop, scene=scene, intent=intent, ratio="1:1", max_retries=2
        )

        self.assertTrue(qa_a["pass"])
        self.assertLess(qa_a["checks"]["clipped_high_pct"], 5.0)
        self.assertGreaterEqual(retries_a, 1)

    def test_auto_regrade_variant_b_recovers_from_severe_clipping(self):
        bright_crop = np.full((100, 100, 3), 0.94, dtype=np.float32)
        bright_crop[10:90, 10:90] = 0.99

        scene = {"cinematic": {"contrast": 5.0, "sat": 3.0}, "tags": []}
        intent = {"contrast": 1.0, "saturation": 0.9, "warmth": 0.8}

        graded_b, qa_b, retries_b = engine._grade_variant_with_qa(
            "B", bright_crop, scene=scene, intent=intent, ratio="1:1", max_retries=2
        )

        self.assertTrue(qa_b["pass"])
        self.assertLess(qa_b["checks"]["clipped_high_pct"], 5.0)

    def test_templates_load_and_resolve(self):
        from ig_automatik.core import templates
        tpl = templates.get_template("miami_vibes")
        self.assertEqual(tpl["name"], "Miami Summer Vibe")
        self.assertIn("xfade", tpl["transitions"])
        self.assertTrue(tpl["ken_burns"])

        # Auto selection by scene
        matched = templates.select_template_for_scene({"scene_type": "sunset", "main_subject": "beach sunset"})
        self.assertEqual(matched["id"], "miami_vibes")

        night_matched = templates.select_template_for_scene({"scene_type": "night", "main_subject": "bar party"})
        self.assertEqual(night_matched["id"], "moody_night")

    def test_segment_filter_with_transitions(self):
        segments = [{"start": 1.0, "end": 4.0}, {"start": 8.0, "end": 11.0}]
        filter_graph = video_tools.build_segment_filter(
            segments,
            "scale=1080:1920",
            include_audio=True,
            transition="fade",
            transition_duration=0.5,
            ken_burns=True,
        )
        self.assertIn("xfade=transition=fade", filter_graph)
        self.assertIn("acrossfade=d=0.5", filter_graph)
        self.assertIn("zoompan", filter_graph)

    def test_caption_file_generation_and_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "output" / "POSTS"
            out_dir.mkdir(parents=True)
            
            caption_data = {
                "hook": "Sunset magic in Florida ✨",
                "caption": "Golden hour never looked better. Feeling the warm breeze and soaking in every moment.",
                "hashtags": ["#MiamiVibes", "#FloridaSunset", "#GoldenHour", "#TravelReels", "#Wanderlust"]
            }
            
            caption_path = engine._save_caption_file(out_dir, "my_photo", caption_data)
            self.assertTrue(caption_path.exists())
            text = caption_path.read_text(encoding="utf-8")
            self.assertIn("Sunset magic in Florida ✨", text)
            self.assertIn("#MiamiVibes", text)
            self.assertIn("#GoldenHour", text)

    def test_vision_payload_parses_instagram_captions(self):
        body = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "scene_type": "sunset",
                        "main_subject": "beach",
                        "subject_importance": 0.8,
                        "subject_box": [0.1, 0.2, 0.6, 0.7],
                        "environment_importance": 0.9,
                        "sky_importance": 0.9,
                        "preserve_colors": ["orange", "blue"],
                        "grading_intent": {"warmth": 0.4, "contrast": 0.2, "saturation": 0.3},
                        "creative_direction": {
                            "preserve": ["face", "skin tones", "sunset sky", "ocean"],
                            "light_mood": "golden_hour",
                            "style_family": "warm_travel",
                            "composition": {
                                "post_4_5": {"subject_priority": 0.82, "environment_priority": 0.70},
                                "story_9_16": {"subject_priority": 0.93, "environment_priority": 0.45}
                            }
                        },
                        "instagram": {
                            "hook": "Chasing sunsets in Miami 🌅",
                            "caption": "Nothing beats this Florida sky.",
                            "hashtags": ["MiamiLife", "#SunsetLovers", "VacationMode"]
                        }
                    })
                }
            }],
            "usage": {"total_tokens": 150}
        }
        
        with patch.object(engine.vision.Config, "load_env", return_value={"OPENROUTER_API_KEY": "test-key"}), \
             patch("urllib.request.urlopen") as mock_url:
            mock_resp = mock_url.return_value.__enter__.return_value
            mock_resp.read.return_value = json.dumps(body).encode("utf-8")
            mock_resp.__enter__.return_value = mock_resp
            
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                f_path = Path(f.name)
                Image.new("RGB", (100, 100), (200, 100, 50)).save(f_path)
            
            try:
                res = engine.vision.analyze(f_path)
                self.assertIsNotNone(res)
                self.assertIn("instagram", res)
                self.assertEqual(res["instagram"]["hook"], "Chasing sunsets in Miami 🌅")
                self.assertEqual(res["instagram"]["hashtags"], ["#MiamiLife", "#SunsetLovers", "#VacationMode"])
                self.assertEqual(res["creative_direction"]["light_mood"], "golden_hour")
                self.assertEqual(res["creative_direction"]["preserve"], ["face", "skin tones", "sunset sky", "ocean"])
                self.assertAlmostEqual(
                    res["creative_direction"]["composition"]["story_9_16"]["subject_priority"], 0.93
                )
            finally:
                f_path.unlink(missing_ok=True)

    def test_apply_lut_to_photo_in_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.jpg"
            Image.new("RGB", (100, 100), (120, 150, 180)).save(source)
            cfg = {
                "output_width_post": 1080,
                "output_width_story": 1080,
                "export_quality": 95,
                "produce_ig": True,
                "produce_archives": False,
                "produce_formats": ["POSTS"],
            }
            
            with patch.object(engine.vision, "is_enabled", return_value=False):
                results = engine.process_photo(cfg, source, root / "output")
                
            self.assertEqual(len(results), 1)
            self.assertTrue(Path(results[0]["files"]["A"]["ig"]).exists())
            self.assertTrue(Path(results[0]["files"]["B"]["ig"]).exists())
            if "lut_b" in results[0]["plan"]:
                self.assertTrue(results[0]["plan"]["lut_b"].endswith(".cube"))

    def test_batch_report_text_lists_skipped_duplicates(self):
        text = engine._build_batch_report_text(["a.jpg"], [], ["b.jpg"])
        self.assertIn("OK: 1", text)
        self.assertIn("Duplikate", text)
        self.assertIn("b.jpg", text)

    def test_config_bool_strings_are_parsed_correctly(self):
        cfg = Config._validate(
            {
                "produce_archives": "false",
                "produce_masters": "true",
                "auto_move_sources": "0",
                "produce_ig": "true",
                "safe_edit_only": "no",
                "video_kenburns": "false",
                "video_best_clips": "false",
                "export_quality": 95,
                "reel_max_duration": 30,
                "best_clips_max_segments": 15,
                "produce_formats": ["POSTS"],
                "output_width_post": 1080,
                "output_width_story": 1080,
            }
        )
        self.assertIs(cfg["produce_archives"], False)
        self.assertIs(cfg["produce_masters"], True)
        self.assertIs(cfg["auto_move_sources"], False)
        self.assertIs(cfg["safe_edit_only"], False)
        self.assertIs(cfg["produce_ig"], True)
        self.assertIs(cfg["video_kenburns"], False)
        self.assertIs(cfg["video_best_clips"], False)

    def test_reel_command_uses_nvenc_when_requested(self):
        command = engine._build_reel_command(
            {"reel_max_duration": 30},
            Path("input.mov"),
            Path("output.mp4"),
            "A",
            use_gpu=True,
        )
        self.assertIn("h264_nvenc", command)
        self.assertIn("-preset", command)
        self.assertIn("p5", command)
        self.assertIn("-cq", command)
        self.assertNotIn("libx264", command)
        self.assertNotIn("-crf", command)

    def test_reel_command_falls_back_to_cpu_when_not_requested(self):
        command = engine._build_reel_command(
            {"reel_max_duration": 30},
            Path("input.mov"),
            Path("output.mp4"),
            "A",
            use_gpu=False,
        )
        self.assertIn("libx264", command)
        self.assertIn("-crf", command)
        self.assertNotIn("h264_nvenc", command)

    def test_nvenc_detected_by_real_encode_probe(self):
        ok = type("Result", (), {"returncode": 0})()
        with patch.object(engine.subprocess, "run", return_value=ok):
            engine._nvenc_available.cache_clear()
            self.assertTrue(engine._nvenc_available())
            engine._nvenc_available.cache_clear()

        fail = type("Result", (), {"returncode": 1})()
        with patch.object(engine.subprocess, "run", return_value=fail):
            engine._nvenc_available.cache_clear()
            self.assertFalse(engine._nvenc_available())
            engine._nvenc_available.cache_clear()

    def test_qa_clipping_percentages_are_pixel_percentages(self):
        black = np.zeros((10, 10, 3), dtype=np.float32)
        white = np.ones((10, 10, 3), dtype=np.float32)

        black_qa = engine.technical_qa(black, "1:1")
        white_qa = engine.technical_qa(white, "1:1")

        self.assertEqual(black_qa["checks"]["clipped_low_pct"], 100.0)
        self.assertEqual(black_qa["checks"]["clipped_high_pct"], 0.0)
        self.assertEqual(white_qa["checks"]["clipped_low_pct"], 0.0)
        self.assertEqual(white_qa["checks"]["clipped_high_pct"], 100.0)

    def test_reference_profile_rejects_visible_darkening(self):
        reference = np.full((40, 40, 3), 0.72, dtype=np.float32)
        candidate = np.full((40, 40, 3), 0.52, dtype=np.float32)

        qa = engine.technical_qa(
            candidate,
            "1:1",
            reference=reference,
            reference_profile=engine.analyze_color_reference(reference),
            variant="A",
        )

        self.assertFalse(qa["pass"])
        self.assertIn("exposure_drift", qa["checks"]["qa_failures"])
        self.assertLess(qa["checks"]["luma_drift_pct"], -20.0)

    def test_original_preserving_recovery_falls_back_to_reference(self):
        reference = np.full((40, 40, 3), 0.72, dtype=np.float32)
        scene = {"natural": {"contrast": 3.0, "sat": 1.0}, "saturation": 0}
        with patch.object(
            engine,
            "grade_variant_a",
            return_value=np.full_like(reference, 0.20),
        ):
            graded, qa, retries = engine._grade_variant_with_qa(
                "A",
                reference,
                scene=scene,
                intent={"contrast": -1.0, "saturation": 0.0, "warmth": 0.0},
                ratio="1:1",
                max_retries=2,
                reference=reference,
            )

        self.assertGreaterEqual(retries, 1)
        self.assertTrue(qa["pass"])
        self.assertEqual(qa["checks"]["fallback"], "original_like_safe_version")
        np.testing.assert_allclose(graded, reference)

    def test_blended_lut_writer_keeps_identity_at_zero_strength(self):
        from ig_automatik.core import lut_engine

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "creative.cube"
            source.write_text(
                "LUT_3D_SIZE 2\n"
                "1 1 1\n1 1 1\n1 1 1\n1 1 1\n"
                "1 1 1\n1 1 1\n1 1 1\n1 1 1\n",
                encoding="utf-8",
            )
            output = root / "identity.cube"
            lut_engine.write_blended_cube(source, 0.0, output)
            table = lut_engine.load_cube_file(output)
            np.testing.assert_allclose(
                table.reshape(-1, 3),
                np.array([
                    [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                    [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
                ], dtype=np.float32),
            )

    def test_fractional_vision_intent_changes_grading(self):
        base = np.linspace(0.1, 0.9, 30, dtype=np.float32).reshape(5, 2, 3)
        scene = {"natural": {"contrast": 3, "sat": 1}, "saturation": 0}

        without_intent = engine.grade_variant_a(base, scene=scene)
        with_intent = engine.grade_variant_a(
            base,
            scene=scene,
            intent={"contrast": 0.5, "saturation": 0.0},
        )

        self.assertFalse(np.array_equal(without_intent, with_intent))

    def test_vision_warmth_intent_changes_grading(self):
        base = np.full((5, 5, 3), 0.5, dtype=np.float32)

        neutral = engine.grade_variant_a(
            base,
            scene={"natural": {"contrast": 3, "sat": 1}, "saturation": 0},
            intent={"warmth": 0.0, "contrast": 0.0, "saturation": 0.0},
        )
        warm = engine.grade_variant_a(
            base,
            scene={"natural": {"contrast": 3, "sat": 1}, "saturation": 0},
            intent={"warmth": 1.0, "contrast": 0.0, "saturation": 0.0},
        )

        self.assertFalse(np.array_equal(neutral, warm))

        neutral_cinematic = engine.grade_variant_b(
            base,
            scene={"cinematic": {"contrast": 6, "sat": 4}, "saturation": 0},
            intent={"warmth": 0.0, "contrast": 0.0, "saturation": 0.0},
        )
        warm_cinematic = engine.grade_variant_b(
            base,
            scene={"cinematic": {"contrast": 6, "sat": 4}, "saturation": 0},
            intent={"warmth": 1.0, "contrast": 0.0, "saturation": 0.0},
        )

        self.assertFalse(np.array_equal(neutral_cinematic, warm_cinematic))

    def test_photo_processing_applies_technical_normalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.jpg"
            Image.new("RGB", (24, 18), (80, 120, 160)).save(source)
            cfg = {
                "output_width_post": 1080,
                "output_width_story": 1080,
                "export_quality": 95,
                "produce_ig": True,
                "produce_archives": False,
                "produce_formats": ["POSTS"],
            }

            with patch.object(engine.vision, "is_enabled", return_value=False), \
                    patch.object(
                        engine,
                        "normalize_technical",
                        wraps=engine.normalize_technical,
                    ) as normalizer:
                engine.process_photo(cfg, source, root / "output")

            self.assertTrue(normalizer.called)

    def test_photo_processing_passes_vision_intent_to_both_variants(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.jpg"
            Image.new("RGB", (24, 18), (80, 120, 160)).save(source)
            cfg = {
                "output_width_post": 1080,
                "output_width_story": 1080,
                "export_quality": 95,
                "produce_ig": True,
                "produce_archives": False,
                "produce_formats": ["POSTS"],
            }
            plan = {
                "scene_type": "portrait",
                "main_subject": "person",
                "grading_intent": {"warmth": 0.4, "contrast": 0.5, "saturation": -0.2},
                "provider": "openrouter",
            }

            with patch.object(engine.vision, "is_enabled", return_value=True), \
                    patch.object(engine.vision, "analyze", return_value=plan), \
                    patch.object(engine, "grade_variant_a", wraps=engine.grade_variant_a) as grade_a, \
                    patch.object(engine, "grade_variant_b", wraps=engine.grade_variant_b) as grade_b:
                engine.process_photo(cfg, source, root / "output")

            self.assertEqual(grade_a.call_args_list[0].kwargs["intent"], plan["grading_intent"])
            self.assertEqual(grade_b.call_args_list[0].kwargs["intent"], plan["grading_intent"])

    def test_manifest_is_unique_per_asset_and_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "output" / "POSTS"
            manifest_dir = root / "manifests"
            cfg = {"manifests_folder": str(manifest_dir)}
            files = {"A": {"ig": out_dir / "input_A.jpg"}}
            qa = {"A": {"pass": True}}

            manifest = engine.ExportManager(cfg).save_manifest(
                out_dir,
                "input",
                {"provider": "heuristic"},
                files,
                qa,
            )

            self.assertEqual(manifest, manifest_dir / "input_POSTS_manifest.json")
            self.assertTrue(manifest.exists())

    def test_export_uses_configured_width_for_small_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            cfg = {"output_width_post": 1080, "export_quality": 95}
            image = np.zeros((640, 512, 3), dtype=np.uint8)

            output = engine.ExportManager(cfg).save_jpg_ig(image, out_dir, "small")

            with Image.open(output) as exported:
                self.assertEqual(exported.width, 1080)

    def test_variant_creates_full_resolution_16bit_master_before_ig_derivative(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "2_FERTIG" / "POSTS"
            output_dir.mkdir(parents=True)
            masters_dir = root / "3_ARCHIV" / "MASTERS"
            cfg = {
                "export_quality": 95,
                "output_width_post": 1080,
                "produce_ig": True,
                "produce_masters": True,
                "masters_folder": str(masters_dir),
            }
            manager = engine.ExportManager(cfg)
            rgb = np.full((1500, 2000, 3), 0.42, dtype=np.float32)
            files = manager.save_variant(
                rgb, output_dir, "beach", "B", output_width=1080, format_name="POSTS"
            )

            self.assertIn("master", files)
            master = cv2.imread(str(files["master"]), cv2.IMREAD_UNCHANGED)
            ig = cv2.imread(str(files["ig"]), cv2.IMREAD_UNCHANGED)
            self.assertEqual(master.shape[:2], (1500, 2000))
            self.assertEqual(master.dtype, np.uint16)
            self.assertEqual(ig.shape[1], 1080)
            self.assertTrue(files["master"].name.endswith("_POSTS_B_master.png"))
            self.assertTrue(files["master"].is_relative_to(masters_dir))

    def test_master_failure_stops_social_derivative_when_masters_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "2_FERTIG" / "POSTS"
            output_dir.mkdir(parents=True)
            cfg = {
                "export_quality": 95,
                "output_width_post": 1080,
                "produce_ig": True,
                "produce_masters": True,
                "masters_folder": str(root / "3_ARCHIV" / "MASTERS"),
            }
            manager = engine.ExportManager(cfg)
            rgb = np.full((100, 100, 3), 0.42, dtype=np.float32)
            with patch.object(manager, "save_master_png", return_value=None):
                files = manager.save_variant(rgb, output_dir, "asset", "A", format_name="POSTS")

            self.assertEqual(files, {})
            self.assertFalse((output_dir / "asset_A.jpg").exists())

    def test_uint16_images_are_scaled_to_unit_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "sixteen_bit.png"
            pixels = np.full((4, 4, 3), 32768, dtype=np.uint16)
            cv2.imwrite(str(source), pixels[:, :, ::-1])

            loaded = engine.load_rgb(source)

            self.assertLessEqual(float(loaded.max()), 1.0)
            self.assertAlmostEqual(float(loaded.mean()), 32768 / 65535, places=3)

    def test_uint16_pillow_images_are_scaled_to_unit_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "sixteen_bit.tiff"
            pixels = np.full((4, 4), 32768, dtype=np.uint16)
            Image.fromarray(pixels).save(source)

            loaded = engine.load_rgb(source)

            self.assertLessEqual(float(loaded.max()), 1.0)
            self.assertAlmostEqual(float(loaded.mean()), 32768 / 65535, places=3)

    def test_failed_export_stops_processing_before_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.jpg"
            Image.new("RGB", (24, 18), (80, 120, 160)).save(source)
            cfg = {
                "output_width_post": 1080,
                "output_width_story": 1080,
                "export_quality": 95,
                "produce_ig": True,
                "produce_archives": False,
                "produce_formats": ["POSTS"],
            }

            with patch.object(engine.vision, "is_enabled", return_value=False), \
                    patch.object(engine.ExportManager, "verify_exports", return_value=False):
                with self.assertRaises(RuntimeError):
                    engine.process_photo(cfg, source, root / "output")

            self.assertTrue(source.exists())

    def test_empty_export_is_not_verified_as_successful(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = engine.ExportManager({})

            self.assertFalse(manager.verify_exports({}))

    def test_failed_archive_is_reported_as_processing_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.jpg"
            Image.new("RGB", (8, 8), (80, 120, 160)).save(source)
            cfg = {
                "input_folder": str(root),
                "output_folder": str(root / "output"),
                "processed_folder": str(root / "archive"),
                "produce_formats": [],
                "auto_move_sources": True,
            }

            with patch.object(engine, "process_photo"), \
                    patch.object(engine.Pipeline, "archive_source", return_value=False), \
                    patch.object(engine, "get_logger") as get_logger:
                engine.run_on_folder(cfg)

            get_logger.return_value.error.assert_called()
            self.assertTrue(source.exists())

    def test_archive_collision_uses_a_unique_name_without_data_loss(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.jpg"
            archive = root / "archive"
            source.write_bytes(b"new original")
            archive.mkdir()
            (archive / "input.jpg").write_bytes(b"old original")

            archived = engine.Pipeline({"processed_folder": str(archive)}).archive_source(source)

            self.assertTrue(archived)
            self.assertFalse(source.exists())
            self.assertEqual((archive / "input.jpg").read_bytes(), b"old original")
            self.assertEqual((archive / "input_2.jpg").read_bytes(), b"new original")

    def test_arw_and_gif_are_consistently_discoverable(self):
        self.assertIn(".arw", Media.PHOTO_EXT)
        self.assertIn(".gif", Media.PHOTO_EXT)
        self.assertTrue(watch.is_media(Path("photo.arw")))
        self.assertTrue(watch.is_media(Path("animation.gif")))
        self.assertIn(".arw", engine.PHOTO_EXT)
        self.assertIn(".gif", engine.PHOTO_EXT)

    def test_failed_reel_export_is_reported_as_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.mp4"
            source.write_bytes(b"video")
            cfg = {"reel_max_duration": 30}

            failed = type("Result", (), {"returncode": 1, "stderr": "ffmpeg failed"})()
            with patch.object(engine, "get_logger"), \
                    patch.object(video_tools, "detect_scenes", return_value=[]), \
                    patch.object(engine.subprocess, "run", return_value=failed):
                with self.assertRaises(RuntimeError):
                    engine.process_reel(cfg, source, root / "output")

    def test_reel_command_is_mobile_compatible(self):
        command = engine._build_reel_command(
            {"reel_max_duration": 30},
            Path("input.mov"),
            Path("output.mp4"),
            "A",
        )

        self.assertIn("+faststart", command)
        self.assertIn("-map", command)
        self.assertIn("0:v:0", command)
        self.assertIn("0:a:0?", command)
        self.assertIn("-r", command)
        self.assertIn("30", command)
        self.assertEqual(command[3], "input.mov")


if __name__ == "__main__":
    unittest.main()
