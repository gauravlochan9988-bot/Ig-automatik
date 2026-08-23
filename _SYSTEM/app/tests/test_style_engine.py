import unittest

import numpy as np

from ig_automatik.core import style_engine


class StyleEngineTests(unittest.TestCase):
    def test_warm_travel_scene_uses_local_intent_and_srgb_candidates(self):
        intent = style_engine.build_style_intent({
            "scene_type": "sunset", "main_subject": "beach in Florida"
        })
        candidates = style_engine.rank_lut_candidates(intent)

        self.assertEqual(intent["family"], "warm_travel")
        self.assertTrue(intent["preserve_sky"])
        self.assertTrue(candidates)
        self.assertNotIn("DisplayP3", candidates[0]["name"])
        self.assertNotIn("CONVERSION", candidates[0]["name"])

    def test_portrait_style_preserves_skin_and_chooses_local_candidate(self):
        intent = style_engine.build_style_intent({
            "scene_type": "portrait", "main_subject": "wedding couple"
        })
        chosen = style_engine.choose_lut(intent)

        self.assertTrue(intent["preserve_skin"])
        self.assertIsNotNone(chosen)
        self.assertGreater(chosen["strength"], 0.0)
        self.assertNotIn("DisplayP3", chosen["name"])

    def test_vision_creative_direction_guides_style_but_stays_locally_validated(self):
        intent = style_engine.build_style_intent({
            "scene_type": "general",
            "main_subject": "woman by the ocean",
            "creative_direction": {
                "style_family": "warm_travel",
                "light_mood": "golden_hour",
                "preserve": ["skin tones", "sunset sky"],
            },
        })
        self.assertEqual(intent["family"], "warm_travel")
        self.assertEqual(intent["light_mood"], "golden_hour")
        self.assertTrue(intent["preserve_skin"])
        self.assertTrue(intent["preserve_sky"])

    def test_lut_blend_strength_is_bounded(self):
        original = np.zeros((2, 2, 3), dtype=np.float32)
        transformed = np.ones((2, 2, 3), dtype=np.float32)
        np.testing.assert_allclose(style_engine.blend_lut(original, transformed, 0.42), 0.42)
        np.testing.assert_allclose(style_engine.blend_lut(original, transformed, 5), 1.0)
        np.testing.assert_allclose(style_engine.blend_lut(original, transformed, -1), 0.0)


if __name__ == "__main__":
    unittest.main()
