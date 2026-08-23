import tempfile
import unittest
from pathlib import Path
import numpy as np

from ig_automatik.core import lut_engine


class LutEngineTests(unittest.TestCase):
    def test_parse_cube_file(self):
        cube_text = """
# Test 2x2x2 cube
LUT_3D_SIZE 2
0.0 0.0 0.0
1.0 0.0 0.0
0.0 1.0 0.0
1.0 1.0 0.0
0.0 0.0 1.0
1.0 0.0 1.0
0.0 1.0 1.0
1.0 1.0 1.0
"""
        with tempfile.NamedTemporaryFile("w", suffix=".cube", delete=False) as f:
            f.write(cube_text)
            f_path = Path(f.name)

        try:
            table = lut_engine.load_cube_file(f_path)
            self.assertIsNotNone(table)
            self.assertEqual(table.shape, (2, 2, 2, 3))
        finally:
            f_path.unlink(missing_ok=True)

    def test_apply_identity_lut(self):
        # 2x2x2 identity LUT
        cube_text = """
LUT_3D_SIZE 2
0.0 0.0 0.0
1.0 0.0 0.0
0.0 1.0 0.0
1.0 1.0 0.0
0.0 0.0 1.0
1.0 0.0 1.0
0.0 1.0 1.0
1.0 1.0 1.0
"""
        with tempfile.NamedTemporaryFile("w", suffix=".cube", delete=False) as f:
            f.write(cube_text)
            f_path = Path(f.name)

        try:
            table = lut_engine.load_cube_file(f_path)
            img = np.array([[[0.0, 0.5, 1.0], [0.25, 0.75, 0.5]]], dtype=np.float32)
            res = lut_engine.apply_lut(img, table)
            self.assertEqual(res.shape, img.shape)
            np.testing.assert_allclose(res, img, atol=1e-3)
        finally:
            f_path.unlink(missing_ok=True)

    def test_list_and_match_luts(self):
        luts = lut_engine.list_luts()
        self.assertGreaterEqual(len(luts), 3)
        
        # Test scene matching
        matched_cinematic = lut_engine.match_lut_for_scene({"scene_type": "night"})
        self.assertIsNotNone(matched_cinematic)
        self.assertTrue(matched_cinematic.exists())
