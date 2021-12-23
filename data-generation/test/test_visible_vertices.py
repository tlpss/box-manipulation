import unittest

import bpy
import numpy as np
from blender_utils.object_ops import make_object
from blender_utils.visible_vertices import get_visible_vertices
from box_generators.box_keypoints_generator_2 import make_scene
from general_utils.make_box import make_box


class TestVisibleVertices(unittest.TestCase):
    def test_visible_vertices_flaps_open(self):
        expected = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        bpy.ops.object.delete()  # delete default cube
        mesh = make_box(0.5, 0.5, 0.5, [-np.pi / 4] * 4, (0.5, 0.5))
        box = make_object("Box", mesh)
        result = get_visible_vertices(box)
        self.assertCountEqual(expected, result)

    def test_visible_vertices_947(self):
        expected = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        box, camera = make_scene(947)
        result = get_visible_vertices(box)
        self.assertCountEqual(expected, result)


if __name__ == "__main__":
    import sys

    sys.argv = [__file__] + (sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])
    # TestVisibleVertices().test_visible_vertices_947()
    unittest.main()
