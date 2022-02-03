import os

import bpy
import numpy as np
from custom_blender_objects.box import Box

os.environ.setdefault("INSIDE_OF_THE_INTERNAL_BLENDER_PYTHON_ENVIRONMENT", "1")
import blenderproc as bproc

output_dir = __file__.split(".py")[0]

bproc.init()

ground = bproc.object.create_primitive("PLANE")
ground.blender_obj.name = "ground"

box_length = np.random.uniform(0.2, 0.5)
box_width = np.random.uniform(0.2, box_length)
box_height = np.random.uniform(0.1, 0.3)

long_flaps_fraction = np.random.uniform(0.45, 0.5)
short_flaps_fraction = np.random.uniform(0.25, 0.5)
flap_size_fractions = (long_flaps_fraction, short_flaps_fraction)

box = Box(box_length, box_width, box_height, Box.outwards_flap_angles(), flap_size_fractions)
z_angle = np.random.uniform(0.0, 2 * np.pi)
box.set_rotation_euler([0, 0, z_angle])

camera = bpy.context.scene.camera
camera.location = (0, 0, 1.25)
