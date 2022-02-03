import os

import airo_blender_toolkit as abt
import bpy
import numpy as np
from box.box import Box

os.environ["INSIDE_OF_THE_INTERNAL_BLENDER_PYTHON_ENVIRONMENT"] = "1"
import blenderproc as bproc


def generate_scene(seed):
    os.environ["BLENDER_PROC_RANDOM_SEED"] = str(seed)
    os.getenv("BLENDER_PROC_RANDOM_SEED")
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
    box.set_location((0, 0, 0.001))
    box.persist_transformation_into_mesh()

    camera = bpy.context.scene.camera
    camera.location = (0, 0, 1.25)
    camera.scale = [0.2] * 3
    camera.data.lens = 24

    home = os.path.expanduser("~")
    haven_folder = os.path.join(home, "assets", "haven")
    hdri_path = bproc.loader.get_random_world_background_hdr_img_path_from_haven(haven_folder)
    hdri_rotation = np.random.uniform(0, 2 * np.pi)
    abt.load_hdri(hdri_path, hdri_rotation)

    return box


if __name__ == "__main__":
    generate_scene(0)
