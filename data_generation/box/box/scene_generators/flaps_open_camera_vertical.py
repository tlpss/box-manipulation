import argparse
import os
import sys

import airo_blender_toolkit as abt
import bpy
import numpy as np
from box.box import Box
from mathutils import Color

os.environ["INSIDE_OF_THE_INTERNAL_BLENDER_PYTHON_ENVIRONMENT"] = "1"
import blenderproc as bproc


def generate_scene(seed):
    os.environ["BLENDER_PROC_RANDOM_SEED"] = str(seed)
    os.getenv("BLENDER_PROC_RANDOM_SEED")
    bproc.init()

    home = os.path.expanduser("~")
    haven_folder = os.path.join(home, "assets", "haven")
    haven_textures_folder = os.path.join(haven_folder, "textures")

    ground = bproc.object.create_primitive("PLANE")
    ground.blender_obj.name = "ground"
    ground.set_scale([5] * 3)
    ground.set_rotation_euler([0, 0, np.random.uniform(0.0, 2 * np.pi)])
    ground_texture = abt.random_texture_name(haven_textures_folder)
    print(ground_texture)
    bproc.api.loader.load_haven_mat(haven_textures_folder, [ground_texture])
    ground_material = bpy.data.materials[ground_texture]
    ground.blender_obj.data.materials.append(ground_material)

    box_length = np.random.uniform(0.2, 0.5)
    box_width = np.random.uniform(0.2, box_length)
    box_height = np.random.uniform(0.1, 0.3)

    long_flaps_fraction = np.random.uniform(0.45, 0.5)
    short_flaps_fraction = np.random.uniform(0.25, 0.5)
    flap_size_fractions = (long_flaps_fraction, short_flaps_fraction)

    box = Box(box_length, box_width, box_height, Box.outwards_flap_angles(), flap_size_fractions)
    box.set_rotation_euler([0, 0, np.random.uniform(0.0, 2 * np.pi)])
    box.set_location((0, 0, 0.001))
    box.persist_transformation_into_mesh()

    box_color = Color()
    box_hue = np.random.uniform(0, 0.14)
    box_saturation = np.random.uniform(0.4, 0.8)
    box_value = np.random.uniform(0.2, 0.5)
    box_color.hsv = box_hue, box_saturation, box_value
    box_material = box.new_material("box")
    box_material.set_principled_shader_value("Base Color", tuple(box_color) + (1,))

    camera = bpy.context.scene.camera
    camera_location = bproc.sampler.part_sphere(center=[0, 0, 0], radius=1.5, mode="INTERIOR", dist_above_center=1.25)
    camera_rotation = bproc.python.camera.CameraUtility.rotation_from_forward_vec((0, 0, 0) - camera_location)
    camera_pose = bproc.math.build_transformation_mat(camera_location, camera_rotation)
    bproc.camera.add_camera_pose(camera_pose)

    camera.scale = [0.2] * 3
    camera.data.lens = 24

    hdri_path = bproc.loader.get_random_world_background_hdr_img_path_from_haven(haven_folder)
    hdri_rotation = np.random.uniform(0, 2 * np.pi)
    abt.load_hdri(hdri_path, hdri_rotation)

    return box


if __name__ == "__main__":
    seed = 0
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1 :]
        parser = argparse.ArgumentParser()
        parser.add_argument("seed", type=int)
        args = parser.parse_known_args(argv)[0]
        seed = args.seed
    generate_scene(seed)
