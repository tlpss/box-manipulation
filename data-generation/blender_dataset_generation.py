"""
Script that generates the dataset
"""
import json
import os

import bpy
import numpy as np
from blender_utils.hdri import load_hdri
from blender_utils.materials import random_box_material
from blender_utils.object_ops import cleanup_scene, look_at, make_object, rotate
from bpy_extras.object_utils import world_to_camera_view
from general_utils.make_box import make_box, random_flag_angles

hdris = [
    "abandoned_church_8k.exr",
    "abandoned_factory_canteen_01_8k.exr",
    "abandoned_factory_canteen_02_8k.exr",
    "abandoned_games_room_01_8k.exr",
    "abandoned_games_room_02_8k.exr",
    "abandoned_greenhouse_8k.exr",
    "abandoned_hall_01_8k.exr",
    "abandoned_workshop_8k.exr",
    "adams_place_bridge_8k.exr",
    "aerodynamics_workshop_8k.exr",
    "aft_lounge_8k.exr",
    "autoshop_01_8k.exr",
    "bell_park_pier_8k.exr",
    "champagne_castle_1_8k.exr",
    "studio_small_09_8k.exr",
    "spaichingen_hill_8k.exr",
    "monbachtal_riverbank_8k.exr",
    "dikhololo_night_8k.exr",
    "sunset_in_the_chalk_quarry_8k.exr",
    "air_museum_playground_8k.exr",
    "sepulchral_chapel_rotunda_8k.exr",
]


def create_box_scene(data):
    cleanup_scene()

    # Box creation
    l = np.random.uniform(0.2, 0.5)
    w = np.random.uniform(0.2, l)
    h = np.random.uniform(0.1, 0.3)
    long_flaps_length = w * np.random.uniform(0.45, 0.5)
    short_flaps_length = l * np.random.uniform(0.25, 0.5)
    flap_lengths = (long_flaps_length, short_flaps_length)
    angles = random_flag_angles()
    mesh = make_box(l, w, h, angles, flap_lengths)
    box = make_object("Box", mesh)
    # set_location(box, (0.02, 0.01, 0))
    z_angle = np.random.uniform(0.0, 2 * np.pi)
    rotate(box, z_angle)
    box.data.materials.append(random_box_material())

    # Ground plane
    bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, -0.005))

    # Lighting
    hdri_id = np.random.choice(len(hdris))
    hdri_rotation = np.random.uniform(0, 2 * np.pi)
    load_hdri(f"data-generation/assets/HDRIs/{hdris[hdri_id]}", hdri_rotation)

    # Camera config
    bpy.ops.object.camera_add(location=(0, 0, 1.25))
    camera = bpy.context.active_object
    camera.data.lens = 24  # focal length in mm
    look_at([0, 0, 0], camera)
    scene = bpy.context.scene
    scene.cycles.samples = 64
    scene.camera = camera

    # Corner keypoints in 2D
    corner_ids = [4, 5, 6, 7]
    flap_corner_ids = [8, 9, 10, 11, 12, 13, 14, 15]
    vertices_uvz = [world_to_camera_view(scene, camera, v.co) for v in box.data.vertices]
    corner_keypoints = [(u, v) for i, (u, v, _) in enumerate(vertices_uvz) if i in corner_ids]
    flap_corner_keypoints = [(u, v) for i, (u, v, _) in enumerate(vertices_uvz) if i in flap_corner_ids]
    flap_center_keypoints = []
    for i in range(4):
        flap_corner0 = np.array(flap_corner_keypoints[2 * i])
        flap_corner1 = np.array(flap_corner_keypoints[2 * i + 1])
        flap_center = tuple(0.5 * (flap_corner0 + flap_corner1))
        flap_center_keypoints.append(flap_center)

    data["corner_keypoints"] = corner_keypoints
    data["flap_corner_keypoints"] = flap_corner_keypoints
    data["flap_center_keypoints"] = flap_center_keypoints


if __name__ == "__main__":
    output_dir = "/home/idlab185/box_dataset_temp5"

    # General settings
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.resolution_x = 256
    scene.render.resolution_y = 256

    dataset = []
    np.random.seed(0)

    for i in range(250):
        image_name = "box" + str(i) + ".png"
        relative_path = os.path.join("images", image_name)
        image_path = os.path.join(output_dir, relative_path)

        data = {
            "image_path": relative_path,
        }

        create_box_scene(data)

        bpy.context.scene.render.filepath = image_path
        bpy.ops.render.render(write_still=True)

        dataset.append(data)

    dataset_json = {"dataset": dataset}

    with open(os.path.join(output_dir, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=2)
