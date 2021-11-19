import json
import os

import bpy
import numpy as np
from blender_utils.box import get_box_keypoints
from blender_utils.hdri import load_hdri
from blender_utils.materials import random_box_material
from blender_utils.object_ops import cleanup_scene, look_at, make_object, rotate
from general_utils.make_box import make_box, random_flap_angles

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


def generate_datapoint(output_dir, seed=0):
    image_name = f"{str(seed)}.png"
    image_path_relative = os.path.join("images", image_name)
    image_path = os.path.join(output_dir, image_path_relative)

    np.random.seed(seed)

    # General settings
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.resolution_x = 256
    scene.render.resolution_y = 256
    scene.render.filepath = image_path

    cleanup_scene()

    # Box creation
    l = np.random.uniform(0.2, 0.5)
    w = np.random.uniform(0.2, l)
    h = np.random.uniform(0.1, 0.3)
    long_flaps_fraction = np.random.uniform(0.45, 0.5)
    short_flaps_fraction = np.random.uniform(0.25, 0.5)
    flap_size_fractions = (long_flaps_fraction, short_flaps_fraction)
    angles = random_flap_angles()
    mesh = make_box(l, w, h, angles, flap_size_fractions)
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
    data_generation_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    hdri_path = os.path.join(data_generation_dir, f"assets/HDRIs/{hdris[hdri_id]}")
    load_hdri(hdri_path, hdri_rotation)

    # Camera config
    bpy.ops.object.camera_add(location=(0, 0, 1.25))
    camera = bpy.context.active_object
    camera.data.lens = 24  # focal length in mm
    look_at([0, 0, 0], camera)
    scene = bpy.context.scene
    scene.cycles.samples = 64
    scene.camera = camera

    # Box keypoints in 2D
    corner_keypoints, flap_corner_keypoints, flap_center_keypoints = get_box_keypoints(box, camera, scene)

    data = {
        "image_path": image_path_relative,
        "corner_keypoints": corner_keypoints,
        "flap_corner_keypoints": flap_corner_keypoints,
        "flap_center_keypoints": flap_center_keypoints,
    }

    json_dir = os.path.join(output_dir, "json")
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)

    json_path = os.path.join(json_dir, f"{str(seed)}.json")

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    # Render image
    bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    output_dir = __file__.split(".py")[0]
    generate_datapoint(output_dir, 0)
