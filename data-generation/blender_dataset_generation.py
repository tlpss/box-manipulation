"""
Script that generates the dataset 
"""
import os
import math
import numpy as np
import json
import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

from general_utils.make_box import make_box, random_flag_angles
from blender_utils.object_ops import make_object, rotate, set_location, cleanup_scene, look_at, select_only
from blender_utils.materials import random_box_material

def render_box(z_angle, image_path):
    cleanup_scene()

    bpy.ops.object.light_add(type="SUN", location=(1, 1, 1))
    sun = bpy.data.objects["Sun"]
    sun.data.energy = 5
    sun.rotation_euler[0] = math.radians(30)
    sun.rotation_euler[1] = math.radians(20)

    bpy.ops.object.light_add(
        type="AREA", align="WORLD", location=(0, 0, 2), scale=(1, 1, 1)
    )
    area = bpy.data.objects["Area"]
    area.data.energy = 50

    bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, -0.005))

    bpy.ops.object.camera_add(location=(0, 0, 1))
    camera = bpy.context.active_object
    camera.data.lens = 50  # focal length in mm
    look_at([0, 0, 0], camera)

    l = np.random.uniform(0.1, 0.5)
    w = np.random.uniform(0.1, l)
    h = np.random.uniform(0.1, 0.3)

    angles = random_flag_angles()
    mesh = make_box(l, w, h, angles)
    box = make_object("Box", mesh)

    set_location(box, (0.02, 0.01, 0))
    rotate(box, z_angle)

    box.data.materials.append(random_box_material())

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = 'RGB'
    scene.cycles.samples = 32
    scene.render.resolution_x = 256
    scene.render.resolution_y = 256
    scene.camera = camera
    scene.render.filepath = image_path
    bpy.ops.render.render(write_still=True)

    corner_ids = [4, 5, 6, 7]

    vertices_uvz = [
        world_to_camera_view(scene, camera, v.co)
        for i, v in enumerate(box.data.vertices)
        if i in corner_ids
    ]
    corner_keypoints = [(u, v) for u, v, z in vertices_uvz]
    return corner_keypoints


if __name__ == "__main__":
    output_dir = "/home/idlab185/box_dataset_temp"
    dataset = []
    np.random.seed(0)

    for i in range(1):
        z_angle = np.random.uniform(0.0, 2 * np.pi)
        image_name = 'box' + str(i)  + '.png'
        relative_path = os.path.join('images', image_name)
        image_path = os.path.join(output_dir, relative_path)
        corner_keypoints = render_box(z_angle, image_path)
        data = {
            "image_path": relative_path,
            "corner_keypoints": corner_keypoints,
        }
        dataset.append(data)

    dataset_json = {"dataset": dataset}

    with open(os.path.join(output_dir, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=2)
