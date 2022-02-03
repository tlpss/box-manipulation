import json
import os

import bpy
from box.scene_generators import flaps_open_camera_vertical


def generate_data(output_dir, box_scene_generator, seed, resolution=256):
    box = box_scene_generator.generate_scene(seed)
    scene = bpy.context.scene

    image_name = f"{str(seed)}.png"
    image_path_relative = os.path.join("images", image_name)
    image_path = os.path.join(output_dir, image_path_relative)

    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution

    scene.render.filepath = image_path

    data = {
        "image_path": image_path_relative,
    }

    keypoints_2D = box.json_ready_keypoints(dimension=2, only_visible=True)
    keypoints_2D_visible = box.json_ready_keypoints(dimension=2, only_visible=False)

    data = data | keypoints_2D | keypoints_2D_visible

    # Saving the data as json
    json_dir = os.path.join(output_dir, "json")
    os.makedirs(json_dir, exist_ok=True)
    json_path = os.path.join(json_dir, f"{str(seed)}.json")
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    default_scene_generator = flaps_open_camera_vertical
    output_dir = default_scene_generator.__name__
    generate_data(output_dir, default_scene_generator, seed=0)
