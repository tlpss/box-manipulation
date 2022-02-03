import argparse
import json
import os
import sys

import box
import bpy


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
    if "--" in sys.argv:
        home = os.path.expanduser("~")
        default_datasets_path = os.path.join(home, "datasets")

        argv = sys.argv[sys.argv.index("--") + 1 :]
        parser = argparse.ArgumentParser()
        parser.add_argument("scene_name")
        parser.add_argument("seed", type=int)
        parser.add_argument(
            "-d", "--dir", dest="datasets_dir", metavar="DATASETS_DIRECTORY", default=default_datasets_path
        )
        args = parser.parse_known_args(argv)[0]

        scene_name = args.scene_name
        output_dir = os.path.join(args.datasets_dir, f"box {scene_name}")

        if scene_name not in box.scene_generators.__all__:
            print(f"No scene found with name {scene_name}")
            exit(-1)
        scene_generator = sys.modules[f"box.scene_generators.{scene_name}"]

        generate_data(output_dir, scene_generator, args.seed)
