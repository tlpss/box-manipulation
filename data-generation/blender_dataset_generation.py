"""
Script that generates the dataset
"""
import datetime
import glob
import json
import os

from box_generators import box_keypoints_generator_0

if __name__ == "__main__":
    home_dir = os.path.expanduser("~")
    output_dir = os.path.join(home_dir, f"box_dataset_{datetime.datetime.now()}")
    os.makedirs(output_dir)

    for i in range(2):
        box_keypoints_generator_0.generate_datapoint(output_dir, i)

    dataset = []

    for file in glob.glob(os.path.join(output_dir, "json/*.json")):
        with open(file) as f:
            dataset.append(json.load(f))

    # TODO add timestamp of generation in json
    dataset_json = {"dataset": dataset}

    with open(os.path.join(output_dir, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=2)

    # TODO copy generation script to output_dir
