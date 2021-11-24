"""
Script that generates the dataset
"""
import argparse
import datetime
import glob
import json
import os
import sys

from box_generators import box_keypoints_generator_0


def generate_dataset(amount_of_samples, datasets_dir):
    dirname = f"boxes n={amount_of_samples} ({datetime.datetime.now()})"
    output_dir = os.path.join(datasets_dir, dirname)
    os.makedirs(output_dir)

    for i in range(amount_of_samples):
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


if __name__ == "__main__":
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1 :]
        parser = argparse.ArgumentParser()
        parser.add_argument("amount_of_samples", type=int)
        parser.add_argument(
            "-d", "--dir", dest="datasets_dir", metavar="DATASETS_DIRECTORY", default=os.path.expanduser("~")
        )
        args = parser.parse_known_args(argv)[0]

        generate_dataset(args.amount_of_samples, args.datasets_dir)
