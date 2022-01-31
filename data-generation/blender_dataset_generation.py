"""
Script that generates the dataset
"""
import argparse
import datetime
import os
import sys

from box_generators import box_keypoints_generator_2


def generate_dataset(amount_of_samples, datasets_dir):
    dirname = f"boxes n={amount_of_samples} ({datetime.datetime.now()})"
    output_dir = os.path.join(datasets_dir, dirname)
    print(output_dir)
    os.makedirs(output_dir)

    for i in range(amount_of_samples):
        box_keypoints_generator_2.generate_datapoint(output_dir, i)
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
