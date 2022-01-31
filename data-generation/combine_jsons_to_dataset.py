import glob
import json
import os

output_dir = os.getcwd()

dataset = []

for file in glob.glob(os.path.join(output_dir, "json/*.json")):
    with open(file) as f:
        dataset.append(json.load(f))

# TODO add timestamp of generation in json
dataset_json = {"dataset": dataset}

with open(os.path.join(output_dir, "dataset.json"), "w") as f:
    json.dump(dataset_json, f, indent=2)
