# box_generators
Scripts that generate a specific randomized box scene, render an image of the scene and store the corresponding box keypoints.

The idea is currently to simply combine the configuration and the generation into a single script. 
This means that e.g. the distribution from which the box-width is sampled, is hardcoded into the script.

Each script contains one main function `generate_datapoint(output_dir, seed=0)`.
Based on the seed, a single images and its label will be written to the output directory.

Later we could consider to write a more general "scene parser" that build a scene based on a json config file.
