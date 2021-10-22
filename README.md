# box-manipulation
Using the AIRO UR3e to open the flaps of a box. 


## Dataset Generation
The dataset generation is done using [Blender v2.93 LTS](https://www.blender.org/download/lts/2-93/)

The `dataset-generation` folder contains 2 python packages with utils (general and blender specific) and a script to generate the dataset using Blender.

You can use the script by doing the following steps:
- download and extract the zip file from Blender
- symlink this repository under `blender-2.93.X.../2.93/scripts`
- move to the blender root folder
- make sure pip is installed ``2.93/python/bin/python3.9 -m ensurepip`
- install the packages locally using `2.93/python/bin/python3.9 -m pip install -e 2.93/python/scripts/box-manipulation/data-generation/blender_utils` and vice versa for the other package (TODO -> fix w/ requirements file)
- run the script with the blender python distribution

## Keypoint Detection


