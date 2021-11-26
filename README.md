# box-manipulation [![pre-commit.ci status](https://results.pre-commit.ci/badge/github/tlpss/box-manipulation/master.svg)](https://results.pre-commit.ci/latest/github/tlpss/box-manipulation/master)

Using the AIRO UR3e to open the flaps of a box.


## Dataset Generation
The dataset generation is done using [Blender v2.93 LTS](https://www.blender.org/download/lts/2-93/)

### Development Environment
The `dataset-generation` folder contains 3 python packages with utils (general and blender specific) and a script to generate the dataset using Blender.

You can use the script by doing the following steps:
- download and extract the zip file from Blender
- move to the blender python folder: `cd ~/blender-X.XX.X-linux-x64/X.XX/python/bin`
- make sure pip is installed `./python3.9 -m ensurepip`
- install the packages locally using:
    - `./python3.9 -m pip install -e ~/box-manipulation/data-generation/blender_utils`
    - `./python3.9 -m pip install -e ~/box-manipulation/data-generation/general_utils`
    - `./python3.9 -m pip install -e ~/box-manipulation/data-generation/box_generators`
    - `./python3.9 -m pip install scipy
    - TODO -> fix w/ requirements file, maybe only install box_generators, all others are its deps?
- run the script with the blender python distribution

Additionally, it is convenient to add the path to blender to your PATH variable in your `.bashrc` file:
```
export PATH="$PATH:/home/.../.../blender-2.93.3-linux-x64/"
```
This allows you to call `blender` from any directory.

### Dataset Structure

####  json layout

example of the json format used in the dataset. Image paths are relative to a base directory
```
{
  "dataset": [
      { "image_path": "path",
        "corner_keypoints": [[1,3], [5,5],[4,7],[10,10]],
        "flap_corner_keypoints": [[1,3], [5,5],[4,7],[10,10], [1,2],[1,3],[2,3],[3,4]],
        "flap_center_keypoints": [[1,3], [5,5],[4,7],[10,10]],
        "metadata": {}
      },
      { "image_path": "path",
        "corner_keypoints": [[1,3], [5,5],[4,7],[10,10]],
        "flap_corner_keypoints": [[1,3], [5,5],[4,7],[10,10], [1,2],[1,3],[2,3],[3,4]],
        "flap_center_keypoints": [[1,3], [5,5],[4,7],[10,10]],
        "metadata": {}
      },
    ]
}
```
## Keypoint Detection
Keypoint Detector uses Pytorch (Lightning) and Weights and biases for experiment tracking and orchestration.
see `keypoint_detection package` for more details.

## Developer Guide

### Code Formatting
This project uses [pre-commit](https://pre-commit.com/) for managing git hooks that do code formatting, checks and linting.

These hooks are defined in `.pre-commit-config.yaml` and pre-commit will download the required tools (e.g. autoflake) if they are not installed in the environment. Furhtermore the pre-commit CI infrastructure will run all the hooks on push and pull request actions.

To install the hooks locally:

- run `pre-commit install` from the development environment (inside the conda env!), which attaches the hooks. From now on, the hooks will run when a commit is made to git.
- to manually run the hooks, run `pre-commit` or ` pre-commit run --al-files` to not only run the against the locally changed files.
