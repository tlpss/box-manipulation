# box-manipulation [![pre-commit.ci status](https://results.pre-commit.ci/badge/github/tlpss/box-manipulation/master.svg)](https://results.pre-commit.ci/latest/github/tlpss/box-manipulation/master)

Using the AIRO UR3e to open the flaps of a box.


## Dataset Generation
The dataset generation is done using [Blender v2.93 LTS](https://www.blender.org/download/lts/2-93/)

### Development Environment
The `dataset-generation` folder contains 2 python packages with utils (general and blender specific) and a script to generate the dataset using Blender.

You can use the script by doing the following steps:
- download and extract the zip file from Blender
- symlink this repository under `blender-2.93.X.../2.93/scripts`
- move to the blender root folder
- make sure pip is installed ``2.93/python/bin/python3.9 -m ensurepip`
- install the packages locally using `2.93/python/bin/python3.9 -m pip install -e 2.93/python/scripts/box-manipulation/data-generation/blender_utils` and vice versa for the other package (TODO -> fix w/ requirements file)
- run the script with the blender python distribution

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

### Development environment
- run vscode remote development or manually build the Dockerfile in `.devcontainer/docker/JupyterLab/Dockerfile`
- notebooks

## Developer Guide

### Code Formatting
This project uses [pre-commit](https://pre-commit.com/) for managing git hooks that do code formatting, checks and linting.

These hooks are defined in `.pre-commit-config.yaml` and pre-commit will download the required tools (e.g. autoflake) if they are not installed in the environment. Furhtermore the pre-commit CI infrastructure will run all the hooks on push and pull request actions.

To install the hooks locally:

- run `pre-commit install` from the development environment (inside the conda env!), which attaches the hooks. From now on, the hooks will run when a commit is made to git.
- to manually run the hooks, run `pre-commit` or ` pre-commit run --al-files` to not only run the against the locally changed files.
