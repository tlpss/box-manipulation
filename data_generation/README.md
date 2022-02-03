# data-generation
Generate images of boxes rendered with Blender.

## Usage
Generating a scene:
```
blender -P flaps_open_camera_vertical.py -- 42
```
Generating a data point:
```
blender -b -P generate_data.py -- flaps_open_camera_vertical 42
```

Generating a dataset:
```
blender -b -P generate_dataset.py -- flaps_open_camera_vertical 1000
```

## Installation:
Add to `.bashrc` for convenience:
```bash
blender_path="/home/$USER/Blender/blender-3.0.0-linux-x64/"
export PATH="$PATH:$blender_path"
alias bpython="$blender_path/3.0/python/bin/python3.9"
```

Installing BlenderProc
```
git clone https://github.com/DLR-RM/BlenderProc.git
cd
bpython -m pip install -e BlenderProc
```

Installing box
```
bpython -m pip install -e box
```

Install `airo-blender-toolkit`

## Downloading assets
Before running the data generation, make sure to download the necessary assets by running:
```
python3 download_assets.py
```
This will create an `assets` folder in your home folder and download HDRIs and textures.


## Overview
![data_generation_overview](data_generation_overview.png)
