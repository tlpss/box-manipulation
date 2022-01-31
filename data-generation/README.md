# data-generation
Tools to generate synthetic images of boxes rendered in Blender.

## REWORK
Currently reworking the data-generation to use BlenderProc.

Add to `.bashrc` for convenience:
```bash
blender_path="/home/$USER/Blender/blender-3.0.0-linux-x64/"
export PATH="$PATH:$blender_path"
alias bpython="$blender_path/3.0/python/bin/python3.9"
```


Installation:
```
git clone https://github.com/DLR-RM/BlenderProc.git
cd BlenderProc
bpython -m pip install -e .
```

Usage of BlenderProc:
```python
import os
os.environ.setdefault("INSIDE_OF_THE_INTERNAL_BLENDER_PYTHON_ENVIRONMENT", "1")
import blenderproc as bproc
```

Then run with:
```
blender -P your_script.py
```

Also because we installed BlenderProc into the blender python, you cannot use the `blenderproc` command but have to use:
```
bpython cli.py <args>
```
We could also add an alias for this to `.bashrc`


## OLD
The code is seperated into three packages.
* `blender-utils`: high-level routines for interacting with Blender e.g. rotating objects, creating materials.
* `general-utils`: utilities that do not rely on Blender, e.g. generating the box geometry
* `box-generators`: scripts that generate and render specific scenes
