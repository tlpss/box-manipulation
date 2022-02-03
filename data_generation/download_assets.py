import os
import subprocess

home = os.path.expanduser("~")
output_folder = os.path.join(home, "assets")

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

bproc_download_script = "BlenderProc/blenderproc/scripts/download_haven.py"
resolution = "1k"
types = "hdris textures"

command = f"python3 {bproc_download_script} {output_folder} --resolution {resolution} --types {types}"

subprocess.run([command], shell=True)
