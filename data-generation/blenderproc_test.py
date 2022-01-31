import os

os.environ.setdefault("INSIDE_OF_THE_INTERNAL_BLENDER_PYTHON_ENVIRONMENT", "1")
import blenderproc as bproc

hdri_path = "/home/idlab185/cloth-manipulation/misc"

hdri = bproc.loader.get_random_world_background_hdr_img_path_from_haven(hdri_path)
bproc.world.set_world_background_hdr_img(hdri)
