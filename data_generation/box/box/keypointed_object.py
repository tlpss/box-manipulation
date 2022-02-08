import os

import airo_blender_toolkit as abt
import bpy
from bpy_extras.object_utils import world_to_camera_view

os.environ["INSIDE_OF_THE_INTERNAL_BLENDER_PYTHON_ENVIRONMENT"] = "1"
from blenderproc.python.types.MeshObjectUtility import MeshObject


class KeypointedObject(MeshObject):
    """Base class for custom object with keypoints.
    Trying this for now but not sure if it's worth it.
    """

    def __init__(self, bpy_object: bpy.types.Object, keypoint_ids: dict[str, list[int]]):
        self.keypoint_ids = keypoint_ids
        super().__init__(bpy_object)

    def __setattr__(self, key, value):
        """Reset the __setattr__ because the Entity parent class disallows assignment."""
        object.__setattr__(self, key, value)

    @property
    def keypoints_3D(self):
        vertices = self.blender_obj.data.vertices
        keypoints = {}
        for key, vertex_ids in self.keypoint_ids.items():
            keypoints[key] = [vertices[id].co for id in vertex_ids]
        return keypoints

    @property
    def keypoints_3D_visible(self):
        keypoints_3D_visible = {}
        for key, coords in self.keypoints_3D.items():
            keypoints_3D_visible[key] = [co for co in coords if abt.is_visible(co)]
        return keypoints_3D_visible

    @property
    def keypoints_2D(self):
        return KeypointedObject.project_to_camera(self.keypoints_3D)

    @property
    def keypoints_2D_visible(self):
        return KeypointedObject.project_to_camera(self.keypoints_3D_visible)

    @staticmethod
    def project_to_camera(keypoints):
        # TODO support multiview, add camera arg to keypoints_2D
        scene = bpy.context.scene
        camera = scene.camera
        keypoints_2D = {}
        for key, coords in keypoints.items():
            keypoints_2D[key] = [world_to_camera_view(scene, camera, co) for co in coords]
        return keypoints_2D

    def json_ready_keypoints(self, dimension=2, only_visible=True):
        if dimension == 2 and only_visible:
            keypoints = self.keypoints_2D_visible
            suffix = "_keypoints_visible"
        elif dimension == 2 and not only_visible:
            keypoints = self.keypoints_2D
            suffix = "_keypoints"
        elif dimension == 3 and only_visible:
            keypoints = self.keypoints_3D_visible
            suffix = "_keypoints_3D_visible"
        else:
            keypoints = self.keypoints_3D
            suffix = "_keypoints_3D"

        keypoints_json = {}
        for key, coords in keypoints.items():
            keypoints_json[key + suffix] = [list(c) for c in coords]
        return keypoints_json