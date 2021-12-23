import numpy as np
from blender_utils.visible_vertices import get_visible_vertices
from bpy_extras.object_utils import world_to_camera_view


def get_box_keypoints(box, camera, scene):
    corner_ids = [4, 5, 6, 7]
    flap_corner_ids = [8, 9, 10, 11, 12, 13, 14, 15]

    visible_ids = get_visible_vertices(box)

    corners = [v for v in box.data.vertices if v.index in corner_ids]
    corners_visible = [v for v in corners if v.index in visible_ids]
    flap_corners = [v for v in box.data.vertices if v.index in flap_corner_ids]
    flap_corners_visible = [v for v in flap_corners if v.index in visible_ids]

    corner_keypoints = [tuple(world_to_camera_view(scene, camera, v.co)) for v in corners]
    corner_keypoints_visible = [tuple(world_to_camera_view(scene, camera, v.co)) for v in corners_visible]

    flap_corner_keypoints = [tuple(world_to_camera_view(scene, camera, v.co)) for v in flap_corners]
    flap_corner_keypoints_visible = [tuple(world_to_camera_view(scene, camera, v.co)) for v in flap_corners_visible]

    flap_center_keypoints = []
    for i in range(4):
        flap_corner0 = np.array(flap_corner_keypoints[2 * i])
        flap_corner1 = np.array(flap_corner_keypoints[2 * i + 1])
        flap_center = tuple(0.5 * (flap_corner0 + flap_corner1))
        flap_center_keypoints.append(flap_center)

    flap_center_keypoints_visible = []

    # Currently assumes is both corners are always visible together
    for i in range(len(flap_corner_keypoints_visible) // 2):
        flap_corner0 = np.array(flap_corner_keypoints_visible[2 * i])
        flap_corner1 = np.array(flap_corner_keypoints_visible[2 * i + 1])
        flap_center = tuple(0.5 * (flap_corner0 + flap_corner1))
        flap_center_keypoints_visible.append(flap_center)

    keypoints = {
        "corner_keypoints": corner_keypoints,
        "flap_corner_keypoints": flap_corner_keypoints,
        "flap_center_keypoints": flap_center_keypoints,
        "corner_keypoints_visible": corner_keypoints_visible,
        "flap_corner_keypoints_visible": flap_corner_keypoints_visible,
        "flap_center_keypoints_visible": flap_center_keypoints_visible,
    }

    return keypoints
