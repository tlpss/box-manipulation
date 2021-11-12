import numpy as np
from bpy_extras.object_utils import world_to_camera_view


def get_box_keypoints(box, camera, scene):
    corner_ids = [4, 5, 6, 7]
    flap_corner_ids = [8, 9, 10, 11, 12, 13, 14, 15]

    corners = [v for i, v in enumerate(box.data.vertices) if i in corner_ids]
    flap_corners = [v for i, v in enumerate(box.data.vertices) if i in flap_corner_ids]

    corner_keypoints = [tuple(world_to_camera_view(scene, camera, v.co)) for v in corners]
    flap_corner_keypoints = [tuple(world_to_camera_view(scene, camera, v.co)) for v in flap_corners]

    flap_center_keypoints = []
    for i in range(4):
        flap_corner0 = np.array(flap_corner_keypoints[2 * i])
        flap_corner1 = np.array(flap_corner_keypoints[2 * i + 1])
        flap_center = tuple(0.5 * (flap_corner0 + flap_corner1))
        flap_center_keypoints.append(flap_center)

    return corner_keypoints, flap_corner_keypoints, flap_center_keypoints
