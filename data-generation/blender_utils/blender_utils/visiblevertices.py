# Adapted from https://github.com/varkenvarken/blenderaddons/blob/master/visiblevertices.py

import bpy
import numpy as np
from blender_utils.object_ops import make_object
from general_utils.make_box import make_box
from mathutils import Vector
from mathutils.geometry import intersect_ray_tri


def intersect_ray_quad_3d(quad, origin, destination):
    ray = destination - origin
    p = intersect_ray_tri(quad[0], quad[1], quad[2], ray, origin)
    if p is None:
        p = intersect_ray_tri(quad[2], quad[3], quad[0], ray, origin)
    return p


def intersect_ray_scene(scene, origin, destination):
    direction = destination - origin
    result, _, _, _, object, _ = scene.ray_cast(
        bpy.context.view_layer.depsgraph,
        origin=origin + direction * 0.0001,
        direction=destination,
    )
    return result


def get_visible_vertices(obj):
    scene = bpy.context.scene
    cam_ob = scene.camera
    cam = bpy.data.cameras[cam_ob.name]  # camera in scene is object type, not a camera type
    cam_mat = cam_ob.matrix_world
    view_frame = cam.view_frame(
        scene=scene
    )  # without a scene the aspect ratio of the camera is not taken into account
    view_frame = [cam_mat @ v for v in view_frame]
    cam_pos = cam_mat @ Vector((0, 0, 0))
    view_center = sum(view_frame, Vector((0, 0, 0))) / len(view_frame)
    view_normal = (view_center - cam_pos).normalized()

    mesh_mat = obj.matrix_world
    mesh = obj.data

    visible_vertices = []

    for v in mesh.vertices:
        vertex_coords = mesh_mat @ v.co
        d = None
        intersection = intersect_ray_quad_3d(
            view_frame, vertex_coords, cam_pos
        )  # check intersection with the camera frame

        if intersection is not None:
            d = intersection - vertex_coords
            # only take into account vertices in front of the camera, not behind it.
            if d.dot(view_normal) < 0:
                d = d.length
                # check intersection with all other objects in scene. We revert the direction, ie. look from the camera to avoid self intersection
                if intersect_ray_scene(scene, vertex_coords, cam_pos):
                    d = None
            else:
                d = None

        visible = d is not None and d > 0.0

        if visible:
            visible_vertices.append(v.index)

    return visible_vertices


if __name__ == "__main__":
    bpy.ops.object.delete()  # delete default cube

    mesh = make_box(1, 1, 1, [-np.pi / 2] * 4, (0.5, 0.5))
    box = make_object("Box", mesh)

    visible_vertices = get_visible_vertices(box)
    print(visible_vertices)