from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation


def rotate_vertex(v, rotation_origin, rotation_axis, angle):
    unit_axis = rotation_axis / np.linalg.norm(rotation_axis)
    r = Rotation.from_rotvec(angle * unit_axis)
    v_new = r.as_matrix() @ (v - rotation_origin) + rotation_origin
    return v_new


def add_flap(edge: Tuple[int, int], flap_length: float, angle: float, vertices, edges, faces):
    """Add the necessary vertices, edges and face to a mesh to create a flap.
    Note that this function modifies the lists its receives as inputs.

    Args:
        edge ([type]): the edge to which the flap will be attached.
        flap_length ([type]): the size of the flap.
        angle ([type]): the angle at which the flap will be placed. 0 is straight up, positive is the right hand direction for the edge as vector.
        vertices ([type]): vertices of the mesh.
        edges ([type]): edges of the mesh.
        faces ([type]): faces of the mesh.
    """
    id0, id1 = edge
    v0 = vertices[id0]
    v1 = vertices[id1]

    id2 = len(vertices)
    id3 = id2 + 1

    v2 = np.copy(v0)
    v3 = np.copy(v1)
    v2[2] += flap_length
    v3[2] += flap_length

    v2 = rotate_vertex(v2, v0, v1 - v0, angle)
    v3 = rotate_vertex(v3, v1, v1 - v0, angle)

    vertices += [v2, v3]
    edges += [(id0, id2), (id2, id3), (id3, id1)]
    faces += [(id0, id2, id3, id1)]


def make_box(
    box_length: float,
    box_width: float,
    box_height: float,
    flap_angles: Tuple[float, float, float, float],
    flap_size_fractions: Tuple[float, float],
):
    """Generate the geometry (a mesh) of a simple rectangular box with 4 rectangular flaps.
     The opposing flaps are always the same length.

    Args:
        box_length (float): The extent of the box along the y-axis.
        box_width (float): The extent of the box along the x-axis.
        box_height (float): The height of the box.
        flap_angles (Tuple[float]): The 4 angles of the flaps in radians. 0 is straight up, positive is inwards.
        flap_size_fractions (Tuple[float]): The sizes of the flaps as a fractions of the distance to the other side of the box.
                                            Expects two fraction, one for the flaps attached to the width and one for those attached to the length.
                                            Usually between 0.25 and 0.5.

    Returns:
        (vertices, edges, faces): the mesh as a tuple of lists of vertices, edges and faces.
    """
    l, w, h = float(box_length), float(box_width), float(box_height)

    # Bottom face
    #  0 ________l_______ 1  ---> y
    #    |              |
    #    |              |
    #    w              |
    #    |              |
    #  3 |______________| 2
    #
    #    |
    #    v  x
    vertices = [  # 0, 1, 2, 3
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, l, 0.0]),
        np.array([w, l, 0.0]),
        np.array([w, 0.0, 0.0]),
    ]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    faces = [(0, 1, 2, 3)]

    # Adding the side faces
    #   4 +------+.  5
    #     |`.    | `.
    #     |7 `+--+---+  6
    #     |   |  |   |
    #   0 +---+--+.  | 1
    #      `. |    `.|
    #      3 `+------+  2

    vertices += [  # 4, 5, 6, 7
        np.array([0.0, 0.0, h]),
        np.array([0.0, l, h]),
        np.array([w, l, h]),
        np.array([w, 0.0, h]),
    ]

    rim_edges = [(4, 5), (5, 6), (6, 7), (7, 4)]
    standing_edges = [(0, 4), (1, 5), (2, 6), (3, 7)]
    edges += rim_edges + standing_edges
    faces += [(0, 4, 5, 1), (1, 5, 6, 2), (2, 3, 7, 6), (0, 4, 7, 3)]

    # Adding the flaps
    frac0, frac1 = flap_size_fractions
    length_flaps_size = w * frac0
    width_flaps_size = l * frac1
    flap_lengths = [length_flaps_size, width_flaps_size] * 2

    for edge, angle, length in zip(rim_edges, flap_angles, flap_lengths):
        add_flap(edge, length, angle, vertices, edges, faces)

    # Shift all vertices so object origin will be at world origin.
    for v in vertices:
        v -= np.array([w / 2.0, l / 2.0, 0])

    return vertices, edges, faces


def inside_box(angle):
    return np.pi / 2 <= angle <= np.pi


def above_box(angle):
    return 0 <= angle <= np.pi / 2


def flap_angles_collision_free(angles):
    for i in range(angles.shape[0] - 1):
        a = angles[i]
        a_prev = angles[i - 1]
        a_next = angles[i + 1]

        if inside_box(a) and (inside_box(a_prev) or inside_box(a_next)):
            return False

        if above_box(a) and (above_box(a_prev) or above_box(a_next)):
            return False
    return True


def random_flap_angles():
    while True:
        angles = np.random.uniform(-np.pi, np.pi, 4)
        if flap_angles_collision_free(angles):
            return angles


if __name__ == "__main__":
    mesh = make_box(1.0, 0.5, 0.25, random_flap_angles(), (0.5, 0.25))
    vertices, edges, faces = mesh
    print(f"Vertices ({len(vertices)}) \n {vertices} \n")
    print(f"Edges ({len(edges)}) \n {edges} \n")
    print(f"Faces ({len(faces)}) \n{ faces} \n")
