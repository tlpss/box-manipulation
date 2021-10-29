import numpy as np
from scipy.spatial.transform import Rotation


def rotate_vertex(v, rotation_origin, rotation_axis, angle):
    unit_axis = rotation_axis / np.linalg.norm(rotation_axis)
    r = Rotation.from_rotvec(angle * unit_axis)
    v_new = r.as_matrix() @ (v - rotation_origin) + rotation_origin
    return v_new


def add_flap(edge, flap_length, angle, vertices, edges, faces):
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
    edges += [(id0, id2), (id2, id3), (id3, id1), (id1, id0)]
    faces += [(id0, id2, id3, id1)]


def make_box(box_length, box_width, box_height, angles):
    l, w, h = box_length, box_width, box_height

    # Bottom face
    vertices = [  # 0, 1, 2, 3
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, l, 0.0]),
        np.array([w, l, 0.0]),
        np.array([w, 0.0, 0.0]),
    ]
    edges = [(0, 1), (1, 2), (2, 3), (0, 3)]
    faces = [(0, 1, 2, 3)]

    # Side faces
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

    for edge, angle in zip(rim_edges, angles):
        add_flap(edge, 0.05, angle, vertices, edges, faces)

    return vertices, edges, faces
