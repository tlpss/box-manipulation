import bpy


def random_box_material():
    box_color = [0.401969, 0.175731, 0.067917, 1.000000]
    box_material = bpy.data.materials.new(name="Box")
    box_material.diffuse_color = box_color
    box_material.use_nodes = True
    box_bsdf = box_material.node_tree.nodes["Principled BSDF"]
    box_bsdf.inputs["Base Color"].default_value = box_color
    box_bsdf.inputs["Roughness"].default_value = 0.9
    return box_material
