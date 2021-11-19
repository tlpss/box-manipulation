# data-generation
Tools to generate synthetic images of boxes rendered in Blender.

The code is seperated into three packages.
* `blender-utils`: high-level routines for interacting with Blender e.g. rotating objects, creating materials.
* `general-utils`: utilities that do not rely on Blender, e.g. generating the box geometry
* `box-generators`: scripts that generate and render specific scenes
