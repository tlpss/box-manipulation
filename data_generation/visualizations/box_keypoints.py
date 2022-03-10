from box.scene_generators.flaps_open_camera_vertical import generate_scene

box = generate_scene(seed=0)
box.visualize_keypoints(radius=0.02)
