import torch

from keypoint_detection.data.dataset import BoxKeypointsDataset


class UnlabeledBoxDataset(BoxKeypointsDataset):
    def __init__(self, json_file: str, image_dir: str, flap_keypoints_type: str = "corner"):
        super().__init__(json_file, image_dir, flap_keypoints_type)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        index = int(index)

        image = self.get_image(index)
        image = self.transform(image)

        return image
