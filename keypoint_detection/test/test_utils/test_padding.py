import unittest

import torch

from keypoint_detection.utils.tensor_padding import pad_tensor_with_nans, unpad_nans_from_tensor


class TestPadding(unittest.TestCase):
    def test_padd(self):
        tensor = torch.tensor([[1.0, 2, 3], [3.2, 4, 5]])

        padded_tensor = pad_tensor_with_nans(tensor, 4)
        self.assertEqual(padded_tensor.shape, (4, 3))
        unpadded_tensor = unpad_nans_from_tensor(padded_tensor)
        self.assertAlmostEqual(torch.sum(tensor - unpadded_tensor), 0)

    def test_pad_full_size_tensor(self):
        tensor = torch.tensor([[1.0, 2, 3], [3.2, 4, 5]])

        padded_tensor = pad_tensor_with_nans(tensor, 2)
        self.assertEqual(padded_tensor.shape, (2, 3))
        unpadded_tensor = unpad_nans_from_tensor(padded_tensor)
        self.assertAlmostEqual(torch.sum(tensor - unpadded_tensor), 0)
