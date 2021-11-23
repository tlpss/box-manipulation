from typing import List, Tuple, Union

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from skimage.feature import peak_local_max


def gaussian_heatmap(
    image_size: Tuple[int, int], center: Union[Tuple[int, int], List[int]], sigma: torch.Tensor
) -> torch.Tensor:
    """
    Creates a Gaussian blob heatmap for a single keypoint.
    The coordinate system is a left-top corner origin with u going left and v going down.

    Args:
        image_size (Tuple(int,int)): image_size (height, width) ! note convention to match store-order and tensor dimensions
        center (Tuple(int,int)): center coordinate (cX,cY) (U,V) ! note: U,V order
        sigma (torch.Tensor): standard deviation of the gaussian blob

    Returns:
        Torch.Tensor: A tensor with zero background, specified size and a Gaussian heatmap around the center.
    """

    u_axis = torch.linspace(0, image_size[1] - 1, image_size[1]) - center[0]
    v_axis = torch.linspace(0, image_size[0] - 1, image_size[0]) - center[1]
    # create grid values in 2D with x and y coordinate centered aroud the keypoint
    xx, yy = torch.meshgrid(v_axis, u_axis)

    ## create gaussian around the centered 2D grids $ exp ( -0.5 (x**2 + y**2) / sigma**2)$
    heatmap = torch.exp(-0.5 * (torch.square(xx) + torch.square(yy)) / torch.square(sigma))
    return heatmap


def generate_keypoints_heatmap(
    image_size: Tuple[int, int], keypoints: List[Tuple[int, int]], sigma: float
) -> torch.Tensor:
    """
    Generates heatmap with gaussian blobs for each keypoint, using the given sigma.
    Max operation is used to combine the heatpoints to avoid local optimum surpression.
    Origin is topleft corner and u goes right, v down.

    Args:
        image_size: Tuple(int,int) that specify (H,W) of the heatmap image
        keypoints: List(Tuple(int,int), ...) with keypoints (u,v).
        sigma: (float) std deviation of the blobs

    Returns:
         Torch.tensor:  A Tensor with the combined heatmaps of all keypoints.
    """

    img = torch.zeros(image_size)  # (h,w) dimensions
    sigma = torch.Tensor([sigma])
    for keypoint in keypoints:
        new_img = gaussian_heatmap(image_size, keypoint, sigma)
        img = torch.maximum(img, new_img)  # piecewise max of 2 Tensors
    return img


def get_keypoints_from_heatmap(heatmap: torch.Tensor, min_keypoint_pixel_distance: int) -> List[List[int]]:
    """
    Extracts all keypoints from a heatmap, where each keypoint is defined as being a local maximum within a 2D mask [ -min_pixel_distance, + pixel_distance]^2
    cf https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.peak_local_max

    Args:
        heatmap (torch.Tensor): heatmap image
        min_keypoint_pixel_distance (int): The size of the local mask

    Returns:
        List(List(x,y)...): A list of 2D keypoints
    """

    np_heatmap = heatmap.cpu().numpy()
    keypoints = peak_local_max(np_heatmap, min_distance=min_keypoint_pixel_distance)
    return keypoints[::, ::-1].tolist()  # convert to (u,v) aka (col,row) coord frame from (row,col)


def overlay_image_with_heatmap(img: torch.Tensor, heatmap: torch.Tensor, alpha=0.4) -> Image:
    """
    Overlays image with the predicted heatmap, which is projected from grayscale to the red channel.
    """
    # Create heatmap image in red channel
    heatmap = torch.cat((heatmap, torch.zeros(2, img.shape[1], img.shape[2])))
    img = TF.to_pil_image(img)  # assuming your image in x
    h_img = TF.to_pil_image(heatmap)

    overlay = Image.blend(img, h_img, alpha)

    return overlay
