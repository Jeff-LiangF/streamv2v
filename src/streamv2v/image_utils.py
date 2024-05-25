from typing import List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
import torchvision


def denormalize(images: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    """
    Denormalize an image array to [0,1].
    """
    return (images / 2 + 0.5).clamp(0, 1)


def pt_to_numpy(images: torch.Tensor) -> np.ndarray:
    """
    Convert a PyTorch tensor to a NumPy image.
    """
    images = images.cpu().permute(0, 2, 3, 1).float().numpy()
    return images


def numpy_to_pil(images: np.ndarray) -> PIL.Image.Image:
    """
    Convert a NumPy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [
            PIL.Image.fromarray(image.squeeze(), mode="L") for image in images
        ]
    else:
        pil_images = [PIL.Image.fromarray(image) for image in images]

    return pil_images


def postprocess_image(
    image: torch.Tensor,
    output_type: str = "pil",
    do_denormalize: Optional[List[bool]] = None,
) -> Union[torch.Tensor, np.ndarray, PIL.Image.Image]:
    if not isinstance(image, torch.Tensor):
        raise ValueError(
            f"Input for postprocessing is in incorrect format: {type(image)}. We only support pytorch tensor"
        )

    if output_type == "latent":
        return image

    do_normalize_flg = True
    if do_denormalize is None:
        do_denormalize = [do_normalize_flg] * image.shape[0]

    image = torch.stack(
        [
            denormalize(image[i]) if do_denormalize[i] else image[i]
            for i in range(image.shape[0])
        ]
    )

    if output_type == "pt":
        return image

    image = pt_to_numpy(image)

    if output_type == "np":
        return image

    if output_type == "pil":
        return numpy_to_pil(image)


def process_image(
    image_pil: PIL.Image.Image, range: Tuple[int, int] = (-1, 1)
) -> Tuple[torch.Tensor, PIL.Image.Image]:
    image = torchvision.transforms.ToTensor()(image_pil)
    r_min, r_max = range[0], range[1]
    image = image * (r_max - r_min) + r_min
    return image[None, ...], image_pil


def pil2tensor(image_pil: PIL.Image.Image) -> torch.Tensor:
    height = image_pil.height
    width = image_pil.width
    imgs = []
    img, _ = process_image(image_pil)
    imgs.append(img)
    imgs = torch.vstack(imgs)
    images = torch.nn.functional.interpolate(
        imgs, size=(height, width), mode="bilinear"
    )
    image_tensors = images.to(torch.float16)
    return image_tensors

### Optical flow utils

def coords_grid(b, h, w, homogeneous=False, device=None):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w))  # [H, W]

    stacks = [x, y]

    if homogeneous:
        ones = torch.ones_like(x)  # [H, W]
        stacks.append(ones)

    grid = torch.stack(stacks, dim=0).float()  # [2, H, W] or [3, H, W]

    grid = grid[None].repeat(b, 1, 1, 1)  # [B, 2, H, W] or [B, 3, H, W]

    if device is not None:
        grid = grid.to(device)

    return grid

def flow_warp(feature, flow, mask=False, padding_mode='zeros'):
    b, c, h, w = feature.size()
    assert flow.size(1) == 2

    grid = coords_grid(b, h, w).to(flow.device) + flow  # [B, 2, H, W]

    return bilinear_sample(feature, grid, padding_mode=padding_mode,
                           return_mask=mask)

def bilinear_sample(img, sample_coords, mode='bilinear', padding_mode='zeros', return_mask=False):
    # img: [B, C, H, W]
    # sample_coords: [B, 2, H, W] in image scale
    if sample_coords.size(1) != 2:  # [B, H, W, 2]
        sample_coords = sample_coords.permute(0, 3, 1, 2)

    b, _, h, w = sample_coords.shape

    # Normalize to [-1, 1]
    x_grid = 2 * sample_coords[:, 0] / (w - 1) - 1
    y_grid = 2 * sample_coords[:, 1] / (h - 1) - 1

    grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, H, W, 2]

    img = torch.nn.functional.grid_sample(img, grid, mode=mode, padding_mode=padding_mode, align_corners=True)

    if return_mask:
        mask = (x_grid >= -1) & (y_grid >= -1) & (x_grid <= 1) & (y_grid <= 1)  # [B, H, W]

        return img, mask

    return img

def forward_backward_consistency_check(fwd_flow, bwd_flow,
                                       alpha=0.1,
                                       beta=0.5
                                       ):
    # fwd_flow, bwd_flow: [B, 2, H, W]
    # alpha and beta values are following UnFlow (https://arxiv.org/abs/1711.07837)
    assert fwd_flow.dim() == 4 and bwd_flow.dim() == 4
    assert fwd_flow.size(1) == 2 and bwd_flow.size(1) == 2
    flow_mag = torch.norm(fwd_flow, dim=1) + torch.norm(bwd_flow, dim=1)  # [B, H, W]

    warped_bwd_flow = flow_warp(bwd_flow, fwd_flow)  # [B, 2, H, W]
    warped_fwd_flow = flow_warp(fwd_flow, bwd_flow)  # [B, 2, H, W]

    diff_fwd = torch.norm(fwd_flow + warped_bwd_flow, dim=1)  # [B, H, W]
    diff_bwd = torch.norm(bwd_flow + warped_fwd_flow, dim=1)

    threshold = alpha * flow_mag + beta

    fwd_occ = (diff_fwd > threshold).float()  # [B, H, W]
    bwd_occ = (diff_bwd > threshold).float()

    return fwd_occ, bwd_occ