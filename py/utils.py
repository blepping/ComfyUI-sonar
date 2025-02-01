from __future__ import annotations

import math

import torch
from comfy.model_management import device_supports_non_blocking
from comfy.utils import common_upscale

from .external import MODULES as EXT

BLENDING_MODES = {"lerp": torch.lerp}
UPSCALE_METHODS = (
    "bilinear",
    "nearest-exact",
    "nearest",
    "area",
    "bicubic",
    "bislerp",
)


def scale_samples(
    samples: torch.Tensor,
    width: int,
    height: int,
    *,
    mode: str = "bicubic",
) -> torch.Tensor:
    return common_upscale(samples, width, height, mode, None)


def init_integrations(integrations) -> None:
    global scale_samples, BLENDING_MODES, UPSCALE_METHODS  # noqa: PLW0603

    bleh = integrations.bleh
    if bleh is None:
        return
    bleh_latentutils = bleh.py.latent_utils
    BLENDING_MODES = bleh_latentutils.BLENDING_MODES
    UPSCALE_METHODS = bleh_latentutils.UPSCALE_METHODS
    scale_samples = bleh_latentutils.scale_samples


EXT.register_init_handler(init_integrations)


def scale_noise(
    noise: torch.Tensor,
    factor: float = 1.0,
    *,
    normalized: bool = True,
    threshold_std_devs: float = 2.5,
    normalize_dims: tuple | None = None,
) -> torch.Tensor:
    numel = noise.numel()
    if not normalized or numel == 0:
        return noise.mul_(factor) if factor != 1 else noise
    if normalize_dims is not None:
        std = noise.std(dim=normalize_dims, keepdim=True)
        noise = noise / std  # noqa: PLR6104
        return noise.sub_(noise.mean(dim=normalize_dims, keepdim=True)).mul_(factor)
    mean, std = noise.mean().item(), noise.std().item()
    threshold = threshold_std_devs / math.sqrt(numel)
    if abs(mean) > threshold:
        noise -= mean
    if abs(1.0 - std) > threshold:
        noise /= std
    return noise.mul_(factor) if factor != 1 else noise


CAN_NONBLOCK = {}


def tensor_to(
    tensor: torch.Tensor,
    dest: torch.Tensor | torch.Device | str,
) -> torch.Tensor:
    device = dest.device if isinstance(dest, torch.Tensor) else dest
    non_blocking = CAN_NONBLOCK.get(device)
    if non_blocking is None:
        non_blocking = device_supports_non_blocking(device)
        CAN_NONBLOCK[device] = non_blocking
    return tensor.to(dest, non_blocking=non_blocking)


def quantile_normalize(
    noise: torch.Tensor,
    *,
    quantile: float = 0.75,
    dim: int | None = 1,
    flatten: bool = True,
    nq_fac: float = 1.0,
    pow_fac: float = 0.5,
) -> torch.Tensor:
    if quantile is None or quantile <= 0 or quantile >= 1:
        return noise
    orig_shape = noise.shape
    if isinstance(quantile, (tuple, list)):
        quantile = torch.tensor(
            quantile,
            device=noise.device,
            dtype=noise.dtype,
        )
    qdim = dim
    if noise.ndim > 1 and flatten:
        if qdim is not None and qdim >= noise.ndim:
            qdim = 1 if noise.ndim > 2 else None
        if qdim is None:
            flatdim = 0
        elif qdim in {0, 1}:
            flatdim = qdim + 1
        elif qdim in {2, 3}:
            noise = noise.movedim(qdim, 1)
            tempshape = noise.shape
            flatdim = 2
        else:
            raise ValueError(
                "Cannot handling quantile normalization flattening dims > 3",
            )
    else:
        flatdim = None
    nq = torch.quantile(
        (noise if flatdim is None else noise.flatten(start_dim=flatdim)).abs(),
        quantile,
        dim=-1,
    )
    nq_shape = tuple(nq.shape) + (1,) * (noise.ndim - nq.ndim)
    nq = nq.mul_(nq_fac).reshape(*nq_shape)
    noise = noise.clamp(-nq, nq)
    noise = torch.copysign(
        torch.pow(torch.abs(noise), pow_fac),
        noise,
    )
    if flatdim is not None and qdim in {2, 3}:
        return (
            noise.reshape(tempshape).movedim(1, qdim).reshape(orig_shape).contiguous()
        )
    return noise


def adjust_slice(s: slice, size: int, offset: int) -> slice:
    if offset == 0:
        return s
    # Input slice must have positive start/stop and be in bounds for the object that will be sliced here.
    start = s.start if s.start is not None else 0
    stop = s.stop if s.stop is not None else size
    if offset < 0:
        adj = min(start, abs(offset))
        return slice(start - adj, stop - adj)
    adj = min(size - stop, offset)
    return slice(start + adj, stop + adj)


def crop_samples(
    tensor: torch.Tensor,
    width: int,
    height: int,
    *,
    mode="center",
    offset_width: int = 0,
    offset_height: int = 0,
):
    if tensor.ndim < 3:
        raise ValueError("Can only handle >= 3 dimensional tensors")
    th, tw = tensor.shape[-2:]
    if (tw, th) == (width, height):
        return tensor
    if tw < width or th < height:
        raise ValueError("Can't crop sample smaller than requested width or height")
    if mode == "center":
        hmode = wmode = "center"
    else:
        hmode, wmode, *splitextra = mode.split("_")
        if splitextra:
            raise ValueError("Bad composite mode")
    if hmode == "top":
        hslice = slice(0, height)
    elif hmode == "center":
        hoffs = (th - height) // 2
        hslice = slice(hoffs, hoffs + height)
    elif hmode == "bottom":
        hslice = slice(th - height, th)
    else:
        raise ValueError("Bad height mode in composite mode")
    if wmode == "left":
        wslice = slice(0, width)
    elif wmode == "center":
        woffs = (tw - width) // 2
        wslice = slice(woffs, woffs + width)
    elif wmode == "right":
        wslice = slice(tw - width, tw)
    else:
        raise ValueError("Bad width mode in composite mode")
    wslice = adjust_slice(wslice, tw, offset_width)
    hslice = adjust_slice(hslice, th, offset_height)
    return tensor[..., hslice, wslice]
