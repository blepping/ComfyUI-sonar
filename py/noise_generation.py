# Noise generation functions shamelessly yoinked from https://github.com/Clybius/ComfyUI-Extra-Samplers
from __future__ import annotations

import math
from enum import Enum, auto
from typing import Callable

import torch
from comfy.utils import common_upscale
from torch import FloatTensor, Generator, Tensor
from torch.distributions import Laplace, StudentT

# ruff: noqa: D412, D413, D417, D212, D407, ANN002, ANN003, FBT001, FBT002, S311


class NoiseType(Enum):
    GAUSSIAN = auto()
    UNIFORM = auto()
    BROWNIAN = auto()
    PERLIN = auto()
    STUDENTT = auto()
    HIGHRES_PYRAMID = auto()
    PYRAMID = auto()
    PYRAMID_MIX = auto()
    PINK = auto()
    LAPLACIAN = auto()
    POWER = auto()
    RAINBOW_MILD = auto()
    RAINBOW_INTENSE = auto()
    GREEN_TEST = auto()
    PYRAMID_OLD = auto()
    PYRAMID_BISLERP = auto()
    HIGHRES_PYRAMID_BISLERP = auto()
    PYRAMID_OLD_BISLERP = auto()
    PYRAMID_OLD_AREA = auto()
    PYRAMID_AREA = auto()
    HIGHRES_PYRAMID_AREA = auto()
    PYRAMID_DISCOUNT5 = auto()
    PYRAMID_MIX_BISLERP = auto()
    PYRAMID_MIX_AREA = auto()

    @classmethod
    def get_names(cls, default=None, skip=None):
        if default is not None:
            yield default.name.lower()
        for nt in cls:
            if nt == default or (skip and nt in skip):
                continue
            yield nt.name.lower()


class NoiseError(Exception):
    pass


def scale_noise(noise, factor=1.0, *, normalized=True, threshold_std_devs=2.5):
    if not normalized or noise.numel() == 0:
        return noise.mul_(factor) if factor != 1 else noise
    mean, std = noise.mean().item(), noise.std().item()
    threshold = threshold_std_devs / math.sqrt(noise.numel())
    if abs(mean) > threshold:
        noise -= mean
    if abs(1.0 - std) > threshold:
        noise /= std
    return noise.mul_(factor) if factor != 1 else noise


def get_positions(block_shape: tuple[int, int]) -> Tensor:
    """
    Generate position tensor.

    Arguments:
        block_shape -- (height, width) of position tensor

    Returns:
        position vector shaped (1, height, width, 1, 1, 2)
    """
    bh, bw = block_shape
    return torch.stack(
        torch.meshgrid(
            [(torch.arange(b) + 0.5) / b for b in (bw, bh)],
            indexing="xy",
        ),
        -1,
    ).view(1, bh, bw, 1, 1, 2)


def unfold_grid(vectors: Tensor) -> Tensor:
    """
    Unfold vector grid to batched vectors.

    Arguments:
        vectors -- grid vectors

    Returns:
        batched grid vectors
    """
    batch_size, _, gpy, gpx = vectors.shape
    return (
        torch.nn.functional.unfold(vectors, (2, 2))
        .view(batch_size, 2, 4, -1)
        .permute(0, 2, 3, 1)
        .view(batch_size, 4, gpy - 1, gpx - 1, 2)
    )


def smooth_step(t: Tensor) -> Tensor:
    """
    Smooth step function [0, 1] -> [0, 1].

    Arguments:
        t -- input values (any shape)

    Returns:
        output values (same shape as input values)
    """
    return t * t * (3.0 - 2.0 * t)


def perlin_noise_tensor(
    vectors: Tensor,
    positions: Tensor,
    step: Callable | None = None,
) -> Tensor:
    """
    Generate perlin noise from batched vectors and positions.

    Arguments:
        vectors -- batched grid vectors shaped (batch_size, 4, grid_height, grid_width, 2)
        positions -- batched grid positions shaped (batch_size or 1, block_height, block_width, grid_height or 1, grid_width or 1, 2)

    Keyword Arguments:
        step -- smooth step function [0, 1] -> [0, 1] (default: `smooth_step`)

    Raises:
        NoiseError: if position and vector shapes do not match

    Returns:
        (batch_size, block_height * grid_height, block_width * grid_width)
    """
    if step is None:
        step = smooth_step

    batch_size = vectors.shape[0]
    # grid height, grid width
    gh, gw = vectors.shape[2:4]
    # block height, block width
    bh, bw = positions.shape[1:3]

    for i in range(2):
        if positions.shape[i + 3] not in {1, vectors.shape[i + 2]}:
            msg = f"Blocks shapes do not match: vectors ({vectors.shape[1]}, {vectors.shape[2]}), positions {gh}, {gw})"
            raise NoiseError(msg)

    if positions.shape[0] not in {1, batch_size}:
        msg = f"Batch sizes do not match: vectors ({vectors.shape[0]}), positions ({positions.shape[0]})"
        raise NoiseError(msg)

    vectors = vectors.view(batch_size, 4, 1, gh * gw, 2)
    positions = positions.view(positions.shape[0], bh * bw, -1, 2)

    step_x = step(positions[..., 0])
    step_y = step(positions[..., 1])

    row0 = torch.lerp(
        (vectors[:, 0] * positions).sum(dim=-1),
        (vectors[:, 1] * (positions - positions.new_tensor((1, 0)))).sum(dim=-1),
        step_x,
    )
    row1 = torch.lerp(
        (vectors[:, 2] * (positions - positions.new_tensor((0, 1)))).sum(dim=-1),
        (vectors[:, 3] * (positions - positions.new_tensor((1, 1)))).sum(dim=-1),
        step_x,
    )
    noise = torch.lerp(row0, row1, step_y)
    return (
        noise.view(
            batch_size,
            bh,
            bw,
            gh,
            gw,
        )
        .permute(0, 3, 1, 4, 2)
        .reshape(batch_size, gh * bh, gw * bw)
    )


def perlin_noise(
    grid_shape: tuple[int, int],
    out_shape: tuple[int, int],
    batch_size: int = 1,
    generator: Generator | None = None,
    *args,
    **kwargs,
) -> Tensor:
    """
    Generate perlin noise with given shape. `*args` and `**kwargs` are forwarded to `Tensor` creation.

    Arguments:
        grid_shape -- Shape of grid (height, width).
        out_shape -- Shape of output noise image (height, width).

    Keyword Arguments:
        batch_size -- (default: {1})
        generator -- random generator used for grid vectors (default: {None})

    Raises:
        NoiseError: if grid and out shapes do not match

    Returns:
        Noise image shaped (batch_size, height, width)
    """
    # grid height and width
    gh, gw = grid_shape
    # output height and width
    oh, ow = out_shape
    # block height and width
    bh, bw = oh // gh, ow // gw

    if oh != bh * gh:
        msg = f"Output height {oh} must be divisible by grid height {gh}"
        raise NoiseError(msg)
    if ow != bw * gw != 0:
        msg = f"Output width {ow} must be divisible by grid width {gw}"
        raise NoiseError(msg)

    angle = torch.empty(
        [batch_size] + [s + 1 for s in grid_shape],
        *args,
        **kwargs,
    ).uniform_(to=2.0 * math.pi, generator=generator)
    # random vectors on grid points
    vectors = unfold_grid(torch.stack((torch.cos(angle), torch.sin(angle)), dim=1))
    # positions inside grid cells [0, 1)
    positions = get_positions((bh, bw)).to(vectors)
    return perlin_noise_tensor(vectors, positions).squeeze(0)


def rand_perlin_like(x):
    noise = torch.randn_like(x) / 2.0
    noise_height = noise.size(dim=2)
    noise_width = noise.size(dim=3)
    for _ in range(2):
        noise += perlin_noise(
            (noise_height, noise_width),
            (noise_height, noise_width),
            batch_size=x.shape[1],  # This should be the number of channels.
        ).to(x.device)
    return scale_noise(noise)


def uniform_noise_like(x):
    return (torch.rand_like(x) - 0.5) * 3.46


def highres_pyramid_noise_like(x, discount=0.7, upscale_mode="bilinear"):
    (
        b,
        c,
        h,
        w,
    ) = x.shape  # EDIT: w and h get over-written, rename for a different variant!
    orig_w, orig_h = w, h
    noise = uniform_noise_like(x)
    rs = torch.rand(4, dtype=torch.float32) * 2 + 2
    for i in range(4):
        r = rs[i]
        h, w = min(orig_h * 15, int(h * (r**i))), min(orig_w * 15, int(w * (r**i)))
        noise += common_upscale(
            torch.randn(b, c, h, w).to(x),
            orig_w,
            orig_h,
            upscale_mode,
            None,
        ).mul_(discount**i)
        if h >= orig_h * 15 or w >= orig_w * 15:
            break  # Lowest resolution is 1x1
    return scale_noise(noise)


def pyramid_old_noise_like(
    x,
    generator=None,
    device="cpu",
    discount=0.8,
    upscale_mode="nearest-exact",
):
    size = x.size()
    b, c, h, w = size
    orig_h, orig_w = h, w
    noise = torch.zeros(size=size, dtype=x.dtype, layout=x.layout, device=device)
    r = 1
    for i in range(5):
        r *= 2
        noise += common_upscale(
            torch.normal(
                mean=0,
                std=0.5**i,
                size=(b, c, h * r, w * r),
                dtype=x.dtype,
                layout=x.layout,
                generator=generator,
                device=device,
            ),
            orig_w,
            orig_h,
            upscale_mode,
            None,
        ).mul_(discount**i)
    return noise.to(device=x.device)


# Copied from https://wandb.ai/johnowhitaker/multires_noise/reports/Multi-Resolution-Noise-for-Diffusion-Model-Training--VmlldzozNjYyOTU2
def pyramid_noise_like(x, discount=0.7, upscale_mode="bilinear"):
    b, c, w, h = (
        x.shape
    )  # NOTE: w and h get over-written, rename for a different variant!
    orig_w, orig_h = w, h
    noise = torch.randn_like(x)
    for i in range(10):
        r = torch.rand(1, device="cpu").item() * 2 + 2  # Rather than always going 2x,
        w, h = max(1, int(w / (r**i))), max(1, int(h / (r**i)))
        noise += common_upscale(
            torch.randn(b, c, w, h).to(x),
            orig_h,
            orig_w,
            upscale_mode,
            None,
        ).mul_(
            discount**i,
        )
        if w == 1 or h == 1:
            break  # Lowest resolution is 1x1
    return scale_noise(noise)


def studentt_noise_like(x):
    noise = StudentT(loc=0, scale=0.2, df=1).rsample(x.size())
    s: FloatTensor = torch.quantile(noise.flatten(start_dim=1).abs(), 0.75, dim=-1)
    s = s.reshape(*s.shape, 1, 1, 1)
    noise = noise.clamp(-s, s)
    return torch.copysign(torch.pow(torch.abs(noise), 0.5), noise)


def green_noise_like(x):
    # The comments said this didn't work and I had to learn the hard way. Turns out it's true!
    width, height = x.size(dim=2), x.size(dim=3)
    noise = torch.randn_like(x)
    scale = 1.0 / (width * height)
    fy = torch.fft.fftfreq(width, device=x.device)[:, None] ** 2
    fx = torch.fft.fftfreq(height, device=x.device) ** 2
    f = fy + fx
    power = torch.sqrt(f)
    power[0, 0] = 1
    noise = torch.fft.ifft2(torch.fft.fft2(noise) / torch.sqrt(power))
    noise *= scale / noise.std()
    noise = torch.real(noise).to(x.device)
    return scale_noise(noise)


def generate_1f_noise(tensor, alpha, k, generator=None):
    """Generate 1/f noise for a given tensor.

    Args:
        tensor: The tensor to add noise to.
        alpha: The parameter that determines the slope of the spectrum.
        k: A constant.

    Returns:
        A tensor with the same shape as `tensor` containing 1/f noise.
    """
    fft = torch.fft.fft2(tensor)
    freq = torch.arange(1, len(fft) + 1, dtype=torch.float)
    spectral_density = k / freq**alpha
    return torch.randn(tensor.shape, generator=generator) * spectral_density


def pink_noise_like(x):
    return scale_noise(generate_1f_noise(x, 2.0, 1.0)).to(x.device)


def laplacian_noise_like(x):
    noise = torch.randn_like(x).div_(4.0)
    noise += Laplace(loc=0, scale=1.0).rsample(x.size()).to(noise.device)
    return scale_noise(noise)


def power_noise_like(tensor, alpha=2, k=1):  # This doesn't work properly right now
    """Generate 1/f noise for a given tensor.

    Args:
        tensor: The tensor to add noise to.
        alpha: The parameter that determines the slope of the spectrum.
        k: A constant.

    Returns:
        A tensor with the same shape as `tensor` containing 1/f noise.
    """
    tensor = torch.randn_like(tensor)
    fft = torch.fft.fft2(tensor)
    freq = torch.arange(1, len(fft) + 1, dtype=torch.float).reshape(
        (len(fft),) + (1,) * (tensor.dim() - 1),
    )
    spectral_density = k / freq**alpha
    noise = torch.rand(tensor.shape).mul_(spectral_density)
    mean = torch.mean(noise, dim=(-2, -1), keepdim=True).to(tensor.device)
    std = torch.std(noise, dim=(-2, -1), keepdim=True).to(tensor.device)
    return noise.to(tensor.device).sub_(mean).div_(std)


__all__ = (
    "NoiseError",
    "NoiseType",
    "green_noise_like",
    "highres_pyramid_noise_like",
    "laplacian_noise_like",
    "pink_noise_like",
    "power_noise_like",
    "pyramid_noise_like",
    "pyramid_old_noise_like",
    "rand_perlin_like",
    "scale_noise",
    "studentt_noise_like",
    "uniform_noise_like",
)
