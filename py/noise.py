# Noise generation functions shamelessly yoinked from https://github.com/Clybius/ComfyUI-Extra-Samplers
from __future__ import annotations

import functools as fun
import math
import operator as op
from enum import Enum, auto
from typing import Callable

import torch
from comfy.k_diffusion import sampling
from torch import FloatTensor, Generator, Tensor

# ruff: noqa: D412, D413, D417, D212, D407, ANN002, ANN003, FBT001, FBT002, S311


def scale_noise(noise, factor=1.0):
    mean, std = noise.mean(), noise.std()
    # print(f"NOISE * {factor:.3}: std={std:.3}, mean={mean:.3}")
    return (noise - mean).div_(std).mul_(factor)


class NoiseType(Enum):
    GAUSSIAN = auto()
    UNIFORM = auto()
    BROWNIAN = auto()
    PERLIN = auto()
    STUDENTT = auto()
    HIGHRES_PYRAMID = auto()
    PINK = auto()
    LAPLACIAN = auto()
    POWER = auto()
    RAINBOW_MILD = auto()
    # RAINBOW_MILD2 = auto()
    RAINBOW_INTENSE = auto()
    # RAINBOW_INTENSE2 = auto()
    # RAINBOW_INTENSE3 = auto()
    GREEN_TEST = auto()


class NoiseError(Exception):
    pass


class CustomNoiseItem:
    def __init__(self, factor, noise_type):
        self.factor = factor
        self.noise_type = noise_type


class CustomNoiseChain:
    def __init__(self, items=None):
        self.items = items if items is not None else []

    def clone(self):
        return CustomNoiseChain(
            [CustomNoiseItem(i.factor, i.noise_type) for i in self.items],
        )

    def add(self, item):
        self.items.append(item)

    def rescaled(self, scale=1.0):
        total = sum(i.factor for i in self.items)
        divisor = total / scale
        divisor = divisor if divisor != 0 else 1.0
        return CustomNoiseChain(
            [CustomNoiseItem(i.factor / divisor, i.noise_type) for i in self.items],
        )

    def __call__(
        self,
        x: Tensor,
        sigma_min=None,
        sigma_max=None,
        seed=None,
        transform=lambda x: x,
        cpu=True,
    ):
        pass

    @torch.no_grad()
    def make_noise_sampler(
        self,
        x: Tensor,
        sigma_min=None,
        sigma_max=None,
        seed=None,
        cpu=True,
    ) -> Callable:
        noise_samplers = tuple(
            get_noise_sampler(
                i.noise_type,
                x,
                sigma_min,
                sigma_max,
                seed=seed,
                cpu=cpu,
                factor=i.factor,
            )
            for i in self.items
        )
        if not noise_samplers or not all(noise_samplers):
            raise ValueError("Failed to get noise sampler")
        scale = sum(i.factor for i in self.items)

        def noise_sampler(sigma, sigma_next):
            result = fun.reduce(
                op.add,
                (ns(sigma, sigma_next) for ns in noise_samplers),
            )
            return scale_noise(result, scale)

        return noise_sampler


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
        Exception: if position and vector shapes do not match

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
        if positions.shape[i + 3] not in (1, vectors.shape[i + 2]):
            msg = f"Blocks shapes do not match: vectors ({vectors.shape[1]}, {vectors.shape[2]}), positions {gh}, {gw})"
            raise NoiseError(msg)

    if positions.shape[0] not in (1, batch_size):
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
        Exception: if grid and out shapes do not match

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
    return noise / noise.std()


def uniform_noise_like(x):
    return (torch.rand_like(x) - 0.5) * 3.46


def highres_pyramid_noise_like(x, discount=0.7):
    (
        b,
        c,
        h,
        w,
    ) = x.shape  # EDIT: w and h get over-written, rename for a different variant!
    orig_h = h
    orig_w = w
    u = torch.nn.Upsample(size=(orig_h, orig_w), mode="bilinear")
    noise = uniform_noise_like(x)
    rs = torch.rand(4, dtype=torch.float32) * 2 + 2
    for i in range(4):
        r = rs[i]
        h, w = min(orig_h * 15, int(h * (r**i))), min(orig_w * 15, int(w * (r**i)))
        noise += u(torch.randn(b, c, h, w).to(x)) * discount**i
        if h >= orig_h * 15 or w >= orig_w * 15:
            break  # Lowest resolution is 1x1
    return noise / noise.std()  # Scaled back to roughly unit variance


def studentt_noise_like(x):
    from torch.distributions import StudentT

    noise = StudentT(loc=0, scale=0.2, df=1).rsample(x.size())
    s: FloatTensor = torch.quantile(noise.flatten(start_dim=1).abs(), 0.75, dim=-1)
    s = s.reshape(*s.shape, 1, 1, 1)
    noise = noise.clamp(-s, s)
    return torch.copysign(torch.pow(torch.abs(noise), 0.5), noise)


def studentt_noise_sampler(
    x,
):  # Produces more subject-focused outputs due to distribution, unsure if this works
    noise = studentt_noise_like(x)
    return lambda _sigma, _sigma_next: noise.to(x.device) / (7 / 3)


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
    return noise / noise.std()


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
    noise = generate_1f_noise(x, 2.0, 1.0)
    noise_mean = torch.mean(noise)
    noise_std = torch.std(noise)
    return noise.sub_(noise_mean).div_(noise_std).to(x.device)


def laplacian_noise_like(x):
    from torch.distributions import Laplace

    noise = torch.randn_like(x) / 4.0
    noise += Laplace(loc=0, scale=1.0).rsample(x.size()).to(noise.device)
    return noise / noise.std()


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
    noise = torch.rand(tensor.shape) * spectral_density
    mean = torch.mean(noise, dim=(-2, -1), keepdim=True).to(tensor.device)
    std = torch.std(noise, dim=(-2, -1), keepdim=True).to(tensor.device)
    return noise.to(tensor.device).sub_(mean).div_(std)


class NoiseSampler:
    def __init__(
        self,
        x: Tensor,
        sigma_min: float | None = None,
        sigma_max: float | None = None,
        seed: int | None = None,
        cpu: bool = False,
        transform: Callable = lambda t: t,
        make_noise_sampler: Callable | None = None,
        normalize_noise=False,
        factor: float = 1.0,
    ):
        try:
            self.noise_sampler = make_noise_sampler(
                x,
                transform(torch.as_tensor(sigma_min))
                if sigma_min is not None
                else None,
                transform(torch.as_tensor(sigma_max))
                if sigma_max is not None
                else None,
                seed=seed,
                cpu=cpu,
            )
        except TypeError:
            self.noise_sampler = make_noise_sampler(x)
        self.factor = factor
        self.normalize_noise = normalize_noise
        self.transform = transform
        self.device = x.device
        self.dtype = x.dtype

    @classmethod
    def simple(cls, f):
        return lambda *args, **kwargs: cls(
            *args,
            **kwargs,
            make_noise_sampler=lambda x, *_args, **_kwargs: lambda _s, _sn: f(x),
        )

    @classmethod
    def wrap(cls, f):
        return lambda *args, **kwargs: cls(*args, **kwargs, make_noise_sampler=f)

    def __call__(self, *args, **kwargs):
        args = (
            self.transform(torch.as_tensor(s)) if s is not None else s for s in args
        )
        noise = self.noise_sampler(*args, **kwargs)
        noise = (
            scale_noise(noise, self.factor)
            if self.normalize_noise
            else noise.mul_(self.factor)
        )
        if hasattr(noise, "to"):
            return noise.to(dtype=self.dtype, device=self.device)
        return noise


NOISE_SAMPLERS: dict[NoiseType, Callable] = {
    NoiseType.BROWNIAN: NoiseSampler.wrap(sampling.BrownianTreeNoiseSampler),
    NoiseType.GAUSSIAN: NoiseSampler.simple(torch.randn_like),
    NoiseType.UNIFORM: NoiseSampler.simple(uniform_noise_like),
    NoiseType.PERLIN: NoiseSampler.simple(rand_perlin_like),
    NoiseType.STUDENTT: NoiseSampler.simple(studentt_noise_like),
    NoiseType.PINK: NoiseSampler.simple(pink_noise_like),
    NoiseType.HIGHRES_PYRAMID: NoiseSampler.simple(highres_pyramid_noise_like),
    NoiseType.RAINBOW_MILD: NoiseSampler.simple(
        lambda x: (green_noise_like(x) * 0.55 + rand_perlin_like(x) * 0.7) * 1.15,
    ),
    NoiseType.RAINBOW_INTENSE: NoiseSampler.simple(
        lambda x: (green_noise_like(x) * 0.75 + rand_perlin_like(x) * 0.5) * 1.15,
    ),
    NoiseType.LAPLACIAN: NoiseSampler.simple(laplacian_noise_like),
    NoiseType.POWER: NoiseSampler.simple(power_noise_like),
    NoiseType.GREEN_TEST: NoiseSampler.simple(green_noise_like),
    # NoiseType.RAINBOW_MILD2: lambda x: lambda _s, _sn: (
    #     green_noise_like(x) * 0.55 + uniform_noise_like(x) * 0.7
    # )
    # * 1.15,
    # NoiseType.RAINBOW_INTENSE2: lambda x: lambda _s, _sn: (
    #     green_noise_like(x) * 0.75 + uniform_noise_like(x) * 0.5
    # )
    # * 1.15,
    # NoiseType.RAINBOW_INTENSE3: lambda x: lambda _s, _sn: (
    #     green_noise_like(x) * 0.75 + highres_pyramid_noise_like(x) * 0.5
    # )
    # * 1.15,
}


def get_noise_sampler(
    noise_type: str | NoiseType | None,
    x: Tensor,
    sigma_min: float | None,
    sigma_max: float | None,
    seed: int | None = None,
    cpu: bool = True,
    factor: float = 1.0,
    normalize_noise=True,
) -> Callable:
    if noise_type is None:
        noise_type = NoiseType.GAUSSIAN
    elif isinstance(noise_type, str):
        noise_type = NoiseType[noise_type.upper()]
    if noise_type == NoiseType.BROWNIAN and (sigma_min is None or sigma_max is None):
        raise ValueError("Must pass sigma min/max when using brownian noise")
    mkns = NOISE_SAMPLERS.get(noise_type)
    if mkns is None:
        raise ValueError("Unknown noise sampler")
    return mkns(
        x,
        sigma_min,
        sigma_max,
        seed=seed,
        cpu=cpu,
        factor=factor,
        normalize_noise=normalize_noise,
    )
