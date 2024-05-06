# Noise generation functions shamelessly yoinked from https://github.com/Clybius/ComfyUI-Extra-Samplers
from __future__ import annotations

import abc
import functools as fun
import math
import operator as op
from enum import Enum, auto
from typing import Callable

import torch
from comfy.k_diffusion import sampling
from torch import FloatTensor, Generator, Tensor
from torch.distributions import StudentT

# ruff: noqa: D412, D413, D417, D212, D407, ANN002, ANN003, FBT001, FBT002, S311


def scale_noise(noise, factor=1.0, threshold_std_devs=2.5):
    mean, std = noise.mean().item(), noise.std().item()
    threshold = threshold_std_devs / math.sqrt(noise.numel())
    if abs(mean) > threshold:
        noise -= mean
    if abs(1.0 - std) > threshold:
        noise /= std
    if factor != 1.0:
        noise *= factor
    return noise


class NoiseType(Enum):
    GAUSSIAN = auto()
    UNIFORM = auto()
    BROWNIAN = auto()
    PERLIN = auto()
    STUDENTT = auto()
    HIGHRES_PYRAMID = auto()
    PYRAMID = auto()
    PINK = auto()
    LAPLACIAN = auto()
    POWER = auto()
    RAINBOW_MILD = auto()
    # RAINBOW_MILD2 = auto()
    RAINBOW_INTENSE = auto()
    # RAINBOW_INTENSE2 = auto()
    # RAINBOW_INTENSE3 = auto()
    GREEN_TEST = auto()

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


class CustomNoiseItemBase(abc.ABC):
    def __init__(self, factor, **kwargs):
        self.factor = factor
        self.keys = set(kwargs.keys())
        for k, v in kwargs.items():
            setattr(self, k, v)

    def clone(self):
        return self.__class__(self.factor, **{k: getattr(self, k) for k in self.keys})

    def set_factor(self, factor):
        self.factor = factor
        return self

    @abc.abstractmethod
    def make_noise_sampler(
        self,
        x: Tensor,
        sigma_min=None,
        sigma_max=None,
        seed=None,
        cpu=True,
    ):
        raise NotImplementedError


class CustomNoiseItem(CustomNoiseItemBase):
    def __init__(self, factor, **kwargs):
        super().__init__(factor, **kwargs)
        if getattr(self, "noise_type", None) is None:
            raise ValueError("Noise type required!")

    @torch.no_grad()
    def make_noise_sampler(
        self,
        x: Tensor,
        sigma_min=None,
        sigma_max=None,
        seed=None,
        cpu=True,
    ):
        return get_noise_sampler(
            self.noise_type,
            x,
            sigma_min,
            sigma_max,
            seed=seed,
            cpu=cpu,
            factor=self.factor,
        )


class CustomNoiseChain:
    def __init__(self, items=None):
        self.items = items if items is not None else []

    def clone(self):
        return CustomNoiseChain(
            [i.clone() for i in self.items],
        )

    def add(self, item):
        self.items.append(item)

    def rescaled(self, scale=1.0):
        total = sum(i.factor for i in self.items)
        divisor = total / scale
        divisor = divisor if divisor != 0 else 1.0
        return CustomNoiseChain(
            [i.clone().set_factor(i.factor / divisor) for i in self.items],
        )

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
            i.make_noise_sampler(
                x,
                sigma_min,
                sigma_max,
                seed=seed,
                cpu=cpu,
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


def pyramid_noise_like(x, generator=None, device="cpu", discount=0.8):
    size = x.size()
    b, c, h, w = size
    orig_h = h
    orig_w = w
    noise = torch.zeros(size=size, dtype=x.dtype, layout=x.layout, device=device)
    r = 1
    for i in range(5):
        r *= 2  # Rather than always going 2x,
        noise += (
            torch.nn.functional.interpolate(
                (
                    torch.normal(
                        mean=0,
                        std=0.5**i,
                        size=(b, c, h * r, w * r),
                        dtype=x.dtype,
                        layout=x.layout,
                        generator=generator,
                        device=device,
                    )
                ),
                size=(orig_h, orig_w),
                mode="nearest-exact",
            )
            * discount**i
        )
    return noise.to(device=x.device)


def studentt_noise_like(x):
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
            noise = noise.to(dtype=self.dtype, device=self.device)
        return noise


class RepeatedNoise:
    def __init__(self, noise_sampler, repeat_length, permute=True):
        self.noise_sampler = noise_sampler
        self.repeat_length = repeat_length
        self.permute = permute

    def clone(self):
        return RepeatedNoise(self.noise_sampler, self.repeat_length)

    def make_noise_sampler(self, x, *args, seed=None, **kwargs):
        ns = self.noise_sampler(x, *args, seed=seed, **kwargs)
        noise_items = []
        permute_options = 2
        u32_max = 0xFFFF_FFFF
        if seed is None:
            seed = torch.randint(
                -u32_max,
                u32_max,
                (1,),
                device="cpu",
                dtype=torch.int64,
            ).item()
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)

        def noise_sampler(s, sn):
            rands = torch.randint(
                u32_max,
                (4,),
                generator=gen,
                dtype=torch.uint32,
            ).tolist()
            if len(noise_items) < self.repeat_length:
                idx = len(noise_items)
                noise_items.append(ns(s, sn))
            else:
                idx = rands[0] % self.repeat_length
            noise = noise_items[idx]
            if not self.permute:
                return noise.clone()
            noise_dims = len(noise.shape)
            match rands[1] % permute_options:
                case 0:
                    if rands[2] <= u32_max // 10:
                        # 10% of the time we return the original tensor instead of flipping
                        noise = noise.clone()
                    else:
                        dim = -1 + (rands[2] % (noise_dims + 1))
                        noise = torch.flip(noise, (dim,))
                case 1:
                    dim = rands[2] % noise_dims
                    count = rands[3] % noise.shape[dim]
                    noise = torch.roll(noise, count, dims=(dim,)).clone()
            return noise

        return noise_sampler


# Modulated noise functions copied from https://github.com/Clybius/ComfyUI-Extra-Samplers
# They probably don't work correctly for normal sampling.
class ModulatedNoise:
    MODULATION_DIMS = (-3, (-2, -1), (-3, -2, -1))

    def __init__(
        self,
        noise_sampler,
        modulation_type="none",
        modulation_strength=2.0,
        modulation_dims=3,
    ):
        self.noise_sampler = noise_sampler
        self.dims = self.MODULATION_DIMS[modulation_dims - 1]
        self.type = modulation_type
        self.strength = modulation_strength
        match self.type:
            case "intensity":
                self.modulation_function = self.intensity_based_multiplicative_noise
            case "frequency":
                self.modulation_function = self.frequency_based_noise
            case "spectral_signum":
                self.modulation_function = self.spectral_modulate_noise
            case _:
                self.modulation_function = None

    def clone(self):
        return ModulatedNoise(self.noise_sampler, self.type, self.strength, self.dims)

    def make_noise_sampler(self, x, *args, **kwargs):
        ns = self.noise_sampler(x, *args, **kwargs)
        if not self.modulation_function:
            return ns
        s_noise = sigma_up = 1.0
        return lambda s, sn: self.modulation_function(
            x,
            ns(s, sn),
            s_noise,
            sigma_up,
            self.strength,
            self.dims,
        )

    @staticmethod
    def intensity_based_multiplicative_noise(
        x,
        noise,
        s_noise,
        sigma_up,
        intensity,
        dims,
    ) -> torch.Tensor:
        """Scales noise based on the intensities of the input tensor."""
        std = torch.std(
            x - x.mean(),
            dim=dims,
            keepdim=True,
        )  # Average across channels to get intensity
        scaling = (
            1 / (std * abs(intensity) + 1.0)
        )  # Scale std by intensity, as not doing this leads to more noise being left over, leading to crusty/preceivably extremely oversharpened images
        additive_noise = noise * s_noise * sigma_up
        scaled_noise = noise * s_noise * sigma_up * scaling + additive_noise

        noise_norm = torch.norm(additive_noise)
        scaled_noise_norm = torch.norm(scaled_noise)
        scaled_noise *= noise_norm / scaled_noise_norm  # Scale to normal noise strength
        return scaled_noise * intensity + additive_noise * (1 - intensity)

    @staticmethod
    def frequency_based_noise(
        z_k,
        noise,
        s_noise,
        sigma_up,
        intensity,
        channels,
    ) -> torch.Tensor:
        """Scales the high-frequency components of the noise based on the given intensity."""
        additive_noise = noise * s_noise * sigma_up

        std = torch.std(
            z_k - z_k.mean(),
            dim=channels,
            keepdim=True,
        )  # Average across channels to get intensity
        scaling = 1 / (std * abs(intensity) + 1.0)
        # Perform Fast Fourier Transform (FFT)
        z_k_freq = torch.fft.fft2(scaling * additive_noise + additive_noise)

        # Get the magnitudes of the frequency components
        magnitudes = torch.abs(z_k_freq)

        # Create a high-pass filter (emphasize high frequencies)
        h, w = z_k.shape[-2:]
        b = abs(
            intensity,
        )  # Controls the emphasis of the high pass (higher frequencies are boosted)
        high_pass_filter = 1 - torch.exp(
            -((torch.arange(h)[:, None] / h) ** 2 + (torch.arange(w)[None, :] / w) ** 2)
            * b**2,
        )
        high_pass_filter = high_pass_filter.to(z_k.device)

        # Apply the filter to the magnitudes
        magnitudes_scaled = magnitudes * (1 + high_pass_filter)

        # Reconstruct the complex tensor with scaled magnitudes
        z_k_freq_scaled = magnitudes_scaled * torch.exp(1j * torch.angle(z_k_freq))

        # Perform Inverse Fast Fourier Transform (IFFT)
        z_k_scaled = torch.fft.ifft2(z_k_freq_scaled)

        # Return the real part of the result
        z_k_scaled = torch.real(z_k_scaled)

        noise_norm = torch.norm(additive_noise)
        scaled_noise_norm = torch.norm(z_k_scaled)

        z_k_scaled *= noise_norm / scaled_noise_norm  # Scale to normal noise strength

        return z_k_scaled * intensity + additive_noise * (1 - intensity)

    @staticmethod
    def spectral_modulate_noise(
        _unused,
        noise,
        s_noise,
        sigma_up,
        intensity,
        channels,
        spectral_mod_percentile=5.0,
    ) -> torch.Tensor:  # Modified for soft quantile adjustment using a novel:tm::c::r: method titled linalg.
        additive_noise = noise * s_noise * sigma_up
        # Convert image to Fourier domain
        fourier = torch.fft.fftn(
            additive_noise,
            dim=channels,
        )  # Apply FFT along Height and Width dimensions

        log_amp = torch.log(torch.sqrt(fourier.real**2 + fourier.imag**2))

        quantile_low = (
            torch.quantile(
                log_amp.abs().flatten(1),
                spectral_mod_percentile * 0.01,
                dim=1,
            )
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(log_amp.shape)
        )

        quantile_high = (
            torch.quantile(
                log_amp.abs().flatten(1),
                1 - (spectral_mod_percentile * 0.01),
                dim=1,
            )
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(log_amp.shape)
        )

        quantile_max = (
            torch.quantile(log_amp.abs().flatten(1), 1, dim=1)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(log_amp.shape)
        )

        # Decrease high-frequency components
        mask_high = log_amp > quantile_high  # If we're larger than 95th percentile

        additive_mult_high = torch.where(
            mask_high,
            1
            - ((log_amp - quantile_high) / (quantile_max - quantile_high)).clamp_(
                max=0.5,
            ),  # (1) - (0-1), where 0 is 95th %ile and 1 is 100%ile
            torch.tensor(1.0),
        )

        # Increase low-frequency components
        mask_low = log_amp < quantile_low
        additive_mult_low = torch.where(
            mask_low,
            1
            + (1 - (log_amp / quantile_low)).clamp_(
                max=0.5,
            ),  # (1) + (0-1), where 0 is 5th %ile and 1 is 0%ile
            torch.tensor(1.0),
        )

        mask_mult = (additive_mult_low * additive_mult_high) ** intensity
        # print(mask_mult)
        filtered_fourier = fourier * mask_mult

        # Inverse transform back to spatial domain
        inverse_transformed = torch.fft.ifftn(
            filtered_fourier,
            dim=channels,
        )  # Apply IFFT along Height and Width dimensions

        return inverse_transformed.real.to(additive_noise.device)


NOISE_SAMPLERS: dict[NoiseType, Callable] = {
    NoiseType.BROWNIAN: NoiseSampler.wrap(sampling.BrownianTreeNoiseSampler),
    NoiseType.GAUSSIAN: NoiseSampler.simple(torch.randn_like),
    NoiseType.UNIFORM: NoiseSampler.simple(uniform_noise_like),
    NoiseType.PERLIN: NoiseSampler.simple(rand_perlin_like),
    NoiseType.STUDENTT: NoiseSampler.simple(studentt_noise_like),
    NoiseType.PINK: NoiseSampler.simple(pink_noise_like),
    NoiseType.HIGHRES_PYRAMID: NoiseSampler.simple(highres_pyramid_noise_like),
    NoiseType.PYRAMID: NoiseSampler.simple(pyramid_noise_like),
    NoiseType.RAINBOW_MILD: NoiseSampler.simple(
        lambda x: (green_noise_like(x) * 0.55 + rand_perlin_like(x) * 0.7) * 1.15,
    ),
    NoiseType.RAINBOW_INTENSE: NoiseSampler.simple(
        lambda x: (green_noise_like(x) * 0.75 + rand_perlin_like(x) * 0.5) * 1.15,
    ),
    NoiseType.LAPLACIAN: NoiseSampler.simple(laplacian_noise_like),
    NoiseType.POWER: NoiseSampler.simple(power_noise_like),
    NoiseType.GREEN_TEST: NoiseSampler.simple(green_noise_like),
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
