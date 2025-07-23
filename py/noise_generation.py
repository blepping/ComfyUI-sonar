# Noise generation functions shamelessly yoinked from https://github.com/Clybius/ComfyUI-Extra-Samplers
from __future__ import annotations

import math
from enum import Enum, auto
from typing import TYPE_CHECKING, Callable, ClassVar, NamedTuple

import torch
from comfy.k_diffusion import sampling
from comfy.model_management import throw_exception_if_processing_interrupted
from torch import FloatTensor, Generator, Tensor
from torch.distributions import Laplace, StudentT

from . import utils
from .utils import (
    fallback,
    normalize_to_scale,
    quantile_normalize,
    scale_noise,
    tensor_to,
)
from .wavelet_functions import Wavelet, ptwav, wavelet_blend, wavelet_scaling

if TYPE_CHECKING:
    from collections.abc import Sequence

# ruff: noqa: D413, D417, D212, ANN002, ANN003


class NoiseType(Enum):
    BROWNIAN = auto()
    COLLATZ = auto()
    DISTRO = auto()
    GAUSSIAN = auto()
    GREEN_TEST = auto()
    GREY = auto()
    HIGHRES_PYRAMID = auto()
    HIGHRES_PYRAMID_AREA = auto()
    HIGHRES_PYRAMID_BISLERP = auto()
    LAPLACIAN = auto()
    ONEF_GREENISH = auto()
    ONEF_GREENISH_MIX = auto()
    ONEF_PINKISH = auto()
    ONEF_PINKISH_MIX = auto()
    ONEF_PINKISHGREENISH = auto()
    PERLIN = auto()
    PINK_OLD = auto()
    POWER_OLD = auto()
    PYRAMID = auto()
    PYRAMID_AREA = auto()
    PYRAMID_BISLERP = auto()
    PYRAMID_DISCOUNT5 = auto()
    PYRAMID_MIX = auto()
    PYRAMID_MIX_AREA = auto()
    PYRAMID_MIX_BISLERP = auto()
    PYRAMID_OLD = auto()
    PYRAMID_OLD_AREA = auto()
    PYRAMID_OLD_BISLERP = auto()
    RAINBOW_INTENSE = auto()
    RAINBOW_MILD = auto()
    STUDENTT = auto()
    UNIFORM = auto()
    VELVET = auto()
    VIOLET = auto()
    WAVELET = auto()
    WHITE = auto()

    @classmethod
    def get_names(cls, default=GAUSSIAN, skip=None):
        if default is not None:
            if isinstance(default, int):
                default = cls(default)
            yield default.name.lower()
        for nt in cls:
            if nt == default or (skip and nt in skip):
                continue
            yield nt.name.lower()


class NoiseError(Exception):
    pass


class NoiseGenerator:
    name = "unknown"
    MIN_DIMS = 1
    MAX_DIMS = 0

    def __init__(
        self,
        x,
        **kwargs,
    ):
        if x.ndim < self.MIN_DIMS:
            errstr = f"Noise generator {self.name} requires at least {self.MIN_DIMS} dimension(s) but got input with shape {x.shape}"
            raise ValueError(errstr)
        if self.MAX_DIMS > 0 and x.ndim > self.MAX_DIMS:
            errstr = f"Noise generator {self.name} requires at most {self.MAX_DIMS} dimension(s) but got input with shape {x.shape}"
            raise ValueError(errstr)
        params = self.ng_params()
        kwarg_params = params | kwargs
        for k in params:
            setattr(self, k, kwarg_params.pop(k))
        self.options = kwarg_params
        self.update_x(x)

    @classmethod
    def ng_params(cls):
        return {
            "normalized": True,
            "force_normalize": None,
            "normalize_dims": None,
            "cpu": True,
            "generator": None,
        }

    def update_x(self, x):
        self.shape = x.shape
        if x.ndim in {4, 5}:
            self.batch, self.channels = x.shape[:2]
            self.height, self.width = x.shape[-2:]
            self.frames = x.shape[-3] if x.ndim == 5 else None
        else:
            self.batch = self.channels = self.frames = self.height = self.width = None
        self.device = x.device
        self.gen_device = torch.device("cpu") if self.cpu else self.device
        self.layout = x.layout
        self.dtype = x.dtype

    def rand_like(
        self,
        *,
        fun=torch.randn,
        cpu=None,
        to_device=True,
        shape=None,
        dtype=None,
        layout=None,
        device=None,
        generator=None,
    ):
        cpu = fallback(cpu, self.cpu)
        noise = fun(
            *fallback(shape, self.shape),
            generator=fallback(generator, self.generator),
            dtype=fallback(dtype, self.dtype),
            layout=fallback(layout, self.layout),
            device=fallback(device, "cpu" if cpu else self.gen_device),
        )
        if to_device and noise.device != self.device:
            noise = tensor_to(noise, self.device)
        return noise

    def output_hook(self, noise):
        if noise.device != self.device:
            noise = tensor_to(noise, self.device)
        return scale_noise(
            noise,
            normalized=self.normalized
            and (self.force_normalize is None or self.force_normalize is True),
            normalize_dims=self.normalize_dims,
        )

    def pre_hook(self):
        pass

    def generate(self):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        self.pre_hook()
        return self.output_hook(self.generate(*args, **kwargs))

    def __str__(self):
        pretty_params = ", ".join(f"{k}={getattr(self, k)!s}" for k in self.ng_params())
        return f"<NoiseGenerator({self.name}): device={self.device}, shape={self.shape}, dtype={self.dtype}, {pretty_params}>"


class FramesToChannelsNoiseGenerator(NoiseGenerator):
    MIN_DIMS = 4
    MAX_DIMS = 5

    def get_adjusted_shape(self):
        if self.frames:
            return (self.batch, self.channels * self.frames, self.height, self.width)
        return (self.batch, self.channels, self.height, self.width)

    def fix_output_frames(self, noise):
        if not self.frames:
            return noise
        return noise.reshape(
            self.batch,
            self.channels,
            self.frames,
            self.height,
            self.width,
        )

    def rand_like(self, *args, shape=None, **kwargs):
        noise = super().rand_like(*args, shape=shape, **kwargs)
        if shape is not None:
            return noise
        adjusted_shape = self.get_adjusted_shape()
        if noise.shape != adjusted_shape:
            return noise.reshape(*adjusted_shape)
        return noise


class MixedNoiseGenerator(NoiseGenerator):
    @classmethod
    def ng_params(cls):
        return super().ng_params() | {
            "name": "mixed_noise",
            "normalized": True,
            "pass_args": frozenset(("cpu",)),
            "noise_mix": (),
            "output_fun": None,
        }

    def __init__(self, x, *args, **kwargs):
        min_dim = max_dim = None
        self.name = kwargs["name"]
        for item in kwargs["noise_mix"]:
            ng_class = item[0] if isinstance(item, (tuple, list)) else item
            cmin, cmax = ng_class.MIN_DIMS, ng_class.MAX_DIMS
            min_dim = max(min_dim if min_dim is not None else cmin, cmin)
            max_dim = min(max_dim if max_dim is not None else cmax, cmax)
        self.MIN_DIMS = min_dim
        self.MAX_DIMS = max_dim
        super().__init__(x, *args, **kwargs)
        ng_list = []
        for ng_class, ng_class_kwargs, transform_fun in self.noise_mix:
            ng_kwargs = {k: v for k, v in kwargs.items() if k in self.pass_args}
            ng_list.append((ng_class(x, **ng_class_kwargs, **ng_kwargs), transform_fun))
        self.ng_list = ng_list

    def generate(self, *args):
        noise = None
        for ng, transform_fun in self.ng_list:
            new_noise = ng(*args)
            if transform_fun is not None:
                new_noise = transform_fun(new_noise)
            noise = new_noise if noise is None else noise.add_(new_noise)
        if self.output_fun is not None:
            noise = self.output_fun(noise)
        return noise


class GaussianNoiseGenerator(NoiseGenerator):
    name = "gaussian"

    @classmethod
    def ng_params(cls):
        return super().ng_params() | {"normalized": False}

    def generate(self, *_args):
        return self.rand_like()


class BrownianNoiseGenerator(NoiseGenerator):
    name = "brownian"

    def __init__(self, x, *args, **kwargs):
        super().__init__(x, *args, **kwargs)
        seed = self.options.get("seed")
        sigma_min = self.options.get("sigma_min")
        sigma_max = self.options.get("sigma_max")
        if sigma_min is None or sigma_max is None:
            raise ValueError("Brownian noise requires sigma_min and sigma_max")
        self.brownian_tree_ns = sampling.BrownianTreeNoiseSampler(
            x,
            sigma_min,
            sigma_max,
            seed=seed,
            cpu=self.cpu,
        )

    @classmethod
    def ng_params(cls):
        return super().ng_params() | {"normalized": False}

    def generate(self, *args):
        return self.brownian_tree_ns(*args)


class PerlinOldNoiseGenerator(FramesToChannelsNoiseGenerator):
    name = "perlin_old"

    @classmethod
    def ng_params(cls):
        return super().ng_params() | {
            "div_fac": 2.0,
            "iterations": 2,
            "blend_mode": "lerp",
        }

    @staticmethod
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

    @staticmethod
    def unfold_grid(vectors: Tensor) -> Tensor:
        """
        Unfold vector grid to batched vectors.

        Arguments:
            vectors -- grid vectors

        Returns:
            batched grid vectors
        """
        batch_size, _channels, gpy, gpx = vectors.shape
        return (
            torch.nn.functional.unfold(vectors, (2, 2))
            .view(batch_size, 2, 4, -1)
            .permute(0, 2, 3, 1)
            .view(batch_size, 4, gpy - 1, gpx - 1, 2)
        )

    @staticmethod
    def smooth_step(t: Tensor) -> Tensor:
        """
        Smooth step function [0, 1] -> [0, 1].

        Arguments:
            t -- input values (any shape)

        Returns:
            output values (same shape as input values)
        """
        return t * t * (3.0 - 2.0 * t)

    @classmethod
    def perlin_noise_tensor(
        cls,
        vectors: Tensor,
        positions: Tensor,
        step: Callable | None = None,
        blend=torch.lerp,
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
            step = cls.smooth_step

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

        row0 = blend(
            (vectors[:, 0] * positions).sum(dim=-1),
            (vectors[:, 1] * (positions - positions.new_tensor((1, 0)))).sum(dim=-1),
            step_x,
        )
        row1 = blend(
            (vectors[:, 2] * (positions - positions.new_tensor((0, 1)))).sum(dim=-1),
            (vectors[:, 3] * (positions - positions.new_tensor((1, 1)))).sum(dim=-1),
            step_x,
        )
        noise = blend(row0, row1, step_y)
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

    @classmethod
    def perlin_noise(
        cls,
        grid_shape: tuple[int, int],
        out_shape: tuple[int, int],
        batch_size: int = 1,
        blend=torch.lerp,
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
        vectors = cls.unfold_grid(
            torch.stack((torch.cos(angle), torch.sin(angle)), dim=1),
        )
        # positions inside grid cells [0, 1)
        positions = tensor_to(cls.get_positions((bh, bw)), vectors)
        return cls.perlin_noise_tensor(vectors, positions, blend=blend).squeeze(0)

    def generate(self, *_args):
        blend = utils.BLENDING_MODES[self.blend_mode]
        noise = self.rand_like(fun=torch.rand).div_(self.div_fac)

        channels, height, width = noise.shape[1:]
        for _ in range(self.iterations):
            noise += self.perlin_noise(
                (height, self.width),
                (height, width),
                batch_size=channels,
                blend=blend,
                dtype=noise.dtype,
                layout=noise.layout,
                device=noise.device,
            )
        return self.fix_output_frames(noise)


class UniformNoiseGenerator(NoiseGenerator):
    name = "uniform"

    @classmethod
    def ng_params(cls):
        return super().ng_params() | {
            "normalized": False,
            "sub_fac": 0.5,
            "mul_fac": 3.46,
            "mean_fac": 0.0,
        }

    def generate(self, *_args):
        return (
            self.rand_like(fun=torch.rand)
            .sub_(self.sub_fac)
            .mul_(self.mul_fac)
            .add_(self.mean_fac)
        )


class HighresPyramidNoiseGenerator(FramesToChannelsNoiseGenerator):
    name = "highres_pyramid"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.noise_generator is None:
            self.noise_generator = UniformNoiseGenerator(
                *args,
                **(kwargs | {"normalized": self.normalize_noise}),
            )

    @classmethod
    def ng_params(cls):
        return super().ng_params() | {
            "normalized": True,
            "discount": 0.7,
            "upscale_mode": "bilinear",
            "iterations": 4,
            "noise_generator": None,
            "normalize_noise": False,
        }

    def generate(self, s, sn):
        adjusted_shape = self.get_adjusted_shape()
        b, c, h, w = adjusted_shape
        orig_w, orig_h = w, h
        noise = self.noise_generator(s, sn).reshape(*adjusted_shape)
        rs = (
            torch.rand(
                self.iterations,
                dtype=torch.float32,
                generator=self.generator,
            ).cpu()
            * 2
            + 2
        )
        for i in range(self.iterations):
            r = rs[i].item()
            h, w = min(orig_h * 15, int(h * (r**i))), min(orig_w * 15, int(w * (r**i)))
            noise += utils.scale_samples(
                tensor_to(torch.randn(b, c, h, w, generator=self.generator), noise),
                orig_w,
                orig_h,
                mode=self.upscale_mode,
            ).mul_(self.discount**i)
            if h >= orig_h * 15 or w >= orig_w * 15:
                break  # Lowest resolution is 1x1
        return self.fix_output_frames(noise)


class PyramidOldNoiseGenerator(FramesToChannelsNoiseGenerator):
    name = "pyramid_old"

    @classmethod
    def ng_params(cls):
        return super().ng_params() | {
            "discount": 0.8,
            "iterations": 5,
            "upscale_mode": "nearest-exact",
            "normalized": False,
        }

    def generate(self, *_args):
        adjusted_shape = self.get_adjusted_shape()
        b, c, h, w = adjusted_shape
        orig_h, orig_w = h, w
        noise = torch.zeros(
            size=adjusted_shape,
            dtype=self.dtype,
            layout=self.layout,
            device=self.gen_device,
        )
        r = 1
        for i in range(self.iterations):
            r *= 2
            noise += utils.scale_samples(
                torch.normal(
                    mean=0,
                    std=0.5**i,
                    size=(b, c, h * r, w * r),
                    dtype=noise.dtype,
                    layout=noise.layout,
                    generator=self.generator,
                    device=noise.device,
                ),
                orig_w,
                orig_h,
                mode=self.upscale_mode,
            ).mul_(self.discount**i)
        return self.fix_output_frames(noise)


class PyramidNoiseGenerator(FramesToChannelsNoiseGenerator):
    name = "pyramid"

    @classmethod
    def ng_params(cls):
        return super().ng_params() | {
            "discount": 0.7,
            "upscale_mode": "bilinear",
            "iterations": 10,
        }

    # Modified from https://wandb.ai/johnowhitaker/multires_noise/reports/Multi-Resolution-Noise-for-Diffusion-Model-Training--VmlldzozNjYyOTU2
    def generate(self, *_args):
        noise = self.rand_like()
        b, c, h, w = noise.shape
        orig_w, orig_h = w, h

        for i in range(self.iterations):
            r = (
                torch.rand(1, generator=self.generator).cpu().item() * 2 + 2
            )  # Rather than always going 2x,
            w, h = max(1, int(w / (r**i))), max(1, int(h / (r**i)))
            noise += utils.scale_samples(
                torch.randn(
                    b,
                    c,
                    h,
                    w,
                    device=noise.device,
                    layout=noise.layout,
                    dtype=noise.dtype,
                ),
                orig_w,
                orig_h,
                mode=self.upscale_mode,
            ).mul_(
                self.discount**i,
            )
            if w == 1 or h == 1:
                break  # Lowest resolution is 1x1
        return self.fix_output_frames(noise)


class StudentTNoiseGenerator(NoiseGenerator):
    name = "studentt"

    @classmethod
    def ng_params(cls):
        return super().ng_params() | {
            "loc": 0,
            "scale": 0.2,
            "df": 1,
            "quantile_fac": 0.75,
            "pow_fac": 0.5,
            "nq_fac": 1.0,
            "normalized": False,
        }

    def generate(self, *_args):
        noise = StudentT(loc=self.loc, scale=self.scale, df=self.df).rsample(self.shape)
        nq: FloatTensor = torch.quantile(
            noise.flatten(start_dim=1).abs(),
            self.quantile_fac,
            dim=-1,
        )
        nq_shape = tuple(nq.shape) + (1,) * (noise.ndim - nq.ndim)
        nq = nq.mul_(self.nq_fac).reshape(*nq_shape)
        noise = noise.clamp(-nq, nq)
        return torch.copysign(torch.pow(torch.abs(noise), self.pow_fac), noise)


class GreenTestNoiseGenerator(FramesToChannelsNoiseGenerator):
    name = "green_test"
    MIN_DIMS = 4
    MAX_DIMS = 5

    @classmethod
    def ng_params(cls):
        return super().ng_params() | {
            "scale_fac": 1.0,
            "x_pow": 2,
            "y_pow": 2,
            "power_base": 1,
        }

    def generate(self, *_args):
        noise = self.rand_like()
        scale = self.scale_fac / (self.width * self.height)
        fy = torch.fft.fftfreq(self.height, device=noise.device)[:, None] ** self.y_pow
        fx = torch.fft.fftfreq(self.width, device=noise.device) ** self.x_pow
        f = fy + fx
        power = torch.sqrt(f)
        power[0, 0] = self.power_base
        noise = torch.fft.ifft2(torch.fft.fft2(noise) / torch.sqrt(power))
        noise *= scale / noise.std()
        return self.fix_output_frames(torch.real(noise))


class PinkOldNoiseGenerator(NoiseGenerator):
    name = "pink_old"

    @classmethod
    def ng_params(cls):
        return super().ng_params() | {"alpha": 2.0, "k": 1.0, "freq": 1.0}

    # Completely wrong implementation here.
    def generate(self, *_args):
        spectral_density = self.k / self.freq**self.alpha
        return self.rand_like() * spectral_density


class OneFNoiseGenerator(FramesToChannelsNoiseGenerator):
    name = "onef"
    MIN_DIMS = 4
    MAX_DIMS = 5

    @classmethod
    def ng_params(cls):
        return super().ng_params() | {
            "alpha": 2.0,
            "k": 1.0,
            "hfac": 1.0,
            "wfac": 1.0,
            "base_power": 1.0,
            "use_sqrt": True,
        }

    # Referenced from: https://github.com/WASasquatch/PowerNoiseSuite
    def generate(self, *_args):
        # batch, _channels, height, width = self.shape

        noise = self.rand_like()

        freq_x = tensor_to(torch.fft.fftfreq(self.height, self.hfac), noise)
        freq_y = tensor_to(torch.fft.fftfreq(self.width, self.wfac), noise)
        fx, fy = torch.meshgrid(freq_x, freq_y, indexing="ij")

        power = (fx**2 + fy**2) ** (-self.alpha / 2.0)
        if self.k != 0:
            power = self.k / power
        power[0, 0] = self.base_power
        power = power.unsqueeze(0).expand(self.batch, 1, self.height, self.width)

        noise_fft = torch.fft.fftn(noise)
        noise_fft /= (
            torch.sqrt(power.to(noise_fft.dtype))
            if self.use_sqrt
            else power.to(noise_fft.dtype)
        )

        return self.fix_output_frames(torch.fft.ifftn(noise_fft).real)


class PowerLawNoiseGenerator(NoiseGenerator):
    name = "powerlaw"

    @classmethod
    def ng_params(cls):
        return super().ng_params() | {
            "alpha": 2.0,
            "div_max_dims": None,
            "use_sign": False,
            "use_div_max_abs": True,
        }

    # Referenced from: https://github.com/WASasquatch/PowerNoiseSuite
    def generate(self, *_args):
        noise = self.rand_like()

        modulation = torch.abs(noise) ** self.alpha
        noise = (torch.sign(noise) if self.use_sign else noise).mul_(modulation)
        if self.div_max_dims is not None:
            noise /= torch.amax(
                torch.abs(noise) if self.use_div_max_abs else noise,
                keepdim=True,
                dim=self.div_max_dims,
            )
        return noise


class LaplacianNoiseGenerator(NoiseGenerator):
    name = "laplacian"

    @classmethod
    def ng_params(cls):
        return super().ng_params() | {"loc": 0, "scale": 1.0, "div_fac": 4.0}

    def generate(self, *_args):
        noise = self.rand_like().div_(self.div_fac)
        noise += tensor_to(
            Laplace(loc=self.loc, scale=self.scale).rsample(self.shape),
            noise.device,
        )
        return noise


class DistroNoiseGenerator(NoiseGenerator):
    name = "distro"

    simple_distros = frozenset((
        "cauchy",
        "exponential",
        "geometric",
        "log_normal",
        "normal",
    ))

    def __init__(self, x, *args, **kwargs):
        super().__init__(x, *args, **kwargs)
        if self.distro not in self.distro_params():
            raise ValueError("Bad distro")

    _distro_params = None

    @classmethod
    def distro_params(cls):
        if cls._distro_params is not None:
            return cls._distro_params
        td = torch.distributions
        tt = torch.Tensor
        cls._distro_params = {
            # Simple
            "exponential": (
                tt.exponential_,
                {
                    "lambd": {
                        "default": 1.0,
                    },
                },
            ),
            "cauchy": (
                tt.cauchy_,
                {
                    "median": {
                        "default": "0.0",
                    },
                    "sigma": {
                        "default": 1.0,
                        "min": 0.0,
                    },
                },
            ),
            "geometric": (
                tt.geometric_,
                {
                    "p": {
                        "default": 0.25,
                    },
                },
            ),
            "log_normal": (
                tt.log_normal_,
                {
                    "mean": {
                        "default": 1.0,
                    },
                    "std": {
                        "default": 2.0,
                    },
                },
            ),
            "normal": (
                tt.normal_,
                {
                    "mean": {
                        "default": 0.0,
                    },
                    "std": {
                        "default": 1.0,
                    },
                },
            ),
            # Complex distros
            "beta": (
                td.Beta,
                {
                    "concentration0": {
                        "default": "0.5",
                    },
                    "concentration1": {
                        "default": "0.5",
                    },
                },
            ),
            "continuous_bernoulli": (
                td.ContinuousBernoulli,
                {
                    "probs": {
                        "default": "0.5",
                    },
                },
            ),
            "dirichlet": (
                td.Dirichlet,
                {
                    "concentration": {
                        "default": "0.5 0.5",
                    },
                },
            ),
            "fisher_snedecor": (
                td.FisherSnedecor,
                {
                    "df1": {
                        "default": "1.0",
                    },
                    "df2": {
                        "default": "2.0",
                    },
                },
            ),
            "gamma": (
                td.Gamma,
                {
                    "concentration": {
                        "default": "1.0",
                    },
                    "rate": {
                        "default": "1.0",
                    },
                },
            ),
            "gumbel": (
                td.Gumbel,
                {
                    "loc": {
                        "default": "1.0",
                    },
                    "scale": {
                        "default": "2.0",
                    },
                },
            ),
            "inverse_gamma": (
                td.InverseGamma,
                {
                    "concentration": {
                        "default": "1.0",
                    },
                    "rate": {
                        "default": "1.0",
                    },
                },
            ),
            "kumaraswamy": (
                td.Kumaraswamy,
                {
                    "concentration0": {
                        "default": "1.0",
                    },
                    "concentration1": {
                        "default": "1.0",
                    },
                },
            ),
            "laplacian": (
                td.Laplace,
                {
                    "loc": {
                        "default": "0.0",
                    },
                    "scale": {
                        "default": "1.0",
                    },
                },
            ),
            "lkjcholesky": (
                td.LKJCholesky,
                {
                    "dim": {
                        "_ty": "INT",
                        "default": 3,
                    },
                    "concentration": {
                        "default": "1.0",
                    },
                },
            ),
            "lrmvariate_normal": (
                lambda loc, cov_factor, cov_diag: td.LowRankMultivariateNormal(
                    loc=loc,
                    cov_factor=cov_factor.reshape(loc.numel(), -1),
                    cov_diag=cov_diag,
                ),
                {
                    "loc": {
                        "default": "0.0 0.0",
                    },
                    "cov_factor": {
                        "default": "1.0 0.0",
                    },
                    "cov_diag": {
                        "default": "1.0 1.0",
                    },
                },
            ),
            "mvariate_normal": (
                lambda loc, cov_multiplier=1.0: td.MultivariateNormal(
                    loc=loc,
                    covariance_matrix=torch.eye(
                        loc.numel(),
                        dtype=loc.dtype,
                        device=loc.device,
                    ).mul_(cov_multiplier),
                ),
                {
                    "loc": {
                        "default": "0.0 0.0",
                    },
                    "cov_multiplier": {
                        "default": 1.0,
                    },
                },
            ),
            "pareto": (
                td.Pareto,
                {
                    "scale": {
                        "default": "1.0",
                    },
                    "alpha": {
                        "default": "1.0",
                    },
                },
            ),
            "poisson": (
                td.Poisson,
                {
                    "rate": {
                        "default": "1.5",
                    },
                },
            ),
            "relaxed_bernoulli": (
                td.RelaxedBernoulli,
                {
                    "temperature": {
                        "default": 0.75,
                    },
                    "probs": {
                        "default": "0.66",
                    },
                },
            ),
            "relaxed_onehotcategorical": (
                td.RelaxedOneHotCategorical,
                {
                    "temperature": {
                        "default": 1.5,
                    },
                    "probs": {
                        "default": "0.33 0.66",
                    },
                },
            ),
            "studentt": (
                td.StudentT,
                {
                    "loc": {
                        "default": "0.0",
                    },
                    "scale": {
                        "default": "1.0",
                    },
                    "df": {
                        "default": "1.0",
                    },
                },
            ),
            "uniform": (
                td.Uniform,
                {
                    "low": {
                        "default": 0.0,
                    },
                    "high": {
                        "default": 1.0,
                    },
                },
            ),
            "vonmises": (
                td.VonMises,
                {
                    "loc": {
                        "default": "1.0",
                    },
                    "concentration": {
                        "default": "1.0",
                    },
                },
            ),
            "weibull": (
                td.Weibull,
                {
                    "scale": {
                        "default": "1.0",
                    },
                    "concentration": {
                        "default": "1.0",
                    },
                },
            ),
            "wishart": (
                lambda df, cov_size=2, cov_multiplier=1.0: td.Wishart(
                    df=df,
                    covariance_matrix=torch.eye(
                        int(cov_size),
                        dtype=df.dtype,
                        device=df.device,
                    ).mul_(cov_multiplier),
                ),
                {
                    "df": {
                        "default": "2.0",
                    },
                    "cov_size": {
                        "_ty": "INT",
                        "default": 2,
                    },
                    "cov_multiplier": {
                        "default": 1.0,
                    },
                },
            ),
        }
        return cls._distro_params

    _build_params = None

    @classmethod
    def build_params(cls):
        if cls._build_params is not None:
            return cls._build_params
        cls._build_params = {
            f"{tykey}_{pkey}": pval
            for tykey, tyval in cls.distro_params().items()
            for pkey, pval in tyval[1].items()
            if not pkey.startswith("_")
        }
        return cls._build_params

    _ng_params = None

    @classmethod
    def ng_params(cls):
        if cls._ng_params is not None:
            return cls._ng_params
        dparams = {
            k: v["default"]
            for k, v in cls.build_params().items()
            if not k.startswith("_")
        }
        cls._ng_params = (
            super().ng_params()
            | {
                "distro": "normal",
                "quantile_norm": 0.85,
                "quantile_norm_flatten": True,
                "quantile_norm_dim": 1,
                "quantile_norm_pow": 0.5,
                "quantile_norm_fac": 1.0,
                "result_index": "-1",
            }
            | dparams
        )
        return cls._ng_params

    def norm_output(self, noise):
        if noise.ndim > len(self.shape):
            if noise.shape[: len(self.shape)] != self.shape:
                errstr = f"Unexpected shape when normalizing distro({self.distro}) noise! Output shape={self.shape}, noise shape={noise.shape}, generator dump: {self}"
                raise RuntimeError(errstr)
            selfdims = len(self.shape)
            result_index = self.result_index
            if not isinstance(result_index, (tuple, list)):
                result_index = (result_index,)
            ri_len = len(result_index)
            if ri_len == 0:
                raise ValueError("When result_index is a list, it must not be empty")
            trim_count = 0
            while noise.ndim > selfdims:
                idx = result_index[trim_count % ri_len]
                if idx < 0:
                    idx = noise.shape[-1] + idx
                noise = noise[..., max(0, min(noise.shape[-1] - 1, idx))]
                trim_count += 1
        return (
            quantile_normalize(
                noise,
                quantile=self.quantile_norm,
                dim=self.quantile_norm_dim,
                flatten=self.quantile_norm_flatten,
                nq_fac=self.quantile_norm_fac,
                pow_fac=self.quantile_norm_pow,
            )
            .reshape(self.shape)
            .contiguous()
        )

    def distro_param(self, val, *, simple_fun=None):
        if isinstance(val, torch.Tensor):
            return simple_fun(val) if simple_fun is not None else val
        if isinstance(val, str):
            val = tuple(float(v) for v in val.split(None))
        if simple_fun is not None:
            if isinstance(val, (float, int)):
                return simple_fun(val)
            if len(val) > 1:
                raise ValueError("Couldn't return result as float")
            return simple_fun(val[0])
        if not isinstance(val, (tuple, list)):
            val = (val,)
        return torch.tensor(
            val,
            dtype=self.dtype,
            device=self.gen_device,
        )

    def get_distro_kwargs(self, distro, ddef, *, simple=False):
        return {
            k: self.distro_param(
                getattr(self, f"{distro}_{k}"),
                simple_fun=None
                if not simple and k != "dim"
                else (int if k == "dim" else float),
            )
            for k in ddef
        }

    def generate(self, *_args):
        distro = self.distro
        dfun, ddef = self.distro_params()[distro]
        is_simple = distro in self.simple_distros
        dkwargs = self.get_distro_kwargs(distro, ddef, simple=is_simple)
        if is_simple:
            noise = torch.empty(
                *self.shape,
                device=self.gen_device,
                dtype=self.dtype,
                layout=self.layout,
            )
            noise = dfun(noise, **dkwargs)
        else:
            dobj = dfun(**dkwargs)
            noise = (
                dobj.rsample if getattr(dobj, "has_rsample", False) else dobj.sample
            )(self.shape)
        return self.norm_output(noise)


class PowerOldNoiseGenerator(NoiseGenerator):
    name = "power_old"

    @classmethod
    def ng_params(cls):
        return super().ng_params() | {"alpha": 2, "k": 1, "normalized": False}

    def generate(self, *_args):
        tensor = self.rand_like()
        fft = torch.fft.fft2(tensor)
        freq = torch.arange(
            1,
            len(fft) + 1,
            dtype=tensor.dtype,
            layout=tensor.layout,
            device=tensor.device,
        ).reshape(
            (len(fft),) + (1,) * (tensor.dim() - 1),
        )
        spectral_density = self.k / freq**self.alpha
        noise = torch.rand(
            tensor.shape,
            device=tensor.device,
            layout=tensor.layout,
            dtype=tensor.dtype,
        ).mul_(spectral_density)
        mean = torch.mean(noise, dim=(-2, -1), keepdim=True)
        std = torch.std(noise, dim=(-2, -1), keepdim=True)
        return noise.sub_(mean).div_(std)


# Idea from https://github.com/ClownsharkBatwing/RES4LYF/ (wave and mode defaults also from that source)
class WaveletFilteredNoiseGenerator(FramesToChannelsNoiseGenerator):
    name = "waveletfilter"
    MIN_DIMS = 4
    MAX_DIMS = 5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        inv_kwargs = {
            k: self.options[k]
            for k in ("inv_mode", "inv_biort", "inv_qshift", "inv_wave")
            if k in self.options
        }
        self.wavelet = Wavelet(
            wave=self.wave,
            level=self.level,
            mode=self.mode,
            use_1d_dwt=self.use_1d_dwt,
            use_dtcwt=self.use_dtcwt,
            biort=self.biort,
            qshift=self.qshift,
            device=self.gen_device,
            **inv_kwargs,
        )

    @classmethod
    def ng_params(cls):
        return super().ng_params() | {
            "mode": "periodization",
            "level": 3,
            "wave": "haar",
            "use_1d_dwt": False,
            "use_dtcwt": False,
            "qshift": "qshift_a",
            "biort": "near_sym_a",
            "yl_scale": 1.0,
            "yh_scales": 1.0,
            "two_step_inverse": False,
            "preblend_yl_scale_low": None,
            "preblend_yh_scales_low": None,
            "preblend_yl_scale_high": None,
            "preblend_yh_scales_high": None,
            "yl_blend_function": torch.lerp,
            "yh_blend_function": torch.lerp,
            "yl_blend_high": 0.0,
            "yh_blend_high": 1.0,
            "noise_sampler": None,
            "noise_sampler_high": None,
        }

    def _fix_shape(self, noise, adjusted_shape):
        if noise.shape != adjusted_shape:
            noise = noise.reshape(*adjusted_shape)
        if self.frames:
            noise = noise.reshape(
                self.batch,
                self.channels * self.frames,
                self.height,
                self.width,
            )
        return noise

    def generate(self, *args):
        adjusted_shape = self.get_adjusted_shape()
        noise = (
            self.rand_like()
            if self.noise_sampler is None
            else self.noise_sampler(*args)
        )
        if self.noise_sampler_high is not None:
            noise_high = self._fix_shape(self.noise_sampler_high(*args), adjusted_shape)
        else:
            noise_high = None
        noise = self._fix_shape(noise, adjusted_shape)
        orig_noise_shape = noise.shape
        need_flat = not self.use_dtcwt and self.use_1d_dwt and noise.ndim > 3
        if need_flat:
            noise = noise.flatten(start_dim=2)
            if noise_high is not None:
                noise_high = noise_high.flatten(start_dim=2)
        yl, yh = self.wavelet.forward(noise)
        if noise_high is not None:
            yl_high, yh_high = self.wavelet.forward(noise_high)
            if (
                self.preblend_yl_scale_high is not None
                or self.preblend_yh_scales_high is not None
            ):
                yl_high, yh_high = wavelet_scaling(
                    yl_high,
                    yh_high,
                    fallback(self.preblend_yl_scale_high, 1.0),
                    fallback(self.preblend_yh_scales_high, 1.0),
                )
            if (
                self.preblend_yl_scale_low is not None
                or self.preblend_yh_scales_low is not None
            ):
                yl, yh = wavelet_scaling(
                    yl,
                    yh,
                    fallback(self.preblend_yl_scale_low, 1.0),
                    fallback(self.preblend_yh_scales_low, 1.0),
                )
            yl, yh = wavelet_blend(
                (yl, yh),
                (yl_high, yh_high),
                yl_factor=self.yl_blend_high,
                yh_factor=self.yh_blend_high,
                blend_function=self.yl_blend_function,
                yh_blend_function=self.yh_blend_function,
            )
            del noise_high, yl_high, yh_high
        yl, yh = wavelet_scaling(
            yl,
            yh,
            self.yl_scale,
            self.yh_scales,
            in_place=True,
        )
        result = self.wavelet.inverse(yl, yh, two_step_inverse=self.two_step_inverse)
        if need_flat:
            result = result.reshape(orig_noise_shape)
        result = self.fix_output_frames(result)
        if result.shape == noise.shape:
            return result
        return result[tuple(slice(0, dl) for dl in noise.shape)]


class ScatternetFilteredNoiseGenerator(FramesToChannelsNoiseGenerator):
    name = "scatternetfilter"
    MIN_DIMS = 4
    MAX_DIMS = 4

    def __init__(self, *args, **kwargs):
        if ptwav is None:
            raise RuntimeError(
                "Scatternet noise requires the pytorch_wavelets package to be installed in your Python environment",
            )
        super().__init__(*args, **kwargs)
        if self.output_mode not in {
            "channels",
            "channels_adjusted",
            "channels_scaled",
            "flat",
            "flat_adjusted",
            "flat_scaled",
        }:
            raise ValueError("Bad output mode")

        scatkwargs = {
            "mode": self.mode,
            "biort": "near_sym_b_bp" if self.use_symmetric_filter else self.biort,
        }
        if self.scatternet_order == 2:
            scatkwargs["qshift"] = (
                "qshift_b_bp" if self.use_symmetric_filter else self.qshift
            )
            self.scatternet = ptwav.ScatLayerj2(**scatkwargs)
        elif self.scatternet_order == 1:
            self.scatternet = ptwav.ScatLayer(**scatkwargs)
        else:
            self.scatternet = torch.nn.Sequential(
                *(
                    ptwav.ScatLayer(**scatkwargs)
                    for _ in range(abs(self.scatternet_order))
                ),
            )

    @classmethod
    def ng_params(cls):
        return super().ng_params() | {
            "mode": "symmetric",
            "magbias": 1e-02,
            "use_symmetric_filter": False,
            "biort": "near_sym_a",
            "qshift": "qshift_a",
            "output_offset": 0.0,
            "scatternet_order": 1,
            "per_channel_scatternet": False,
            "output_mode": "channels_adjusted",
            # If None, uses probselect when available, otherwise bilinear.
            "upscale_mode": None,
            "noise_sampler": None,
        }

    def _fix_shape(self, noise, adjusted_shape):
        if self.frames:
            noise = noise.reshape(
                self.batch,
                self.channels * self.frames,
                self.height,
                self.width,
            )
        elif noise.shape != adjusted_shape:
            noise = noise.reshape(*adjusted_shape)
        return noise

    def generate(self, *args):
        adjusted_shape = self.get_adjusted_shape()
        scaled = self.output_mode.endswith("_scaled")
        adjusted = scaled or self.output_mode.endswith("_adjusted")
        order = abs(self.scatternet_order)
        order_spatial_compensation = 2**order
        output_mode = (
            self.output_mode.split("_", 1)[0] if adjusted else self.output_mode
        )
        spatial_compensation = 1 if adjusted else order_spatial_compensation
        if self.noise_sampler is None:
            temp_shape = (
                (
                    *adjusted_shape[:2],
                    adjusted_shape[-2] * spatial_compensation,
                    adjusted_shape[-1] * spatial_compensation,
                )
                if spatial_compensation != 1
                else adjusted_shape
            )
            noise = self.rand_like(shape=temp_shape)
        else:
            noise = self.noise_sampler(*args)
        if scaled:
            upscale_mode = self.upscale_mode
            if upscale_mode is None:
                upscale_mode = (
                    "probselect"
                    if "probselect" in utils.UPSCALE_METHODS
                    else "bilinear"
                )
            noise = utils.scale_samples(
                noise,
                adjusted_shape[-1] * order_spatial_compensation,
                adjusted_shape[-2] * order_spatial_compensation,
                mode=upscale_mode,
            )
        if self.scatternet_order == 0:
            return self.fix_output_frames(noise)
        self.scatternet = self.scatternet.to(device=self.device, dtype=self.dtype)
        if self.per_channel_scatternet:
            # To C, B, 1, H, W
            noise = torch.stack(
                tuple(
                    self.scatternet(noise[:, chan : chan + 1])
                    for chan in range(self.channels)
                ),
                dim=0,
            )
        else:
            # To 1, B, C, H, W
            noise = self.scatternet(noise)[None]
        base_channels = 1 if self.per_channel_scatternet else self.channels
        if output_mode == "flat":
            noise = noise.reshape(noise.shape[0], self.batch, -1)
            initial_size = math.prod(
                self.shape[(2 if self.per_channel_scatternet else 1) :],
            )
        elif adjusted:
            initial_size = base_channels
        else:
            initial_size = base_channels * ((2**order) ** 2)
        increment = 1 if output_mode == "flat" else base_channels
        out_size = noise.shape[2]
        offset_size = (out_size - initial_size) / increment
        output_offset = self.output_offset
        if output_offset == 0 or abs(output_offset) >= 1:
            output_offset = int(output_offset)
            if output_offset < 0:
                output_offset = (offset_size + 1) + output_offset
        else:
            if output_offset < 0:
                output_offset += 1.0
            output_offset = round(offset_size * output_offset)
        base_idx = int(output_offset * increment)
        # print(
        #     f"\nSCAT: shape={noise.shape}, adj_shape={adjusted_shape}, offset={output_offset}, initial_size={initial_size}, out_size={out_size}, offset_size={offset_size}, incr={increment}, base_idx={base_idx}",
        # )
        noise = noise[:, :, base_idx : base_idx + initial_size]
        # print(f"\nSCAT2: {noise.shape}")
        noise = (
            noise.squeeze(2).movedim(0, 1) if self.per_channel_scatternet else noise[0]
        )
        # print(f"\nSCAT3: {noise.shape}")
        if output_mode == "channels":
            noise = noise[..., : self.height, : self.width]
        # print(
        #     f"\nSCAT4: {noise.shape} -> {adjusted_shape} -- numel: {noise.numel()}, adjnumel={math.prod(adjusted_shape)}",
        # )
        return noise.reshape(adjusted_shape).contiguous()


class WaveletNoiseOctave(NamedTuple):
    octave: int
    height: int
    width: int
    amplitude: float
    total_amplitude: float


class WaveletNoiseGenerator(FramesToChannelsNoiseGenerator):
    name = "wavelet"
    MIN_DIMS = 4
    MAX_DIMS = 5

    @classmethod
    def ng_params(cls):
        return super().ng_params() | {
            "octave_scale_mode": "adaptive_avg_pool2d",
            "octave_rescale_mode": "bilinear",
            "post_octave_rescale_mode": "bilinear",
            "initial_amplitude": 1.0,
            "persistence": 0.5,
            "octaves": 4,
            "octave_height_factor": 0.5,
            "octave_width_factor": 0.5,
            "height_factor": 2.0,
            "width_factor": 2.0,
            "min_height": 4,
            "min_width": 4,
            "update_blend": 1.0,
            "update_blend_function": torch.lerp,
            "noise_sampler": None,
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_octave_data()

    def set_internal_noise_sampler(self, noise_sampler: object) -> None:
        self.noise_sampler = noise_sampler

    def set_octave_data(self) -> tuple:
        adjusted_shape = self.get_adjusted_shape()
        height, width = adjusted_shape[-2:]
        amplitude = self.initial_amplitude
        total_amplitude = 0.0
        curr_height, curr_width = height, width
        octave_data = []
        is_reverse = self.octaves < 0
        octaves = (
            range(self.octaves)
            if not is_reverse
            else reversed(range(abs(self.octaves)))
        )
        for octave in octaves:
            curr_height /= self.height_factor**octave
            curr_width /= self.width_factor**octave
            if (
                amplitude == 0
                or curr_height < self.min_height
                or curr_width < self.min_width
                or curr_height * self.octave_height_factor < 1
                or curr_width * self.octave_width_factor < 1
            ):
                if is_reverse and not octave_data:
                    curr_height, curr_width = height, width
                    continue
                break
            total_amplitude += abs(amplitude)
            octave_data.append(
                WaveletNoiseOctave(
                    octave=octave,
                    height=curr_height,
                    width=curr_width,
                    amplitude=amplitude,
                    total_amplitude=total_amplitude,
                ),
            )
            amplitude *= self.persistence
        if not octave_data or not total_amplitude:
            raise ValueError("Unworkable parameters for wavelet noise")
        self.octave_data = tuple(octave_data)

    def _generate_octave(self, *args: list, shape: Sequence) -> torch.Tensor:
        height, width = shape[-2:]
        noise = (
            self.noise_sampler(*args)[..., :height, :width].reshape(shape)
            if self.noise_sampler
            else self.rand_like(shape=(*shape[:-2], height, width))
        )
        scaled_height = int(max(1, height * self.octave_height_factor))
        scaled_width = int(max(1, width * self.octave_width_factor))
        scaled_noise = utils.scale_samples(
            utils.scale_samples(
                noise,
                scaled_width,
                scaled_height,
                mode=self.octave_scale_mode,
            ),
            width=width,
            height=height,
            mode=self.octave_rescale_mode,
        )
        return self.update_blend_function(
            noise,
            noise - scaled_noise,
            self.update_blend,
        )

    def generate(self, *args: list) -> torch.Tensor:
        adjusted_shape = self.get_adjusted_shape()
        height, width = adjusted_shape[-2:]
        curr_shape = list(adjusted_shape)
        result = torch.zeros(
            adjusted_shape,
            device=self.device,
            dtype=self.dtype,
            layout=self.layout,
        )
        for od in self.octave_data:
            curr_shape[-2:] = (int(od.height), int(od.width))
            octave_output = self._generate_octave(*args, shape=curr_shape)
            if octave_output.shape != result.shape:
                octave_output = utils.scale_samples(
                    octave_output,
                    width,
                    height,
                    mode=self.post_octave_rescale_mode,
                )
            result += octave_output.mul_(od.amplitude)
        if self.octave_data[-1].total_amplitude != 0:
            result /= self.octave_data[-1].total_amplitude
        return self.fix_output_frames(result)


class CollatzNoiseGenerator(NoiseGenerator):
    name = "collatz"

    chain_cache: ClassVar[dict] = {}

    @classmethod
    def ng_params(cls):
        return super().ng_params() | {
            "adjust_scale": False,
            "iteration_sign_flipping": True,
            "chain_length": (1, 1, 2, 2, 3, 3),
            "iterations": 10,
            "rmin": -8000.0,
            "rmax": 8000.0,
            "flatten": False,
            "dims": (-1, -1, -2, -2),
            # values, ratios, mults, adds
            # seed_x_ratios, seed_x_mults, seed_x_adds
            # noise_x_ratios, noise_x_mults, noise_x_adds
            "output_mode": "values",
            "quantile": 0.5,
            "quantile_strategy": "clamp",
            "noise_dtype": torch.float32,
            "integer_math": True,
            "even_multiplier": 0.5,
            "even_addition": 0.0,
            "odd_multiplier": 3.0,
            "odd_addition": 1.0,
            "add_preserves_sign": True,
            "chain_offset": 5,
            "break_loops": True,
            "seed_mode": "default",
            "seed_noise_sampler": None,
            "mix_noise_sampler": None,
        }

    @staticmethod
    def _get_iter_slices(n_dims, dim, idx, stride) -> list:
        result = [slice(None)] * n_dims
        result[dim] = slice(idx, None, stride)
        return result

    def _generate_iteration(
        self,
        *args,
        dim: int,
        chain_length: int,
        flatten: False,
        shape=None,
    ):
        dtype, device = self.dtype, self.device
        out_shape = shape = fallback(shape, self.shape)
        if dim >= len(shape):
            raise ValueError("Requested dimension out of range")
        rmin, rmax = self.rmin, self.rmax
        emul, eadd = self.even_multiplier, self.even_addition
        omul, oadd = self.odd_multiplier, self.odd_addition
        keepsign = self.add_preserves_sign
        intmode = self.integer_math
        rmaxsubmin = rmax - rmin
        if flatten:
            shape = torch.Size((*shape[:dim], math.prod(shape[dim:])))
        size = shape[dim]
        chain_length = min(size, chain_length)
        n_chunks = math.ceil(size / chain_length)
        chain_length += self.chain_offset
        result_shape = list(shape)
        chunk_shape = result_shape.copy()
        result_shape[dim] = chain_length * n_chunks
        chunk_shape[dim] = n_chunks
        result = torch.zeros(result_shape, dtype=self.noise_dtype, device=device)
        adds, muls = result.clone(), result.clone()
        if self.seed_noise_sampler is not None:
            orig_noise = self.seed_noise_sampler(*args)[
                tuple(slice(None, sz) for sz in chunk_shape)
            ].to(result)
            if flatten:
                orig_noise = orig_noise.flatten(start_dim=dim)
            orig_noise = normalize_to_scale(
                orig_noise[tuple(slice(None, sz) for sz in chunk_shape)],
                1e-06,
                1.0,
                dim=tuple(range(1, len(chunk_shape))),
            )
        else:
            orig_noise = self.rand_like(
                fun=torch.rand,
                shape=chunk_shape,
                dtype=result.dtype,
            )
        noise = orig_noise * (rmaxsubmin + 1) + rmin
        # Derp.
        noise = torch.where(noise == 0, noise.max() / noise.numel(), noise)
        if self.seed_mode != "default":
            noise = torch.where(
                (noise % 2.0) < 1
                if self.seed_mode == "force_odd"
                else (noise % 2.0) >= 1,
                noise + 1,
                noise,
            )
        if noise.device != self.device:
            noise = tensor_to(noise, self.device)
        slice_0 = self._get_iter_slices(result.ndim, dim, 0, chain_length)
        for chainidx in range(chain_length):
            if chainidx == 0:
                muls[slice_0] = 1.0
                result[slice_0] = noise
                continue
            slice_curr = self._get_iter_slices(result.ndim, dim, chainidx, chain_length)
            slice_prev = self._get_iter_slices(
                result.ndim,
                dim,
                chainidx - 1,
                chain_length,
            )
            prev = result[slice_prev]
            prev_trunc = utils.trunc_decimals(prev, 2)
            need_reset = (
                ((prev_trunc >= 1.0) & (prev_trunc < 1.001))
                | (prev_trunc.abs() < 0.001)
                if self.break_loops
                else False
            )
            prev_evens = prev % 2 < 1.0
            prev_adds, prev_muls = adds[slice_prev], muls[slice_prev]
            muls_next = (
                torch.where(
                    prev_evens,
                    prev_muls if emul == 1 else prev_muls * emul,
                    prev_muls if omul == 1 else prev_muls * omul,
                )
                if emul != 1 or omul != 1
                else prev_muls
            )
            muls[slice_curr] = (
                torch.where(need_reset, 1.0, muls_next)
                if need_reset is not False
                else muls_next
            )
            curr_muls = muls[slice_curr]
            prev_adds_scaled = prev_adds * curr_muls
            prev_sign = prev.sign() if keepsign else 1.0
            adds_next = (
                torch.where(
                    prev_evens,
                    prev_adds_scaled
                    if eadd == 0
                    else prev_adds_scaled + eadd * prev_sign,
                    prev_adds_scaled
                    if oadd == 0
                    else prev_adds_scaled + oadd * prev_sign,
                )
                if eadd != 0 or oadd != 0
                else prev_adds_scaled
            )
            adds[slice_curr] = (
                torch.where(need_reset, 0.0, adds_next)
                if need_reset is not False
                else adds_next
            )
            curr_adds = adds[slice_curr]
            result_next = utils.maybe_apply(
                (noise * curr_muls).add_(curr_adds),
                intmode,
                torch.trunc,
            )
            result[slice_curr] = (
                torch.where(need_reset, noise, result_next)
                if need_reset is not False
                else result_next
            )
        output_slice = tuple(
            slice(None, sz) for sz in (shape if flatten else out_shape)
        )
        return self._iteration_output(
            *args,
            result_chains=result,
            orig_noise=orig_noise,
            noise=noise,
            raw_adds=adds,
            muls=muls,
            chain_length=chain_length,
            dim=dim,
            output_shape=out_shape,
            output_slice=output_slice,
            dtype=dtype,
        )

    def _trim_chain_offset(
        self,
        t: torch.Tensor,
        dim: int,
        chain_length: int,
    ) -> torch.Tensor:
        co = self.chain_offset
        if co < 1:
            return t
        chunks = t.split(chain_length, dim)
        slices = [slice(None)] * t.ndim
        slices[dim] = slice(co, None)
        return torch.cat(
            tuple(chunk[slices] for chunk in chunks),
            dim=dim,
        )

    def _iteration_output(
        self,
        *args,
        result_chains: torch.Tensor,
        orig_noise: torch.Tensor,
        noise: torch.Tensor,
        raw_adds: torch.Tensor,
        muls: torch.Tensor,
        chain_length: int,
        dim: int,
        output_shape: Sequence,
        output_slice: Sequence,
        dtype: str | torch.dtype,
    ) -> torch.Tensor:
        omode = self.output_mode
        quantile = self.quantile
        noise_exp = noise.repeat_interleave(chain_length, dim)
        nadds = raw_adds.div_(noise_exp)
        ratios = result_chains / noise_exp
        if omode in {"values", "ratios", "seed_x_ratios", "noise_x_ratios"}:
            out1 = ratios
        elif omode in {"mults", "seed_x_mults", "noise_x_mults"}:
            out1 = muls
        elif omode in {"adds", "seed_x_adds", "noise_x_adds"}:
            out1 = nadds
        else:
            raise ValueError("Bad output mode")
        out1 = self._trim_chain_offset(out1, dim=dim, chain_length=chain_length)
        if quantile not in {0, 1}:
            out1 = utils.quantile_normalize(
                out1,
                quantile=quantile,
                dim=0,
                strategy=self.quantile_strategy,
            )
        out1 = out1[output_slice].reshape(output_shape).to(dtype=dtype)
        if omode in {"ratios", "mults", "adds"}:
            return out1
        if omode in {"values", "seed_x_ratios", "seed_x_mults", "seed_x_adds"}:
            out2 = orig_noise.repeat_interleave(chain_length - self.chain_offset, dim)
        elif omode in {"noise_x_ratios", "noise_x_mults", "noise_x_adds"}:
            out2 = (
                self.rand_like(dtype=out1.dtype)
                if self.mix_noise_sampler is None
                else self.mix_noise_sampler(*args)
            )
        out2 = out2[output_slice].reshape(output_shape).to(dtype=dtype)
        return out2 * out1

    def generate(self, *args):
        out_dims = len(self.shape)
        dims = tuple(dim if dim >= 0 else out_dims + dim for dim in self.dims)
        n_dims, n_chainlens = len(dims), len(self.chain_length)
        if not all(0 <= d < out_dims for d in dims):
            raise ValueError("Dimension out of range")
        dtype, device = self.dtype, self.device
        result = torch.zeros(self.shape, dtype=dtype, device=device)
        it_scale = 1.0 / self.iterations
        for iteration in range(self.iterations):
            if iteration > 0 and (iteration % 25) == 0:
                # It's soooo slow!
                throw_exception_if_processing_interrupted()
            temp = self._generate_iteration(
                *args,
                dim=dims[iteration % n_dims],
                chain_length=self.chain_length[iteration % n_chainlens],
                flatten=self.flatten,
            ).mul_(
                it_scale
                * (-1 if self.iteration_sign_flipping and (iteration & 1) == 1 else 1),
            )
            result += temp
        if self.adjust_scale:
            result = normalize_to_scale(
                result,
                -1.0,
                1.0,
                dim=tuple(range(1 if result.ndim < 4 else 2, result.ndim)),
            )
        return result


__all__ = (
    "BrownianNoiseGenerator",
    "CollatzNoiseGenerator",
    "DistroNoiseGenerator",
    "GaussianNoiseGenerator",
    "GreenTestNoiseGenerator",
    "HighresPyramidNoiseGenerator",
    "LaplacianNoiseGenerator",
    "MixedNoiseGenerator",
    "NoiseError",
    "NoiseType",
    "OneFNoiseGenerator",
    "PerlinOldNoiseGenerator",
    "PinkOldNoiseGenerator",
    "PowerLawNoiseGenerator",
    "PowerOldNoiseGenerator",
    "PyramidNoiseGenerator",
    "PyramidOldNoiseGenerator",
    "ScatternetFilteredNoiseGenerator",
    "StudentTNoiseGenerator",
    "UniformNoiseGenerator",
    "WaveletFilteredNoiseGenerator",
    "WaveletNoiseGenerator",
)
