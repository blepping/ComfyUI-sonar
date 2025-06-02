# Noise generation functions shamelessly yoinked from https://github.com/Clybius/ComfyUI-Extra-Samplers
from __future__ import annotations

import math
from enum import Enum, auto
from functools import partial
from typing import Callable

import torch
from comfy.k_diffusion import sampling
from torch import FloatTensor, Generator, Tensor
from torch.distributions import Laplace, StudentT

try:
    import pytorch_wavelets as ptwav

    HAVE_WAVELETS = True
except ImportError:
    HAVE_WAVELETS = False

from . import utils
from .utils import (
    fallback,
    normalize_to_scale,
    quantile_normalize,
    scale_noise,
    tensor_to,
)

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
        # print("CREATE NG", self, kwargs)

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

    def rand_like(self, *, fun=torch.randn, cpu=None, to_device=True):
        cpu = cpu if cpu is not None else self.cpu
        noise = fun(
            *self.shape,
            generator=self.generator,
            dtype=self.dtype,
            layout=self.layout,
            device=self.gen_device,
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

    def rand_like(self, *args, **kwargs):
        noise = super().rand_like(*args, **kwargs)
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
        self.uniform_ng = UniformNoiseGenerator(
            *args,
            **(
                kwargs
                | {
                    "normalized": self.uniform_normalized,
                    "normalize_dims": self.options.get("uniform_normalize_dims"),
                }
            ),
        )

    @classmethod
    def ng_params(cls):
        return super().ng_params() | {
            "normalized": True,
            "uniform_normalized": False,
            "discount": 0.7,
            "upscale_mode": "bilinear",
            "iterations": 4,
        }

    def generate(self, s, sn):
        adjusted_shape = self.get_adjusted_shape()
        b, c, h, w = adjusted_shape
        orig_w, orig_h = w, h
        noise = self.uniform_ng(s, sn).reshape(*adjusted_shape)
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
class WaveletNoiseGenerator(FramesToChannelsNoiseGenerator):
    name = "wavelet"
    MIN_DIMS = 4
    MAX_DIMS = 5

    def __init__(self, *args, **kwargs):
        if not HAVE_WAVELETS:
            raise RuntimeError(
                "Wavelet noise requires the pytorch_wavelets package installed in your environment",
            )
        super().__init__(*args, **kwargs)
        if self.use_dtcwt:
            self.wavelet_forward = ptwav.DTCWTForward(
                J=self.level,
                mode=self.mode,
                biort=self.biort,
                qshift=self.qshift,
            ).to(self.gen_device)
            self.wavelet_inverse = ptwav.DTCWTInverse(
                mode=self.options.get("inv_mode", self.mode),
                biort=self.options.get("inv_biort", self.biort),
                qshift=self.options.get("inv_qshift", self.qshift),
            ).to(
                self.gen_device,
            )
        else:
            self.wavelet_forward = ptwav.DWTForward(
                J=self.level,
                wave=self.wave,
                mode=self.mode,
            ).to(self.gen_device)
            self.wavelet_inverse = ptwav.DWTInverse(
                wave=self.options.get("inv_wave", self.wave),
                mode=self.options.get("inv_mode", self.mode),
            ).to(
                self.gen_device,
            )

    @classmethod
    def ng_params(cls):
        return super().ng_params() | {
            "mode": "periodization",
            "level": 3,
            "wave": "haar",
            "use_dtcwt": False,
            "qshift": "qshift_a",
            "biort": "near_sym_a",
            "yl_scale": 1.0,
            "yh_scales": None,
            "noise_sampler": None,
        }

    def generate(self, *args):
        adjusted_shape = self.get_adjusted_shape()
        noise = (
            self.rand_like()
            if self.noise_sampler is None
            else self.noise_sampler(*args)
        )
        if noise.shape != adjusted_shape:
            noise = noise.reshape(*adjusted_shape)
        if self.frames:
            noise = noise.reshape(
                self.batch,
                self.channels * self.frames,
                self.height,
                self.width,
            )
        yl, yh = self.wavelet_forward(noise)
        if self.yl_scale != 1:
            yl *= self.yl_scale
        if self.yh_scales is not None:
            yh_scales = self.yh_scales
            if isinstance(yh_scales, (int, float)):
                yh_scales = (yh_scales,) * len(yh)
            # print("SCALES", self.yl_scale, yh_scales)
            for hscale, ht in zip(yh_scales, yh):
                # print(">> SCALING", hscale)
                if isinstance(hscale, (int, float)):
                    ht *= hscale  # noqa: PLW2901
                    continue
                for lidx in range(min(ht.shape[2], len(hscale))):
                    # print(">>    SCALE IDX", lidx)
                    ht[:, :, lidx, :, :] *= hscale[lidx]
        result = self.fix_output_frames(self.wavelet_inverse((yl, yh)))
        if result.shape == noise.shape:
            return result
        return result[tuple(slice(0, dl) for dl in noise.shape)]


class CollatzNoiseGenerator(NoiseGenerator):
    name = "collatz"

    @classmethod
    def ng_params(cls):
        return super().ng_params() | {
            "adjust_scale": True,
            "use_initial": True,
            "iteration_sign_flipping": True,
            "chain_length": (1, 1, 2, 2, 3, 3),
            "iterations": 500,
            "rmin": -8000.0,
            "rmax": 8000.0,
            "flatten": False,
            "dims": (-1, -1, -2, -2),
            "variant": 2,
        }

    @staticmethod
    def _get_iter_slices(n_dims, dim, offset, stride) -> tuple:
        return tuple(
            slice(None) if didx != dim else slice(offset, None, stride)
            for didx in range(n_dims)
        )

    def _generate_iteration(
        self,
        *,
        dim: int,
        chain_length: int,
        flatten: False,
        shape=None,
        integer_division=True,
    ):
        dtype, device = self.dtype, self.device
        out_shape = shape = fallback(shape, self.shape)
        if dim >= len(shape):
            raise ValueError("Requested dimension out of range")
        rmin, rmax = self.rmin, self.rmax
        rmaxsubmin = rmax - rmin
        if flatten:
            shape = torch.Size((*shape[:dim], math.prod(shape[dim:])))
        size = shape[dim]
        chain_length = min(size, chain_length)
        n_chunks = math.ceil(size / chain_length)
        result_shape = tuple(
            (chain_length * n_chunks) if idx == dim else sz
            for idx, sz in enumerate(shape)
        )
        chunk_shape = tuple(
            n_chunks if idx == dim else sz for idx, sz in enumerate(shape)
        )
        result = torch.zeros(result_shape, dtype=torch.float32, device=device)
        noise = (
            torch.rand(
                chunk_shape,
                generator=self.generator,
                dtype=torch.float32,
                device=self.gen_device,
                layout=self.layout,
            )
            .mul_(rmaxsubmin + 1)
            .add_(rmin)
        )
        # noise = torch.where((noise % 2.0) < 1, noise + 1, noise)
        if noise.device != self.device:
            noise = tensor_to(noise, self.device)
        for chainidx in range(chain_length):
            if chainidx == 0 and self.use_initial:
                result[self._get_iter_slices(result.ndim, dim, 0, chain_length)] = noise
                continue
            chunk = (
                noise
                if chainidx == 0
                else result[
                    self._get_iter_slices(result.ndim, dim, chainidx - 1, chain_length)
                ]
            )
            result[self._get_iter_slices(result.ndim, dim, chainidx, chain_length)] = (
                torch.where(
                    chunk == 1,
                    noise,
                    torch.where(
                        chunk % 2 < 1,
                        chunk // 2 if integer_division else chunk / 2,
                        chunk * 3 + chunk.sign(),
                    ),
                )
            )
        result = result.sub_(rmin).div_(rmaxsubmin).to(dtype=dtype)
        return result[
            tuple(slice(None, sz) for sz in (shape if flatten else out_shape))
        ].reshape(out_shape)

    def generate(self, *_args):
        out_dims = len(self.shape)
        dims = tuple(dim if dim >= 0 else out_dims + dim for dim in self.dims)
        n_dims, n_chainlens = len(dims), len(self.chain_length)
        if not all(0 <= d < out_dims for d in dims):
            raise ValueError("Dimension out of range")
        dtype, device = self.dtype, self.device
        result = torch.zeros(self.shape, dtype=dtype, device=device)
        gen_function = partial(
            self._generate_iteration,
            integer_division=self.variant == 2,
        )
        it_scale = 1.0 / self.iterations
        for iteration in range(self.iterations):
            temp = gen_function(
                dim=dims[iteration % n_dims],
                chain_length=self.chain_length[iteration % n_chainlens],
                flatten=self.flatten,
            ).mul_(
                it_scale
                * (-1 if self.iteration_sign_flipping and (iteration & 1) == 1 else 1),
            )
            result += temp
        if self.adjust_scale:
            result = normalize_to_scale(result, -1.0, 1.0, dim=1)
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
    "StudentTNoiseGenerator",
    "UniformNoiseGenerator",
    "WaveletNoiseGenerator",
)
