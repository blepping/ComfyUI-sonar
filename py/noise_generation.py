# Noise generation functions shamelessly yoinked from https://github.com/Clybius/ComfyUI-Extra-Samplers
from __future__ import annotations

import math
from enum import Enum, auto
from functools import lru_cache
from typing import Callable

import torch
from comfy.k_diffusion import sampling
from comfy.model_management import device_supports_non_blocking
from comfy.utils import common_upscale
from torch import FloatTensor, Generator, Tensor
from torch.distributions import Laplace, StudentT

try:
    import pytorch_wavelets as ptwav

    HAVE_WAVELETS = True
except ImportError:
    HAVE_WAVELETS = False

from .external import MODULES as EXT

# ruff: noqa: D412, D413, D417, D212, D407, ANN002, ANN003, FBT001, FBT002, S311


class NoiseType(Enum):
    BROWNIAN = auto()
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


def scale_noise(
    noise,
    factor=1.0,
    *,
    normalized=True,
    threshold_std_devs=2.5,
    normalize_dims=None,
):
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


if "bleh" in EXT:
    scale_samples = EXT["bleh"].py.latent_utils.scale_samples
    BLENDING_MODES = EXT["bleh"].py.latent_utils.BLENDING_MODES
else:
    BLENDING_MODES = {"lerp": torch.lerp}

    def scale_samples(
        samples,
        width,
        height,
        *,
        mode="bicubic",
    ):
        return common_upscale(samples, width, height, mode, None)


CAN_NONBLOCK = {}


def tensor_to(tensor, dest):
    device = dest.device if isinstance(dest, torch.Tensor) else dest
    non_blocking = CAN_NONBLOCK.get(device)
    if non_blocking is None:
        non_blocking = device_supports_non_blocking(device)
        CAN_NONBLOCK[device] = non_blocking
    return tensor.to(dest, non_blocking=non_blocking)


class NoiseGenerator:
    name = "unknown"
    SAVE_X = False
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
        params = self.ng_params
        kwarg_params = params | kwargs
        for k in params:
            setattr(self, k, kwarg_params.pop(k))
        self.options = kwarg_params
        self.update_x(x)
        print("CREATE NG", self, kwargs)

    @classmethod
    @property
    def ng_params(cls):
        return {
            "normalized": True,
            "force_normalize": None,
            "normalize_dims": None,
            "cpu": True,
            "generator": None,
        }

    def update_x(self, x):
        self.x = None if not self.SAVE_X else x.detach().clone()
        self.shape = x.shape
        self.batch, self.channels, self.height, self.width = (
            x.shape if x.ndim == 4 else (None,) * 4
        )
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
        # print("GEN NOISE", noise)
        if to_device and noise.device != self.device:
            noise = tensor_to(noise, self.device)
        # print("MADE NOISE", noise)
        return noise

    def output_hook(self, noise):
        # print("NOISE OUT1", noise)
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
        pretty_params = ", ".join(f"{k}={getattr(self, k)!s}" for k in self.ng_params)
        return f"<NoiseGenerator({self.name}): device={self.device}, shape={self.shape}, dtype={self.dtype}, {pretty_params}>"


class MixedNoiseGenerator(NoiseGenerator):
    MIN_DIMS = MAX_DIMS = 4

    @classmethod
    @property
    def ng_params(cls):
        return super().ng_params | {
            "name": "mixed_noise",
            "normalized": True,
            "pass_args": frozenset(("cpu",)),
            "noise_mix": (),
            "output_fun": None,
        }

    def __init__(self, x, *args, **kwargs):
        super().__init__(x, *args, **kwargs)
        ng_list = []
        for item in self.noise_mix:
            if isinstance(item, (tuple, list)):
                ng_class, transform_fun = item
            else:
                ng_class, transform_fun = item, None
            ng_kwargs = {k: v for k, v in kwargs.items() if k in self.pass_args}
            ng_list.append((ng_class(x, **ng_kwargs), transform_fun))
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
    @property
    def ng_params(cls):
        return super().ng_params | {"normalized": False}

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
    @property
    def ng_params(cls):
        return super().ng_params | {"normalized": False}

    def generate(self, *args):
        return self.brownian_tree_ns(*args)


class PerlinOldNoiseGenerator(NoiseGenerator):
    name = "perlin_old"
    MIN_DIMS = MAX_DIMS = 4

    @classmethod
    @property
    def ng_params(cls):
        return super().ng_params | {
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
        blend = BLENDING_MODES[self.blend_mode]
        noise = self.rand_like(fun=torch.rand).div_(self.div_fac)

        _batch, channels, noise_height, noise_width = noise.shape
        for _ in range(self.iterations):
            noise += self.perlin_noise(
                (noise_height, noise_width),
                (noise_height, noise_width),
                batch_size=channels,  # This should be the number of channels.
                blend=blend,
                dtype=noise.dtype,
                layout=noise.layout,
                device=noise.device,
            )
        return noise


class UniformNoiseGenerator(NoiseGenerator):
    name = "uniform"

    @classmethod
    @property
    def ng_params(cls):
        return super().ng_params | {
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


class HighresPyramidNoiseGenerator(NoiseGenerator):
    name = "highres_pyramid"
    MIN_DIMS = MAX_DIMS = 4

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
    @property
    def ng_params(cls):
        return super().ng_params | {
            "normalized": True,
            "uniform_normalized": False,
            "discount": 0.7,
            "upscale_mode": "bilinear",
            "iterations": 4,
        }

    def generate(self, s, sn):
        (
            b,
            c,
            h,
            w,
        ) = self.shape  # EDIT: w and h get over-written, rename for a different variant!

        orig_w, orig_h = w, h
        noise = self.uniform_ng(s, sn)
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
            noise += scale_samples(
                tensor_to(torch.randn(b, c, h, w, generator=self.generator), noise),
                orig_w,
                orig_h,
                mode=self.upscale_mode,
            ).mul_(self.discount**i)
            if h >= orig_h * 15 or w >= orig_w * 15:
                break  # Lowest resolution is 1x1
        return noise


class PyramidOldNoiseGenerator(NoiseGenerator):
    name = "pyramid_old"
    MIN_DIMS = MAX_DIMS = 4

    @classmethod
    @property
    def ng_params(cls):
        return super().ng_params | {
            "discount": 0.8,
            "iterations": 5,
            "upscale_mode": "nearest-exact",
            "normalized": False,
        }

    def generate(self, *_args):
        b, c, h, w = self.shape
        orig_h, orig_w = h, w
        noise = torch.zeros(
            size=self.shape,
            dtype=self.dtype,
            layout=self.layout,
            device=self.gen_device,
        )
        r = 1
        for i in range(self.iterations):
            r *= 2
            noise += scale_samples(
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
        return noise


class PyramidNoiseGenerator(NoiseGenerator):
    name = "pyramid"
    MIN_DIMS = MAX_DIMS = 4

    @classmethod
    @property
    def ng_params(cls):
        return super().ng_params | {
            "discount": 0.7,
            "upscale_mode": "bilinear",
            "iterations": 10,
        }

    # Modified from https://wandb.ai/johnowhitaker/multires_noise/reports/Multi-Resolution-Noise-for-Diffusion-Model-Training--VmlldzozNjYyOTU2
    def generate(self, *_args):
        b, c, w, h = (
            self.shape
        )  # NOTE: w and h get over-written, rename for a different variant!

        orig_w, orig_h = w, h
        noise = self.rand_like()
        for i in range(self.iterations):
            r = (
                torch.rand(1, generator=self.generator).cpu().item() * 2 + 2
            )  # Rather than always going 2x,
            w, h = max(1, int(w / (r**i))), max(1, int(h / (r**i)))
            noise += scale_samples(
                torch.randn(
                    b,
                    c,
                    w,
                    h,
                    device=noise.device,
                    layout=noise.layout,
                    dtype=noise.dtype,
                ),
                orig_h,
                orig_w,
                mode=self.upscale_mode,
            ).mul_(
                self.discount**i,
            )
            if w == 1 or h == 1:
                break  # Lowest resolution is 1x1
        return noise


class StudentTNoiseGenerator(NoiseGenerator):
    name = "studentt"

    @classmethod
    @property
    def ng_params(cls):
        return super().ng_params | {
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


class GreenTestNoiseGenerator(NoiseGenerator):
    name = "green_test"
    MIN_DIMS = MAX_DIMS = 4

    @classmethod
    @property
    def ng_params(cls):
        return super().ng_params | {
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
        return torch.real(noise)


class PinkOldNoiseGenerator(NoiseGenerator):
    name = "pink_old"

    @classmethod
    @property
    def ng_params(cls):
        return super().ng_params | {"alpha": 2.0, "k": 1.0, "freq": 1.0}

    # Completely wrong implementation here.
    def generate(self, *_args):
        spectral_density = self.k / self.freq**self.alpha
        return self.rand_like() * spectral_density


class OneFNoiseGenerator(NoiseGenerator):
    name = "onef"
    MIN_DIMS = MAX_DIMS = 4

    @classmethod
    @property
    def ng_params(cls):
        return super().ng_params | {
            "alpha": 2.0,
            "k": 1.0,
            "hfac": 1.0,
            "wfac": 1.0,
            "base_power": 1.0,
            "use_sqrt": True,
        }

    # Referenced from: https://github.com/WASasquatch/PowerNoiseSuite
    def generate(self, *_args):
        batch, _channels, height, width = self.shape

        noise = self.rand_like()

        freq_x = tensor_to(torch.fft.fftfreq(height, self.hfac), noise)
        freq_y = tensor_to(torch.fft.fftfreq(width, self.wfac), noise)
        fx, fy = torch.meshgrid(freq_x, freq_y, indexing="ij")

        power = (fx**2 + fy**2) ** (-self.alpha / 2.0)
        if self.k != 0:
            power = self.k / power
        power[0, 0] = self.base_power
        power = power.unsqueeze(0).expand(batch, 1, height, width)

        noise_fft = torch.fft.fftn(noise)
        noise_fft /= (
            torch.sqrt(power.to(noise_fft.dtype))
            if self.use_sqrt
            else power.to(noise_fft.dtype)
        )

        return torch.fft.ifftn(noise_fft).real


class PowerLawNoiseGenerator(NoiseGenerator):
    name = "powerlaw"

    @classmethod
    @property
    def ng_params(cls):
        return super().ng_params | {
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
    @property
    def ng_params(cls):
        return super().ng_params | {"loc": 0, "scale": 1.0, "div_fac": 4.0}

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
        if self.distro not in self.distro_params:
            raise ValueError("Bad distro")

    @classmethod
    @property
    @lru_cache(maxsize=1)
    def distro_params(cls):
        td = torch.distributions
        tt = torch.Tensor
        return {
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
            "beta": (
                td.Beta,
                {
                    "alpha": {
                        "default": "0.5",
                    },
                    "beta": {
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
                    "alpha": {
                        "default": "1.0",
                    },
                    "beta": {
                        "default": "1.0",
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
        }

    @classmethod
    @lru_cache(maxsize=1)
    def build_params(cls):
        return {
            f"{tkey}_{pkey}": pval
            for tkey, tval in cls.distro_params.items()
            for pkey, pval in tval[1].items()
        }

    @classmethod
    @property
    @lru_cache(maxsize=1)
    def ng_params(cls):
        dparams = {k: v["default"] for k, v in cls.build_params().items()}
        return (
            super().ng_params
            | {
                "distro": "normal",
                "quantile_norm": 0.85,
                "quantile_norm_flatten": True,
                "quantile_norm_dim": 1,
                "quantile_norm_pow": 0.5,
                "quantile_norm_fac": 1.0,
                "result_index": -1,
            }
            | dparams
        )

    def norm_quantile(self, noise):
        if noise.ndim > len(self.shape):
            if (
                noise.ndim != len(self.shape) + 1
                or noise.shape[: len(self.shape)] != self.shape
            ):
                errstr = f"Unexpected shape when normalizing distro({self.distro}) noise! Output shape={self.shape}, noise shape={noise.shape}, generator dump: {self}"
                raise RuntimeError(errstr)
            idx = self.result_index
            if idx < 0:
                idx = noise.shape[-1] + idx
            noise = noise[..., max(0, min(noise.shape[-1] - 1, idx))]
        if (
            self.quantile_norm is None
            or self.quantile_norm <= 0
            or self.quantile_norm >= 1
        ):
            return noise
        qdim = self.quantile_norm_dim
        if qdim is not None and not self.quantile_norm_flatten:
            return self.norm_quantile_noflatten(noise)
        if qdim in {2, 3}:
            noise = noise.movedim(qdim, 1)
        tempshape = noise.shape
        if qdim is None:
            flatnoise = noise.flatten()
        elif qdim in {0, 1}:
            flatnoise = noise.flatten(start_dim=qdim + 1)
        elif qdim in {2, 3}:
            flatnoise = noise.flatten(start_dim=2)
        else:
            raise ValueError("Unexpected quantile_norm_dim!")
        nq = torch.quantile(
            flatnoise.abs(),
            self.quantile_norm,
            dim=-1,
        )
        del flatnoise
        nq_shape = tuple(nq.shape) + (1,) * (noise.ndim - nq.ndim)
        nq = nq.mul_(self.quantile_norm_fac).reshape(*nq_shape)
        noise = noise.clamp(-nq, nq)
        result = torch.copysign(
            torch.pow(torch.abs(noise), self.quantile_norm_pow),
            noise,
        ).reshape(tempshape)
        if qdim in {2, 3}:
            result = result.movedim(1, qdim)
        return result.reshape(self.shape).contiguous()

    def norm_quantile_noflatten(self, noise):
        qdim = self.quantile_norm_dim
        nq = torch.quantile(
            noise.abs(),
            self.quantile_norm,
            dim=qdim,
            keepdim=True,
        ).mul_(self.quantile_norm_fac)
        noise = noise.clamp(-nq, nq)
        return (
            torch.copysign(
                torch.pow(torch.abs(noise), self.quantile_norm_pow),
                noise,
            )
            .reshape(self.shape)
            .contiguous()
        )

    def distro_param(self, val, *, force_float=False):
        if isinstance(val, torch.Tensor):
            return float(val) if force_float else val
        if isinstance(val, str):
            val = tuple(float(v) for v in val.split(None))
        if force_float:
            if isinstance(val, (float, int)):
                return float(val)
            if len(val) > 1:
                raise ValueError("Couldn't return result as float")
            return float(val[0])
        if not isinstance(val, (tuple, list)):
            val = (val,)
        return torch.tensor(
            val,
            dtype=self.dtype,
            device=self.gen_device,
        )

    def get_distro_kwargs(self, distro, ddef, *, force_float=False):
        return {
            k: self.distro_param(
                getattr(self, f"{distro}_{k}"),
                force_float=force_float,
            )
            for k in ddef
        }

    def generate(self, *_args):
        distro = self.distro
        dfun, ddef = self.distro_params[distro]
        is_simple = distro in self.simple_distros
        dkwargs = self.get_distro_kwargs(distro, ddef, force_float=is_simple)
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
        return self.norm_quantile(noise)


class PowerOldNoiseGenerator(NoiseGenerator):
    name = "power_old"

    @classmethod
    @property
    def ng_params(cls):
        return super().ng_params | {"alpha": 2, "k": 1, "normalized": False}

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
class WaveletNoiseGenerator(NoiseGenerator):
    name = "wavelet"
    MIN_DIMS = MAX_DIMS = 4

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
    @property
    def ng_params(cls):
        return super().ng_params | {
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
        noise = (
            self.rand_like()
            if self.noise_sampler is None
            else self.noise_sampler(*args)
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
        return self.wavelet_inverse((yl, yh))


__all__ = (
    "BrownianNoiseGenerator",
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
    "scale_noise",
)
