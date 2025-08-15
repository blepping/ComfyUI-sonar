from __future__ import annotations

import math
from typing import Callable

import torch
from comfy.k_diffusion import sampling
from torch import FloatTensor, Generator, Tensor
from torch.distributions import Laplace, StudentT

from .. import utils
from ..utils import tensor_to

# ruff: noqa: D413, D417, D212, ANN002, ANN003
from .base import FramesToChannelsNoiseGenerator, NoiseError, NoiseGenerator


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
