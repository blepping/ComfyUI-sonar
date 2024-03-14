from __future__ import annotations

import math
import os
import random

import folder_paths
import torch
from comfy.k_diffusion.sampling import BrownianTreeNoiseSampler
from PIL import Image
from torch import Tensor

from .nodes import SonarCustomNoiseNodeBase
from .noise import CustomNoiseItemBase

# ruff: noqa: ANN003, FBT001, FBT002


class PowerNoiseItem(CustomNoiseItemBase):
    def __init__(self, factor, **kwargs):
        super().__init__(factor, **kwargs)
        self.max_freq = max(self.max_freq, self.min_freq)

    def make_filter(self, shape, oversample=4, rel_bw=0.125):
        """Construct a band-pass * 1/f^alpha filter in rfft space."""
        height, width = shape[-2:]
        hfreq_bins = width // 2 + 1

        # Flat unit gain frequency response
        if self.mix < 1.0:
            flat = torch.ones(1, 1, height, hfreq_bins)
        if self.mix <= 0.0:
            return flat

        # Start with an over-sampled fftshift(rfft2freq()) grid. uses complex
        # numbers for convenient 2d rotation (unrelated to the fft complex phase
        # space)
        fc = torch.complex(
            # real-fftfreq
            torch.linspace(0, 0.5, oversample * hfreq_bins),
            # normal fftfreq
            torch.linspace(
                -(height // 2) / height,
                ((height - 1) // 2) / height,
                oversample * height,
            ).unsqueeze(1),
        )
        # Rotate, stretch and p-norm
        if abs(self.rotate) >= 1e-3:
            fc *= torch.exp(1.0j * torch.deg2rad(torch.scalar_tensor(self.rotate)))
        if self.stretch > 1.0:
            fc.real *= self.stretch
        else:
            fc.imag *= 1.0 / self.stretch
        if abs(self.pnorm - 2.0) < 1e-3:
            d = fc.abs()
        else:
            d = (
                torch.view_as_real(fc)
                .abs()
                .pow(self.pnorm)
                .sum(-1)
                .pow(1.0 / self.pnorm)
            )

        # filter gain function
        op = torch.empty_like(d)
        m_highpass = d >= self.min_freq
        m_lowpass = d < self.max_freq
        m_band = m_highpass & m_lowpass
        # 1 / f^alpha for the band-pass region
        op[m_band] = d[m_band].pow(-self.alpha)
        # easing gaussian (TODO: try cosine windows)
        m_lowpass = ~m_lowpass
        op[m_lowpass] = math.pow(self.max_freq, -self.alpha) * torch.exp(
            -(d[m_lowpass] - self.max_freq).square() / (rel_bw * self.max_freq) ** 2,
        )
        if self.min_freq > 0.0:
            m_highpass = ~m_highpass
            op[m_highpass] = math.pow(self.min_freq, -self.alpha) * torch.exp(
                -(d[m_highpass] - self.min_freq).square()
                / (rel_bw * self.min_freq) ** 2,
            )
        op = torch.nn.functional.interpolate(
            op[None, None, ...],
            (height, hfreq_bins),
            mode="bilinear",
            align_corners=True,
        )
        op = op.roll(-(height // 2), -2)  # ifftshift
        if self.alpha > 0:
            # In general, the mean offset should be kept as is, sampled from
            # N(0, 1 / sqrt(H*W) ). However, gain goes to inf when alpha>0.
            op[..., 0, 0] = 0

        # Scale to unit power gain, then mix flat filter
        mean_pow_gain = op.mean()
        if mean_pow_gain <= 0.0:
            # don't fail catastrophically when something broke
            return flat
        op *= 1.0 / mean_pow_gain
        if self.mix < 1.0:
            op = torch.lerp(flat, op, self.mix, out=op)
        return op.sqrt_()

    def make_noise_sampler(
        self,
        x: Tensor,
        sigma_min: float | None,
        sigma_max: float | None,
        seed: int | None,
        cpu: bool = True,
    ):
        shape = x.shape
        device = x.device
        time_brownian = self.time_brownian
        if self.time_brownian:
            if sigma_min is None:
                raise ValueError(
                    "time correlated brownian mode is valid only for stochastic samplers",
                )
            brownian_tree = BrownianTreeNoiseSampler(
                x,
                sigma_min,
                sigma_max,
                seed=seed,
                cpu=cpu,
            )

        common_mode = self.common_mode
        if common_mode > 0.0:
            b, c, h, w = shape
            torch.eye(c, c)
            channel_mixer = torch.lerp(
                torch.eye(c, c),
                torch.ones(c, c) / c,
                common_mode,
            )
            channel_mixer = channel_mixer.sqrt().to(device, non_blocking=True)

        filter_rfft = self.make_filter(shape).to(device, non_blocking=True)

        def sampler(sigma, sigma_next):
            if time_brownian:
                noise = brownian_tree(sigma, sigma_next).to(device)
                noise_rfft = torch.fft.rfft2(noise, norm="ortho")
            else:
                noise_rfft = torch.randn(
                    (*shape[:-1], filter_rfft.shape[-1]),
                    dtype=torch.complex64,
                    device=device,
                )
            noise = torch.fft.irfft2(
                noise_rfft.mul_(filter_rfft),
                s=shape[-2:],
                norm="ortho",
            )

            if common_mode > 0.0:
                noise = channel_mixer @ noise.swapaxes(0, 1).reshape(c, -1)
                noise = noise.reshape(c, b, h, w).swapaxes(1, 0)
            return noise.mul_(self.factor)

        return sampler

    def preview(self, size=(128, 128)):
        filter_rfft = self.make_filter(size, oversample=1)
        filter_fft = rfft2_to_fft2(filter_rfft)
        noise = torch.fft.irfft2(
            torch.randn(filter_rfft.shape, dtype=torch.complex64) * filter_rfft,
            s=size,
            norm="ortho",
        )
        kernel = torch.fft.irfft2(filter_rfft, s=size, norm="ortho")
        kernel = kernel.roll((size[0] // 2, size[1] // 2), (-2, -1))
        img = (
            torch.cat(
                [
                    filter_fft.mul_(1 / 3).tanh_().mul_(256.0),
                    kernel.mul_(1 / 3).tanh_().add_(1.0).mul_(128.0),
                    noise.mul_(1 / 3).tanh_().add_(1.0).mul_(128.0),
                ],
                dim=-1,
            )
            .clamp(0, 255)
            .to(torch.uint8)
        )
        return Image.fromarray(img[0, 0].numpy())


def rfft2_to_fft2(x):
    """Apply hermitian-summetry to reconstruct the second half of a fft.

    Only for previews.
    """
    height, width = x.shape[-2:]
    x_r = x.roll(height // 2, -2)  # torch.fft.fftshift(x, -2)
    x_l = x_r[..., 1 : -1 if width & 1 else None]
    x_l = torch.flip(x_l.conj(), dims=(-2, -1))
    if height & 1 == 0:
        x_l = x_l.roll(1, -2)
    return torch.cat([x_l, x_r], dim=-1)


class SonarPowerNoiseNode(SonarCustomNoiseNodeBase):
    @classmethod
    def INPUT_TYPES(cls):
        result = super().INPUT_TYPES()
        result["required"] |= {
            "time_brownian": ("BOOLEAN", {"default": False}),
            "alpha": (
                "FLOAT",
                {
                    "default": 0.0,
                    "min": -5.0,
                    "max": 5.0,
                    "step": 0.001,
                    "round": False,
                },
            ),
            "max_freq": (
                "FLOAT",
                {
                    "default": 0.7071,
                    "min": 0.0,
                    "max": 0.7071,
                    "step": 0.001,
                    "round": False,
                },
            ),
            "min_freq": (
                "FLOAT",
                {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 0.7071,
                    "step": 0.001,
                    "round": False,
                },
            ),
            "stretch": (
                "FLOAT",
                {
                    "default": 1.0,
                    "min": 0.01,
                    "max": 100,
                    "step": 0.1,
                    "round": False,
                },
            ),
            "rotate": (
                "FLOAT",
                {
                    "default": 0,
                    "min": -90,
                    "max": 90,
                    "step": 5,
                    "round": False,
                },
            ),
            "pnorm": (
                "FLOAT",
                {
                    "default": 2,
                    "min": 0.125,
                    "max": 100,
                    "step": 0.1,
                    "round": False,
                },
            ),
            "mix": (
                "FLOAT",
                {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.001,
                    "round": False,
                },
            ),
            "common_mode": (
                "FLOAT",
                {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.001,
                    "round": False,
                },
            ),
            "preview": (["none", "no_mix", "mix"],),
        }
        return result

    def get_item_class(self):
        return PowerNoiseItem

    def go(
        self,
        preview="none",
        **kwargs,
    ):
        result = super().go(**kwargs)
        if preview == "none":
            return result
        if preview == "no_mix":
            kwargs["mix"] = 1.0
        img = PowerNoiseItem(**kwargs).preview()

        output_dir = folder_paths.get_temp_directory()
        prefix_append = "sonar_temp_" + "".join(
            random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5)  # noqa: S311
        )
        full_output_folder, filename, counter, subfolder, _ = (
            folder_paths.get_save_image_path(prefix_append, output_dir)
        )
        filename = f"{filename}_{counter:05}_.png"
        file_path = os.path.join(full_output_folder, filename)  # noqa: PTH118
        img.save(file_path, compress_level=1)

        return {
            "ui": {
                "images": [
                    {"filename": filename, "subfolder": subfolder, "type": "temp"},
                ],
            },
            "result": result,
        }
