from __future__ import annotations

import math
from typing import Optional

import torch
from comfy.k_diffusion.sampling import BrownianTreeNoiseSampler
from torch import Tensor

from .nodes import SonarCustomNoiseNodeBase
from .noise import CustomNoiseItemBase, scale_noise

# ruff: noqa: D412, D413, D417, D212, D407, ANN002, ANN003, FBT001, FBT002, S311


class PowerNoiseItem(CustomNoiseItemBase):
    def __init__(self, factor, **kwargs):
        super().__init__(factor, **kwargs)
        self.cached_filter = None
        self.lowpass = max(self.lowpass, self.highpass)

    @torch.no_grad()
    def make_filter(self, shape, oversample=4, rel_bw=0.25, device=None):
        """Construct a band-pass * 1/f^alpha filter in rfft space."""
        if self.cached_filter is not None and self.cached_filter.shape[-2:] == shape:
            return self.cached_filter.to(device)

        height, width = shape[-2:]
        hfreq_bins = width // 2 + 1

        # Flat unit gain frequency response
        if self.mix < 1.0:
            flat = torch.ones(1, 1, height, hfreq_bins) * (
                1.0 / math.sqrt(height * hfreq_bins)
            )
        if self.mix <= 0.0:
            flat = self.cached_filter = flat.to(device, non_blocking=True)
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
        fc.real *= self.stretch
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
        m_highpass = d > self.highpass
        m_lowpass = d < self.lowpass
        m_band = m_highpass & m_lowpass
        # 1 / f^alpha for the band-pass region
        op[m_band] = d[m_band].pow(-self.alpha)
        # easing gaussians (TODO: try cosine windows)
        m_lowpass = ~m_lowpass
        op[m_lowpass] = math.pow(self.lowpass, -self.alpha) * torch.exp(
            -(d[m_lowpass] - self.lowpass).square() / (rel_bw * self.lowpass) ** 2,
        )
        if self.highpass > 0.0:
            m_highpass = ~m_highpass
            op[m_highpass] = math.pow(self.highpass, -self.alpha) * torch.exp(
                -(d[m_highpass] - self.highpass).square()
                / (rel_bw * self.highpass) ** 2,
            )
        op = torch.nn.functional.interpolate(
            op[None, None, ...],
            (height, hfreq_bins),
            mode="bilinear",
            align_corners=True,
        )
        op = op.roll(-(height // 2), -2)  # ifftshift
        if self.alpha <= 0:
            # In general, the mean offset should be kept as is, sampled from
            # N(0, 1 / sqrt(H*W) ). However, gains goes to inf when alpha<0.
            op[..., 0, 0] = 0

        # Scale to unit power gain, then mix flat filter
        op_sq = op.square()
        op_sq_sum = op_sq.sum()
        if self.mix >= 1.0:
            if op_sq_sum > 0:
                op *= 1.0 / op_sq_sum.sqrt()
        elif op_sq_sum > 0:
            op_sq *= 1.0 / op_sq_sum
            op = torch.lerp(flat.square(), op_sq, self.mix).sqrt()
        else:
            op = flat

        op = op.to(device, non_blocking=True)
        self.cached_filter = op
        return op

    @torch.no_grad()
    def make_noise_sampler(
        self,
        x: Tensor,
        sigma_min: Optional[float],
        sigma_max: Optional[float],
        seed: Optional[int],
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
            channel_mixer = torch.lerp(
                torch.eye(c, c),
                torch.ones(c, c) / c,
                common_mode,
            )
            channel_mixer = channel_mixer.to(device)

        filter_rfft = self.make_filter(shape, device=device)

        @torch.no_grad()
        def sampler(sigma, sigma_next):
            if time_brownian:
                noise = brownian_tree(sigma, sigma_next)
                noise_rfft = torch.fft.rfft2(noise, norm="ortho")
            else:
                noise_rfft = torch.randn(
                    (*shape[:-1], filter_rfft.shape[-1]),
                    dtype=torch.complex64,
                    device=device,
                )
            noise = torch.fft.irfft2(
                filter_rfft * noise_rfft,
                s=shape[-2:],
                norm="ortho",
            )

            if common_mode > 0.0:
                noise = channel_mixer @ noise.swapaxes(0, 1).reshape(c, -1)
                noise = noise.reshape(c, b, h, w).swapaxes(1, 0)
            return noise * self.factor
            return scale_noise(noise, self.factor)

        return sampler


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
            "lowpass": (
                "FLOAT",
                {
                    "default": 0.7071,
                    "min": 0.0,
                    "max": 0.7071,
                    "step": 0.001,
                    "round": False,
                },
            ),
            "highpass": (
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
                    "min": 1.0,
                    "max": 1e3,
                    "step": 0.001,
                    "round": False,
                },
            ),
            "rotate": (
                "FLOAT",
                {
                    "default": 0,
                    "min": -90,
                    "max": 90,
                    "step": 0.001,
                    "round": False,
                },
            ),
            "pnorm": (
                "FLOAT",
                {
                    "default": 2,
                    "min": 0.125,
                    "max": 100,
                    "step": 0.001,
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
        }
        return result

    def get_item_class(self):
        return PowerNoiseItem
