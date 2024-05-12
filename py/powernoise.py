from __future__ import annotations

import math
import os
import random

import folder_paths
import torch
from comfy.k_diffusion.sampling import BrownianTreeNoiseSampler
from PIL import Image
from torch import Tensor

from .nodes import SonarCustomNoiseNodeBase, SonarNormalizeNoiseNodeMixin
from .noise import CustomNoiseItemBase
from .noise_generation import scale_noise

# ruff: noqa: ANN003, FBT001, FBT002


class PowerNoiseItem(CustomNoiseItemBase):
    def __init__(self, factor, *, channel_correlation, **kwargs):
        channel_correlation = torch.tensor(
            tuple(float(val) for val in channel_correlation.split(",")),
            device="cpu",
            dtype=torch.float,
        ).clamp(0.0, 1.0)
        super().__init__(factor, **kwargs)
        self.max_freq = max(self.max_freq, self.min_freq)
        self.channel_correlation = channel_correlation

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

    def make_noise_sampler_internal(
        self,
        x: Tensor,
        noise_sampler,
        filter_rfft,
        normalized=True,
    ):
        shape = x.shape
        device = x.device
        time_brownian = self.time_brownian

        common_mode = min(self.common_mode, 1.0)
        if common_mode > 0.0:
            b, c, h, w = shape
            correlation_count = c * (c - 1) // 2
            channel_correlation = self.channel_correlation[:correlation_count]
            channel_correlation = torch.cat(
                (
                    channel_correlation * self.common_mode,
                    torch.full(
                        (correlation_count - channel_correlation.numel(),),
                        self.common_mode,
                    ),
                ),
            )
            channel_mixer = torch.eye(c)
            channel_mixer[*torch.tril_indices(c, c, offset=-1)] = channel_correlation
            channel_mixer = torch.linalg.cholesky(channel_mixer).to(
                device,
                non_blocking=True,
            )

        def sampler(sigma, sigma_next):
            noise = noise_sampler(sigma, sigma_next).to(device)
            noise_rfft = (
                torch.fft.rfft2(noise, norm="ortho") if time_brownian else noise
            )
            noise = torch.fft.irfft2(
                noise_rfft.mul_(filter_rfft),
                s=shape[-2:],
                norm="ortho",
            )

            if common_mode > 0.0:
                noise = channel_mixer @ noise.swapaxes(0, 1).reshape(c, -1)
                noise = noise.reshape(c, b, h, w).swapaxes(1, 0)
            return scale_noise(noise, self.factor, normalized=normalized)

        return sampler

    def make_noise_sampler(
        self,
        x: Tensor,
        sigma_min: float | None,
        sigma_max: float | None,
        seed: int | None,
        cpu: bool = True,
        normalized=True,
    ):
        shape, device = x.shape, x.device
        filter_rfft = self.make_filter(shape).to(device, non_blocking=True)
        if self.time_brownian:
            if sigma_min is None:
                raise ValueError(
                    "time correlated brownian mode is valid only for stochastic samplers",
                )
            noise_sampler = BrownianTreeNoiseSampler(
                x,
                sigma_min,
                sigma_max,
                seed=seed,
                cpu=cpu,
            )
        else:

            def noise_sampler(_s, _sn):
                return torch.randn(
                    (*shape[:-1], filter_rfft.shape[-1]),
                    dtype=torch.complex64,
                    device=device,
                )

        return self.make_noise_sampler_internal(
            x,
            noise_sampler,
            filter_rfft,
            normalized=normalized,
        )

    def preview(self, size=(128, 128)):
        filter_rfft = self.make_filter(size, oversample=1)
        filter_fft = rfft2_to_fft2(filter_rfft)
        noise = torch.fft.irfft2(
            filter_rfft
            * torch.randn(
                filter_rfft.shape,
                dtype=torch.complex64,
                generator=torch.Generator().manual_seed(0),
            ),
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


class PowerFilterNoiseItem(PowerNoiseItem):
    def __init__(self, factor, *, noise, normalize_noise, normalize_result, **kwargs):
        super().__init__(
            factor,
            noise=noise.clone(),
            normalize_noise=normalize_noise,
            normalize_result=normalize_result,
            **kwargs,
        )

    def clone_key(self, k):
        if k == "noise":
            return self.noise.clone()
        return super().clone_key(k)

    def make_noise_sampler(
        self,
        x: Tensor,
        sigma_min: float | None,
        sigma_max: float | None,
        seed: int | None,
        cpu: bool = True,
        normalized=True,
    ):
        shape, device = x.shape, x.device
        normalize_noise = self.get_normalize("normalize_noise", False)  # noqa: FBT003
        normalize_result = self.get_normalize("normalize_result", normalized)
        filter_rfft = self.make_filter(shape).to(device, non_blocking=True)
        noise_sampler = self.noise.make_noise_sampler(
            x,
            sigma_min,
            sigma_max,
            seed,
            cpu,
            normalized=normalize_noise,
        )

        return self.make_noise_sampler_internal(
            x,
            noise_sampler,
            filter_rfft,
            normalized=normalize_result,
        )


class SonarPowerNoiseNode(SonarCustomNoiseNodeBase):
    @classmethod
    def INPUT_TYPES(cls, *args: list, **kwargs: dict):
        result = super().INPUT_TYPES(*args, **kwargs)
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
            "channel_correlation": (
                "STRING",
                {
                    "default": "1, 1, 1, 1, 1, 1",
                    "multiline": False,
                    "dynamicPrompts": False,
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
        img = self.get_item_class()(**kwargs).preview()

        output_dir = folder_paths.get_temp_directory()
        prefix_append = "sonar_temp_" + "".join(
            random.choice("abcdefghijklmnopqrstupvxyz")  # noqa: S311
            for x in range(5)
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


class SonarPowerFilterNoiseNode(SonarPowerNoiseNode, SonarNormalizeNoiseNodeMixin):
    @classmethod
    def INPUT_TYPES(cls):
        result = super().INPUT_TYPES(include_rescale=False, include_chain=False)
        del result["required"]["time_brownian"]
        result["required"] |= {
            "sonar_custom_noise": ("SONAR_CUSTOM_NOISE",),
            "normalize_result": (("default", "forced", "disabled"),),
            "normalize_noise": (("default", "forced", "disabled"),),
        }
        return result

    def get_item_class(self):
        return PowerFilterNoiseItem

    def go(
        self,
        factor,
        sonar_custom_noise,
        normalize_noise,
        normalize_result,
        preview="none",
        **kwargs: dict,
    ):
        return super().go(
            factor=factor,
            noise=sonar_custom_noise,
            normalize_noise=self.get_normalize(normalize_noise),
            normalize_result=self.get_normalize(normalize_result),
            preview=preview,
            time_brownian=True,
            **kwargs,
        )
