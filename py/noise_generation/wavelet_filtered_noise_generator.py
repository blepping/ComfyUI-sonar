# ruff: noqa: ANN002, ANN003
from __future__ import annotations

import torch

from ..utils import fallback
from ..wavelet_functions import Wavelet, wavelet_blend, wavelet_scaling
from .base import FramesToChannelsNoiseGenerator

F = torch.nn.functional


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
