# Some noise generation functions shamelessly yoinked from https://github.com/Clybius/ComfyUI-Extra-Samplers
from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch

from .. import utils
from .base import FramesToChannelsNoiseGenerator

if TYPE_CHECKING:
    from collections.abc import Sequence

# ruff: noqa: ANN002, ANN003


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
