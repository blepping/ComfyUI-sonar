# ruff: noqa: ANN002, ANN003
from __future__ import annotations

import math

import torch

from .. import utils
from ..wavelet_functions import ptwav
from .base import FramesToChannelsNoiseGenerator

F = torch.nn.functional


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
