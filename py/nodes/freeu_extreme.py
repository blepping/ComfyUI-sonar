from __future__ import annotations

import torch

from .. import utils
from .base import SonarInputTypes, SonarLazyInputTypes
from .powernoise import PowerFilter


def ffilter(x, pfilter, normalization_factor=1.0, cfg_idx=None, filter_cache=None):
    cache_key = None
    if filter_cache is not None and cfg_idx is not None:
        cache_key = (cfg_idx, x.shape[-2:])
        filter_rfft = filter_cache.get(cache_key)
    if filter_rfft is None:
        filter_rfft = PowerFilter.normalize(
            pfilter.build(x.shape),
            x.shape,
            normalization_factor=normalization_factor,
        ).to(x.device, non_blocking=True)
    if cache_key:
        filter_cache[cache_key] = filter_rfft
    x_rfft = torch.fft.rfft2(x.to(torch.float32), norm="ortho")
    x_filt = torch.fft.irfft2(
        x_rfft.mul_(filter_rfft),
        s=x.shape[-2:],
        norm="ortho",
    )
    return x_filt.to(x.dtype, non_blocking=True)


class FreeUExtremeConfigNode:
    DESCRIPTION = "Allows setting configuration for FreeU Extreme."
    RETURN_TYPES = ("FRUX_CONFIG",)
    FUNCTION = "go"
    CATEGORY = "model_patches"

    INPUT_TYPES = SonarLazyInputTypes(
        lambda: SonarInputTypes()
        .req_bool_stage_1(
            default=True,
            tooltip="Controls whether this configuration applies to stage 1.",
        )
        .req_bool_stage_2(
            default=False,
            tooltip="Controls whether this configuration applies to stage 2.",
        )
        .req_bool_stage_3(
            default=False,
            tooltip="Controls whether this configuration applies to stage 3.",
        )
        .req_field_target(
            ("backbone", "skip", "both"),
            default="backbone",
            tooltip="Controls whether this filter applies to backbone or skip layers (or both).",
        )
        .req_floatpct_start(
            default=0.0,
            tooltip="Start time as percentage of sampling this configuration applies to. Inclusive.",
        )
        .req_floatpct_end(
            default=1.0,
            tooltip="End time as percentage of sampling this configuration applies to. Inclusive.",
        )
        .req_floatpct_slice(
            default=1.0,
            tooltip="Percentage of the layer the FreeU effect is applied to.",
        )
        .req_floatpct_slice_offset(
            default=0.0,
            tooltip="Offset as a percentage the layer is applied to. For example if slice is 0.25 and slice_offset is 0.25 then the filter will apply to the range 25% through 50%.",
        )
        .req_float_filter_norm(
            default=0.0,
            min=-10.0,
            max=10.0,
            tooltip="Normalization factor applied to the filter. 1.0 means 100% normalized.",
        )
        .req_float_scale(
            default=1.0,
            tooltip="Strength of the effects applied by this configuration.",
        )
        .req_float_blend(
            default=1.0,
            tooltip="Blends the filtered result based on the specified strength where 1.0 means 100% filtered.",
        )
        .req_selectblend_blend_mode(
            tooltip="Mode used when blending. Generally only has an effect when blend is set to values other than 0 or 1",
        )
        .req_bool_hidden_mean(
            default=True,
            tooltip="You can think of this as FreeU V2 mode.",
        )
        .req_bool_final(
            default=True,
            tooltip="When enabled, other configurations won't be considered if this one matched. Otherwise, multiple configurations/filter effects can be stacked.",
        )
        .opt_field_sonar_power_filter_opt(
            "SONAR_POWER_FILTER",
            tooltip="Optionally attach a Power Filter here to set filtering parameters.",
        )
        .opt_field_frux_config_opt(
            "FRUX_CONFIG",
            tooltip="Optionally attach another configuration node here.",
        ),
    )

    @classmethod
    def go(cls, **kwargs: dict):
        return (FreeUExtremeConfig(**kwargs),)


class FreeUExtremeConfig:
    _keys = (
        "target",
        "stage_1",
        "stage_2",
        "stage_3",
        "start",
        "end",
        "slice",
        "slice_offset",
        "filter_norm",
        "scale",
        "blend",
        "blend_mode",
        "hidden_mean",
        "final",
        "sonar_power_filter",
        "frux_config",
    )

    def __init__(
        self,
        *,
        target,
        stage_1=False,
        stage_2=False,
        stage_3=False,
        start=0.0,
        end=1.0,
        slice=1.0,  # noqa: A002
        slice_offset=0.0,
        filter_norm=1.0,
        scale=1.0,
        blend=1.0,
        blend_mode=None,
        hidden_mean=True,
        final=True,
        sonar_power_filter_opt=None,
        frux_config_opt=None,
    ):
        self.target = target
        self.stage_1 = stage_1
        self.stage_2 = stage_2
        self.stage_3 = stage_3
        self.start = start
        self.end = end
        self.slice = slice
        self.slice_offset = slice_offset
        self.filter_norm = filter_norm
        self.scale = scale
        self.blend = blend
        self.blend_mode = blend_mode
        self.hidden_mean = hidden_mean
        self.final = final
        self.sonar_power_filter = sonar_power_filter_opt
        self.frux_config = frux_config_opt

    def get_config_list(self):
        result = [self]
        curr = self
        while cfg := curr.frux_config:
            curr = cfg
            if (
                cfg.start >= 1
                or cfg.end <= 0
                or cfg.blend == 0
                or not (cfg.stage_1 or cfg.stage_2 or cfg.stage_3)
            ):
                continue
            result.append(cfg)
        result.reverse()
        return result

    # Hidden mean function modified from https://github.com/WASasquatch/FreeU_Advanced
    def get_scale(self, h: torch.Tensor) -> torch.Tensor:
        if not self.hidden_mean:
            return self.scale
        hmean = h.mean(1).unsqueeze(1)
        hmax, hmin = (
            op(hmean.view(hmean.shape[0], -1), dim=-1, keepdim=True)[0]
            for op in (torch.max, torch.min)
        )
        hmean -= hmin.unsqueeze(2).unsqueeze(3)
        hmean /= (hmax - hmin).unsqueeze(2).unsqueeze(3)
        return 1.0 + (self.scale - 1.0) * hmean

    def check_match(self, pct, stage, is_skip=False):
        if pct < self.start or pct > self.end:
            return False
        if not getattr(self, f"stage_{stage}"):
            return False
        return not self.target not in {"skip" if is_skip else "backbone", "both"}

    def apply(self, idx, x, filter_cache, cpu_fft=False):
        _batch, features, _height, _width = x.shape
        scale = self.get_scale(x)
        slice_size = int(features * self.slice)
        slice_offs = int(features * self.slice_offset)

        xslice = (
            self.apply_filter(
                idx,
                x[:, slice_offs : slice_offs + slice_size],
                filter_cache,
                cpu_fft=cpu_fft,
            )
            * scale
        )
        x[:, slice_offs : slice_offs + slice_size] = (
            xslice
            if self.blend == 1.0
            else utils.BLENDING_MODES[self.blend_mode](
                x[:, slice_offs : slice_offs + slice_size],
                xslice,
                self.blend,
            )
        )
        return x

    def apply_filter(self, idx, xslice, filter_cache, cpu_fft=False):
        filt = self.sonar_power_filter
        if filt is None:
            return xslice
        device = xslice.device
        if cpu_fft:
            xslice = xslice.to("cpu")
        xslice = ffilter(
            xslice,
            filt,
            normalization_factor=self.filter_norm,
            cfg_idx=idx,
            filter_cache=filter_cache,
        )
        if cpu_fft:
            xslice = xslice.to(device)
        return xslice

    def clone(self):
        return self.__class__(**{k: getattr(self, k) for k in self._keys})

    def __repr__(self):
        meh = {k: getattr(self, k) for k in self._keys}
        return f"<FRUXConfig: {meh}>"


class FreeUExtremeNode:
    DESCRIPTION = "Main FreeU Extreme node. Allows patching a model with the FreeU (V2) effect with more control."
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "go"
    CATEGORY = "model_patches"

    INPUT_TYPES = (
        SonarInputTypes()
        .req_model(tooltip="Model to patch.")
        .req_bool_cpu_fft(
            tooltip="Controls whether to perform FFT calculations on the CPU. May be necessary for some GPUs that don't have native support for FFT )operations at the cost of performance.",
        )
        .opt_field_input_config(
            "FRUX_CONFIG",
            tooltip="Allows specifying configuration for input blocks.",
        )
        .opt_field_middle_config(
            "FRUX_CONFIG",
            tooltip="Allows specifying configuration for middle blocks.",
        )
        .opt_field_output_config(
            "FRUX_CONFIG",
            tooltip="Allows specifying configuration for output blocks.",
        )
    )

    @classmethod
    def go(
        cls,
        model,
        cpu_fft,
        input_config=None,
        middle_config=None,
        output_config=None,
    ):
        model_channels = model.model.model_config.unet_config["model_channels"]
        stages = {model_channels * 4: 1, model_channels * 2: 2, model_channels: 3}
        icfg, mcfg, ocfg = (
            () if cfg is None else cfg.get_config_list()
            for cfg in (input_config, middle_config, output_config)
        )
        m = model.clone()
        ms = m.get_model_object("model_sampling")
        filter_cache = {}

        def handler(_typ, h_shape, cfg, x, toptions, is_skip=False):
            stage = stages.get(h_shape[1])
            if stage is None:
                return x
            sigma = toptions["sigmas"].max().detach().cpu()
            pct = 1.0 - (ms.timestep(sigma) / 999.0)
            for idx, ci in enumerate(cfg):
                if not ci.check_match(pct, stage, is_skip):
                    continue
                x = ci.apply(idx, x, filter_cache, cpu_fft=cpu_fft)
                if ci.final:
                    break
            return x

        def in_patch(h, toptions):
            return handler("input", h.shape, icfg, h, toptions)

        def mid_patch(h, toptions):
            return handler("middle", h.shape, mcfg, h, toptions)

        def out_patch(h, hsp, toptions):
            h = handler("output", h.shape, ocfg, h, toptions)
            hsp = handler("output", h.shape, ocfg, hsp, toptions, is_skip=True)
            return h, hsp

        if icfg:
            m.set_model_input_block_patch(in_patch)
        if mcfg:
            m.set_model_patch(mid_patch, "middle_block_patch")
        if ocfg:
            m.set_model_output_block_patch(out_patch)
        return (m,)


NODE_CLASS_MAPPINGS = {
    "FreeUExtremeConfig": FreeUExtremeConfigNode,
    "FreeUExtreme": FreeUExtremeNode,
}
