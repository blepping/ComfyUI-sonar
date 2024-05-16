from __future__ import annotations

import torch

from .external import MODULES as EXTERNAL_MODULES
from .powernoise import PowerFilter


def ffilter(x, pfilter, normalization_factor=1.0):
    filter_rfft = PowerFilter.normalize(
        pfilter.build(x.shape),
        x.shape,
        normalization_factor=normalization_factor,
    ).to(device="cpu")
    x_rfft = torch.fft.rfft2(x.to(torch.float32), norm="ortho").to(device="cpu")
    x_filt = torch.fft.irfft2(
        x_rfft.mul_(filter_rfft),
        s=x.shape[-2:],
        norm="ortho",
    )
    return x_filt.to(x.device, x.dtype)


class FreeUExtremeConfigNode:
    RETURN_TYPES = ("FRUX_CONFIG",)
    FUNCTION = "go"
    CATEGORY = "model_patches"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stage_1": ("BOOLEAN", {"default": True}),
                "stage_2": ("BOOLEAN", {"default": False}),
                "stage_3": ("BOOLEAN", {"default": False}),
                "target": (("backbone", "skip", "both"),),
                "start": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "round": False,
                    },
                ),
                "end": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "round": False,
                    },
                ),
                "slice": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "round": False,
                    },
                ),
                "slice_offset": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "round": False,
                    },
                ),
                "filter_norm": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "round": False,
                    },
                ),
                "scale": (
                    "FLOAT",
                    {
                        "default": 1,
                        "min": -100.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": False,
                    },
                ),
                "blend": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -10.0,
                        "max": 10.0,
                        "step": 0.1,
                        "round": False,
                    },
                ),
                "blend_mode": (
                    ("lerp",)
                    if "bleh" not in EXTERNAL_MODULES
                    else tuple(
                        EXTERNAL_MODULES["bleh"].py.latent_utils.BLENDING_MODES.keys(),
                    ),
                ),
                "hidden_mean": ("BOOLEAN", {"default": True}),
                "final": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "sonar_power_filter": ("SONAR_POWER_FILTER",),
                "frux_config": ("FRUX_CONFIG",),
            },
        }

    def go(self, **kwargs: dict):
        return (FreeUExtremeConfig(**kwargs),)


class FreeUExtremeConfig:
    def __init__(self, **kwargs: dict):
        self._keys = tuple(kwargs.keys())
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get_config_list(self):
        result = [self]
        curr = self
        while cfg := getattr(curr, "frux_config", None):
            curr = cfg
            if (
                cfg.start >= 1
                or cfg.end <= 0
                or not (cfg.stage_1 or cfg.stage_2 or cfg.stage_3)
            ):
                continue
            result.append(cfg)
        result.reverse()
        return result

    def __str__(self):  # noqa: D105
        meh = {k: getattr(self, k) for k in self._keys}
        return f"<FRUXConfig: {meh}>"


class FreeUExtremeNode:
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "go"
    CATEGORY = "model_patches"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"model": ("MODEL",)},
            "optional": {
                "input_config": ("FRUX_CONFIG",),
                "middle_config": ("FRUX_CONFIG",),
                "output_config": ("FRUX_CONFIG",),
            },
        }

    def go(self, model, input_config=None, middle_config=None, output_config=None):
        model_channels = model.model.model_config.unet_config["model_channels"]
        stages = {model_channels * 4: 1, model_channels * 2: 2, model_channels: 3}
        icfg = [] if input_config is None else input_config.get_config_list()
        mcfg = [] if middle_config is None else middle_config.get_config_list()
        ocfg = [] if output_config is None else output_config.get_config_list()
        m = model.clone()
        ms = m.get_model_object("model_sampling")
        blend_ops = (
            {"lerp": torch.lerp}
            if "bleh" not in EXTERNAL_MODULES
            else EXTERNAL_MODULES["bleh"].py.latent_utils.BLENDING_MODES
        )

        # Hidden mean function modified from https://github.com/WASasquatch/FreeU_Advanced
        def hidden_mean(h):
            hmean = h.mean(1).unsqueeze(1)
            hmax, hmin = (
                op(hmean.view(hmean.shape[0], -1), dim=-1, keepdim=True)[0]
                for op in (torch.max, torch.min)
            )
            hmean -= hmin.unsqueeze(2).unsqueeze(3)
            hmean /= (hmax - hmin).unsqueeze(2).unsqueeze(3)
            return hmean

        def handler(_typ, stage, cfg, x, toptions, is_skip=False):
            batch, features, height, width = x.shape
            sigma = toptions["sigmas"].max().detach().cpu()
            pct = 1.0 - (ms.timestep(sigma) / 999.0)

            if stage is None:
                return x
            for ci in cfg:
                if pct < ci.start or pct > ci.end:
                    continue
                if not getattr(ci, f"stage_{stage}"):
                    continue
                if ci.target not in ("skip" if is_skip else "backbone", "both"):
                    continue
                scale = ci.scale
                if ci.hidden_mean:
                    scale = 1.0 + (scale - 1.0) * hidden_mean(x)
                slice_size = int(features * ci.slice)
                slice_offs = int(features * ci.slice_offset)

                xslice = x[:, slice_offs : slice_offs + slice_size]
                filt = getattr(ci, "sonar_power_filter", None)
                if filt is not None:
                    xslice = ffilter(xslice, filt, normalization_factor=ci.filter_norm)
                xslice = xslice * scale
                x[:, slice_offs : slice_offs + slice_size] = blend_ops[ci.blend_mode](
                    x[:, slice_offs : slice_offs + slice_size],
                    xslice,
                    ci.blend,
                )
                if ci.final:
                    break
            return x

        def in_patch(h, toptions):
            stage = stages.get(h.shape[1])
            return handler("input", stage, icfg, h, toptions)

        def mid_patch(h, toptions):
            stage = stages.get(h.shape[1])
            return handler("middle", stage, mcfg, h, toptions)

        def out_patch(h, hsp, toptions):
            stage = stages.get(h.shape[1])
            h = handler("output", stage, ocfg, h, toptions)
            hsp = handler("output", stage, ocfg, hsp, toptions, is_skip=True)
            return h, hsp

        if icfg:
            m.set_model_input_block_patch(in_patch)
        if mcfg:
            m.set_model_patch(mid_patch, "middle_block_patch")
        if ocfg:
            m.set_model_output_block_patch(out_patch)
        return (m,)
