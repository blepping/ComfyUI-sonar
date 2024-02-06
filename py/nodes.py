from __future__ import annotations

from comfy import samplers

from . import noise
from .sonar import (
    GuidanceConfig,
    GuidanceType,
    HistoryType,
    SonarEuler,
    SonarEulerAncestral,
    SonarNaive,
)


class GuidanceConfigNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "factor": (
                    "FLOAT",
                    {
                        "default": 0.01,
                        "min": -2.0,
                        "max": 2.0,
                        "step": 0.001,
                        "round": False,
                    },
                ),
                "guidance_type": (tuple(t.name.lower() for t in GuidanceType),),
                "start_step": ("INT", {"default": 1, "min": 1}),
                "end_step": ("INT", {"default": 9999, "min": 1}),
                "latent": ("LATENT",),
            },
        }

    RETURN_TYPES = ("SONAR_GUIDANCE_CFG",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "make_guidance_cfg"

    def make_guidance_cfg(
        self,
        guidance_type,
        factor,
        start_step,
        end_step,
        latent,
    ):
        return (
            GuidanceConfig(
                guidance_type=GuidanceType[guidance_type.upper()],
                factor=factor,
                start_step=start_step,
                end_step=end_step,
                latent=latent.get("samples"),
            ),
        )


class SamplerNodeSonarEuler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "momentum": (
                    "FLOAT",
                    {
                        "default": 0.95,
                        "min": -0.5,
                        "max": 2.5,
                        "step": 0.01,
                        "round": False,
                    },
                ),
                "momentum_hist": (
                    "FLOAT",
                    {
                        "default": 0.75,
                        "min": -1.5,
                        "max": 1.5,
                        "step": 0.01,
                        "round": False,
                    },
                ),
                "momentum_init": (tuple(t.name for t in HistoryType),),
                "direction": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -30.0,
                        "max": 15.0,
                        "step": 0.01,
                        "round": False,
                    },
                ),
                "s_noise": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.01,
                        "round": False,
                    },
                ),
            },
        }

    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, momentum, momentum_hist, momentum_init, direction, s_noise):
        return (
            samplers.KSAMPLER(
                SonarEuler.sampler,
                {
                    "momentum_init": HistoryType[momentum_init],
                    "momentum": momentum,
                    "momentum_hist": momentum_hist,
                    "direction": direction,
                    "s_noise": s_noise,
                },
            ),
        )


class SamplerNodeSonarEulerAncestral(SamplerNodeSonarEuler):
    @classmethod
    def INPUT_TYPES(cls):
        result = super().INPUT_TYPES()
        result["required"]["eta"] = (
            "FLOAT",
            {
                "default": 1.0,
                "min": 0.0,
                "max": 100.0,
                "step": 0.01,
                "round": False,
            },
        )
        result["required"]["noise_type"] = (
            tuple(t.name.lower() for t in noise.NoiseType),
        )
        return result

    def get_sampler(
        self,
        momentum,
        momentum_hist,
        momentum_init,
        direction,
        noise_type,
        eta,
        s_noise,
    ):
        return (
            samplers.KSAMPLER(
                SonarEulerAncestral.sampler,
                {
                    "momentum_init": HistoryType[momentum_init],
                    "momentum": momentum,
                    "momentum_hist": momentum_hist,
                    "direction": direction,
                    "noise_type": noise_type,
                    "eta": eta,
                    "s_noise": s_noise,
                },
            ),
        )


class SamplerNodeSonarNaive(SamplerNodeSonarEuler):
    @classmethod
    def INPUT_TYPES(cls):
        result = super().INPUT_TYPES()
        result["required"].update(
            {
                "noise_type": (tuple(t.name.lower() for t in noise.NoiseType),),
            },
        )
        result["optional"] = {"guidance_cfg_opt": ("SONAR_GUIDANCE_CFG",)}
        return result

    def get_sampler(
        self,
        momentum,
        momentum_hist,
        momentum_init,
        direction,
        noise_type,
        s_noise,
        guidance_cfg_opt=None,
    ):
        return (
            samplers.KSAMPLER(
                SonarNaive.sampler,
                {
                    "momentum_init": HistoryType[momentum_init],
                    "momentum": momentum,
                    "momentum_hist": momentum_hist,
                    "direction": direction,
                    "noise_type": noise_type,
                    "s_noise": s_noise,
                    "guidance": guidance_cfg_opt,
                },
            ),
        )
