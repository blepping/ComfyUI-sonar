from __future__ import annotations

import torch
from comfy import samplers

from . import noise
from .sonar import (
    GuidanceConfig,
    GuidanceType,
    HistoryType,
    SonarConfig,
    SonarDPMPPSDE,
    SonarEuler,
    SonarEulerAncestral,
)


class NoisyLatentLikeNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise_type": (
                    tuple(
                        t.name.lower()
                        for t in noise.NoiseType
                        if t is not noise.NoiseType.BROWNIAN
                    ),
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "latent": ("LATENT",),
            },
            "optional": {
                "custom_noise_opt": ("SONAR_CUSTOM_NOISE",),
            },
        }

    RETURN_TYPES = ("LATENT",)
    CATEGORY = "latent/noise"

    FUNCTION = "go"

    def go(
        self,
        noise_type,
        seed,
        latent,
        custom_noise_opt=None,
    ):
        if custom_noise_opt is not None:
            ns = custom_noise_opt.make_noise_sampler(latent["samples"])
        else:
            ns = noise.get_noise_sampler(
                noise.NoiseType[noise_type.upper()],
                latent["samples"],
                None,
                None,
                seed=None,
                use_cpu=True,
            )
        randst = torch.random.get_rng_state()
        try:
            torch.random.manual_seed(seed)
            result = ns(None, None)
        finally:
            torch.random.set_rng_state(randst)
        return ({"samples": result},)


class SonarCustomNoiseNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "factor": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -100.0,
                        "max": 100.0,
                        "step": 0.001,
                        "round": False,
                    },
                ),
                "rescale": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.001,
                        "round": False,
                    },
                ),
                "noise_type": (
                    tuple(
                        t.name.lower()
                        for t in noise.NoiseType
                        if t is not noise.NoiseType.BROWNIAN
                    ),
                ),
            },
            "optional": {
                "sonar_custom_noise_opt": ("SONAR_CUSTOM_NOISE",),
            },
        }

    RETURN_TYPES = ("SONAR_CUSTOM_NOISE",)
    CATEGORY = "advanced/noise"
    FUNCTION = "go"

    def go(self, factor, rescale, noise_type, sonar_custom_noise_opt=None):
        nis = (
            sonar_custom_noise_opt.clone()
            if sonar_custom_noise_opt
            else noise.CustomNoise()
        )
        if factor != 0:
            nis.add(noise.CustomNoiseItem(factor, noise_type))
        return (nis if rescale == 0 else nis.rescaled(rescale),)


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


class SamplerNodeSonarBase:
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
                "rand_init_noise_type": (
                    tuple(
                        t.name.lower()
                        for t in noise.NoiseType
                        if t is not noise.NoiseType.BROWNIAN
                    ),
                ),
            },
            "optional": {
                "guidance_cfg_opt": ("SONAR_GUIDANCE_CFG",),
            },
        }

    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"


class SamplerNodeSonarEuler(SamplerNodeSonarBase):
    @classmethod
    def INPUT_TYPES(cls):
        result = super().INPUT_TYPES()
        result["required"].update(
            {
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
        )
        return result

    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(
        self,
        momentum,
        momentum_hist,
        momentum_init,
        direction,
        rand_init_noise_type,
        s_noise,
        guidance_cfg_opt=None,
    ):
        cfg = SonarConfig(
            momentum=momentum,
            init=HistoryType[momentum_init.upper()],
            momentum_hist=momentum_hist,
            direction=direction,
            rand_init_noise_type=noise.NoiseType[rand_init_noise_type.upper()],
            guidance=guidance_cfg_opt,
        )
        return (
            samplers.KSAMPLER(
                SonarEuler.sampler,
                {
                    "s_noise": s_noise,
                    "sonar_config": cfg,
                },
            ),
        )


class SamplerNodeSonarEulerAncestral(SamplerNodeSonarEuler):
    @classmethod
    def INPUT_TYPES(cls):
        result = super().INPUT_TYPES()
        result["required"].update(
            {
                "eta": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.01,
                        "round": False,
                    },
                ),
                "noise_type": (tuple(t.name.lower() for t in noise.NoiseType),),
            },
        )
        result["optional"].update(
            {
                "custom_noise_opt": ("SONAR_CUSTOM_NOISE",),
            },
        )
        return result

    def get_sampler(
        self,
        momentum,
        momentum_hist,
        momentum_init,
        direction,
        rand_init_noise_type,
        noise_type,
        eta,
        s_noise,
        guidance_cfg_opt=None,
        custom_noise_opt=None,
    ):
        cfg = SonarConfig(
            momentum=momentum,
            init=HistoryType[momentum_init.upper()],
            momentum_hist=momentum_hist,
            direction=direction,
            rand_init_noise_type=noise.NoiseType[rand_init_noise_type.upper()],
            noise_type=noise.NoiseType[noise_type.upper()],
            custom_noise=custom_noise_opt.clone() if custom_noise_opt else None,
            guidance=guidance_cfg_opt,
        )
        return (
            samplers.KSAMPLER(
                SonarEulerAncestral.sampler,
                {
                    "sonar_config": cfg,
                    "eta": eta,
                    "s_noise": s_noise,
                },
            ),
        )


class SamplerNodeSonarDPMPPSDE(SamplerNodeSonarEuler):
    @classmethod
    def INPUT_TYPES(cls):
        result = super().INPUT_TYPES()
        result["required"].update(
            {
                "eta": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.01,
                        "round": False,
                    },
                ),
                "noise_type": (tuple(t.name.lower() for t in noise.NoiseType),),
            },
        )
        result["optional"].update(
            {
                "custom_noise_opt": ("SONAR_CUSTOM_NOISE",),
            },
        )
        return result

    def get_sampler(
        self,
        momentum,
        momentum_hist,
        momentum_init,
        direction,
        rand_init_noise_type,
        noise_type,
        eta,
        s_noise,
        guidance_cfg_opt=None,
        custom_noise_opt=None,
    ):
        cfg = SonarConfig(
            momentum=momentum,
            init=HistoryType[momentum_init.upper()],
            momentum_hist=momentum_hist,
            direction=direction,
            rand_init_noise_type=noise.NoiseType[rand_init_noise_type.upper()],
            noise_type=noise.NoiseType[noise_type.upper()],
            custom_noise=custom_noise_opt.clone() if custom_noise_opt else None,
            guidance=guidance_cfg_opt,
        )
        return (
            samplers.KSAMPLER(
                SonarDPMPPSDE.sampler,
                {
                    "sonar_config": cfg,
                    "eta": eta,
                    "s_noise": s_noise,
                },
            ),
        )
