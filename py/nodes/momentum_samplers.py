# ruff: noqa: TID252

from __future__ import annotations

from comfy import samplers

from ..external import IntegratedNode
from ..noise import NoiseType
from ..sonar import (
    GuidanceConfig,
    GuidanceType,
    HistoryType,
    SonarConfig,
    SonarDPMPPSDE,
    SonarEuler,
    SonarEulerAncestral,
)
from .base import NOISE_INPUT_TYPES_HINT, WILDCARD_NOISE


class GuidanceConfigNode:
    DESCRIPTION = "Allows specifying extended guidance parameters for Sonar samplers."

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
                        "tooltip": "Controls the strength of the guidance. You'll generally want to use fairly low values here.",
                    },
                ),
                "guidance_type": (
                    tuple(t.name.lower() for t in GuidanceType),
                    {
                        "tooltip": "Method to use when calculating guidance. When set to linear, will simply LERP the guidance at the specified strength. When set to Euler, will do a Euler step toward the guidance instead.",
                    },
                ),
                "start_step": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "tooltip": "First zero-based step the guidance is active.",
                    },
                ),
                "end_step": (
                    "INT",
                    {
                        "default": 9999,
                        "min": 0,
                        "tooltip": "Last zero-based step the guidance is active.",
                    },
                ),
                "latent": (
                    "LATENT",
                    {"tooltip": "Latent to use as a reference for guidance."},
                ),
            },
        }

    RETURN_TYPES = ("SONAR_GUIDANCE_CFG",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "make_guidance_cfg"

    @classmethod
    def make_guidance_cfg(
        cls,
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


class SamplerNodeSonarBase(metaclass=IntegratedNode):
    DESCRIPTION = "Sonar - momentum based sampler node."

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
                        "tooltip": "How much of the normal result to keep during sampling. 0.95 means 95% normal, 5% from history. When set to 1.0 effectively disables momentum.",
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
                        "tooltip": "How much of the existing history to leave at each update. 0.75 means keep 75%, mix in 25% of the new result.",
                    },
                ),
                "momentum_init": (
                    tuple(t.name for t in HistoryType),
                    {
                        "tooltip": "Initial value used for momentum history. ZERO - history starts zeroed out. RAND - History is initialized with a random value. SAMPLE - History is initialized from the latent at the start of sampling.",
                    },
                ),
                "direction": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -30.0,
                        "max": 15.0,
                        "step": 0.01,
                        "round": False,
                        "tooltip": "Multiplier applied to the result of normal sampling.",
                    },
                ),
                "rand_init_noise_type": (
                    tuple(NoiseType.get_names(skip=(NoiseType.BROWNIAN,))),
                    {
                        "tooltip": "Noise type to use when momentum_init is set to RANDOM.",
                    },
                ),
            },
            "optional": {
                "guidance_cfg_opt": (
                    "SONAR_GUIDANCE_CFG",
                    {
                        "tooltip": "Optional input for extended guidance parameters.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"


class SamplerNodeSonarEuler(SamplerNodeSonarBase):
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    @classmethod
    def get_sampler(
        cls,
        *,
        momentum,
        momentum_hist,
        momentum_init,
        direction,
        rand_init_noise_type,
        guidance_cfg_opt=None,
    ):
        cfg = SonarConfig(
            momentum=momentum,
            init=HistoryType[momentum_init.upper()],
            momentum_hist=momentum_hist,
            direction=direction,
            rand_init_noise_type=NoiseType[rand_init_noise_type.upper()],
            guidance=guidance_cfg_opt,
        )
        return (samplers.KSAMPLER(SonarEuler.sampler, {"sonar_config": cfg}),)


class SamplerNodeSonarEulerAncestral(SamplerNodeSonarEuler):
    @classmethod
    def INPUT_TYPES(cls):
        result = super().INPUT_TYPES()
        result["required"].update(
            {
                "s_noise": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -1000.0,
                        "max": 1000.0,
                        "step": 0.01,
                        "round": False,
                        "tooltip": "Multiplier for noise added during ancestral or SDE sampling.",
                    },
                ),
                "eta": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -1000.0,
                        "max": 1000.0,
                        "step": 0.01,
                        "round": False,
                        "tooltip": "Basically controls the ancestralness of the sampler. When set to 0, you will get a non-ancestral (or SDE) sampler.",
                    },
                ),
                "noise_type": (
                    tuple(NoiseType.get_names()),
                    {
                        "tooltip": "Noise type used during ancestral or SDE sampling. Only used when the custom noise input is not connected.",
                    },
                ),
            },
        )
        result["optional"].update(
            {
                "custom_noise_opt": (
                    WILDCARD_NOISE,
                    {
                        "tooltip": f"Optional input for custom noise used during ancestral or SDE sampling. When connected, the built-in noise_type selector is ignored.\n{NOISE_INPUT_TYPES_HINT}",
                    },
                ),
            },
        )
        return result

    @classmethod
    def get_sampler(
        cls,
        *,
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
            rand_init_noise_type=NoiseType[rand_init_noise_type.upper()],
            noise_type=NoiseType[noise_type.upper()],
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
                "s_noise": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -1000.0,
                        "max": 1000.0,
                        "step": 0.01,
                        "round": False,
                        "tooltip": "Multiplier for noise added during ancestral or SDE sampling.",
                    },
                ),
                "eta": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -1000.0,
                        "max": 1000.0,
                        "step": 0.01,
                        "round": False,
                        "tooltip": "Basically controls the ancestralness of the sampler. When set to 0, you will get a non-ancestral (or SDE) sampler.",
                    },
                ),
                "noise_type": (
                    tuple(NoiseType.get_names(default=NoiseType.BROWNIAN)),
                    {
                        "tooltip": "Noise type used during ancestral or SDE sampling. Only used when the custom noise input is not connected.",
                    },
                ),
            },
        )
        result["optional"].update(
            {
                "custom_noise_opt": (
                    WILDCARD_NOISE,
                    {
                        "tooltip": f"Optional input for custom noise used during ancestral or SDE sampling. When connected, the built-in noise_type selector is ignored.\n{NOISE_INPUT_TYPES_HINT}",
                    },
                ),
            },
        )
        return result

    @classmethod
    def get_sampler(
        cls,
        *,
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
            rand_init_noise_type=NoiseType[rand_init_noise_type.upper()],
            noise_type=NoiseType[noise_type.upper()],
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


NODE_CLASS_MAPPINGS = {
    "SamplerSonarEuler": SamplerNodeSonarEuler,
    "SamplerSonarEulerA": SamplerNodeSonarEulerAncestral,
    "SamplerSonarDPMPPSDE": SamplerNodeSonarDPMPPSDE,
    "SonarGuidanceConfig": GuidanceConfigNode,
}
