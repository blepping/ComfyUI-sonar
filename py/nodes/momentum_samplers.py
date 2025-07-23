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
from .base import SonarInputTypes, SonarLazyInputTypes


class GuidanceConfigNode(metaclass=IntegratedNode):
    DESCRIPTION = "Allows specifying extended guidance parameters for Sonar samplers."

    INPUT_TYPES = SonarLazyInputTypes(
        lambda: SonarInputTypes()
        .req_float_factor(
            default=0.01,
            min=-2.0,
            max=2.0,
            tooltip="Controls the strength of the guidance. You'll generally want to use fairly low values here.",
        )
        .req_field_guidance_type(
            tuple(t.name.lower() for t in GuidanceType),
            default="linear",
            tooltip="Method to use when calculating guidance. When set to linear, will simply LERP the guidance at the specified strength. When set to Euler, will do a Euler step toward the guidance instead.",
        )
        .req_int_start_step(
            default=0,
            min=0,
            tooltip="First zero-based step the guidance is active.",
        )
        .req_int_end_step(
            default=9999,
            min=0,
            tooltip="Last zero-based step the guidance is active.",
        )
        .req_latent(tooltip="Latent to use as a reference for guidance."),
    )

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


class SamplerNodeSonarBase:
    DESCRIPTION = "Sonar - momentum based sampler node."

    INPUT_TYPES = SonarLazyInputTypes(
        lambda: SonarInputTypes()
        .req_float_momentum(
            default=0.95,
            min=-0.5,
            max=2.5,
            tooltip="How much of the normal result to keep during sampling. 0.95 means 95% normal, 5% from history. When set to 1.0 effectively disables momentum.",
        )
        .req_float_momentum_hist(
            default=0.75,
            min=-1.5,
            max=1.5,
            tooltip="How much of the existing history to leave at each update. 0.75 means keep 75%, mix in 25% of the new result.",
        )
        .req_field_momentum_init(
            tuple(t.name for t in HistoryType),
            default="ZERO",
            tooltip="Initial value used for momentum history. ZERO - history starts zeroed out. RAND - History is initialized with a random value. SAMPLE - History is initialized from the latent at the start of sampling.",
        )
        .req_float_direction(
            default=1.0,
            min=-30.0,
            max=15.0,
            tooltip="Multiplier applied to the result of normal sampling.",
        )
        .req_field_init_noise_type(
            tuple(NoiseType.get_names(skip=(NoiseType.BROWNIAN,))),
            default="gaussian",
            tooltip="Noise type to use when momentum_init is set to RANDOM.",
        )
        .opt_field_guidance_cfg_opt(
            "SONAR_GUIDANCE_CFG",
            tooltip="Optional input for extended guidance parameters.",
        ),
    )

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
    INPUT_TYPES = SonarLazyInputTypes(
        lambda: SonarInputTypes(parent=SamplerNodeSonarEuler)
        .req_float_s_noise(
            default=1.0,
            tooltip="Multiplier for noise added during ancestral or SDE sampling.",
        )
        .req_float_eta(
            default=1.0,
            tooltip="Basically controls the ancestralness of the sampler. When set to 0, you will get a non-ancestral (or SDE) sampler.",
        )
        .req_selectnoise_noise_type(
            tooltip="Noise type used during ancestral or SDE sampling. Only used when the custom noise input is not connected.",
        )
        .opt_customnoise_custom_noise_opt(
            tooltip="Optional input for custom noise used during ancestral or SDE sampling. When connected, the built-in noise_type selector is ignored.",
        ),
    )

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


class SamplerNodeSonarDPMPPSDE(SamplerNodeSonarEulerAncestral):
    INPUT_TYPES = SonarLazyInputTypes(
        lambda: SonarInputTypes(
            parent=SamplerNodeSonarEulerAncestral,
        ).req_selectnoise_noise_type(default="brownian"),
    )

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
