from __future__ import annotations

from comfy import samplers

from .. import external, noise
from .base import (
    NoiseNoChainInputTypes,
    SonarCustomNoiseNodeBase,
    SonarInputTypes,
    SonarLazyInputTypes,
    SonarNormalizeNoiseNodeMixin,
)

NODE_CLASS_MAPPINGS = {}


bleh = None


class SonarBlendFilterNoiseNode(
    SonarCustomNoiseNodeBase,
    SonarNormalizeNoiseNodeMixin,
):
    DESCRIPTION = "Custom noise type that allows blending and filtering the output of another noise generator using ComfyUI-bleh."

    INPUT_TYPES = SonarLazyInputTypes(
        lambda: NoiseNoChainInputTypes()
        .req_customnoise_sonar_custom_noise()
        .req_selectblend(insert_modes=("simple_add",), default="simple_add")
        .req_field_ffilter(
            () if bleh is None else tuple(bleh.py.latent_utils.FILTER_PRESETS.keys()),
        )
        .req_string_ffilter_custom(default="")
        .req_float_ffilter_scale(default=1.0)
        .req_float_ffilter_strength(default=0.0)
        .req_int_ffilter_threshold(default=1, min=1, max=32)
        .req_field_enhance_mode(
            ("none",)
            if bleh is None
            else ("none", *bleh.py.latent_utils.ENHANCE_METHODS),
            default="none",
        )
        .req_float_enhance_strength(default=0.0)
        .req_field_affect(("result", "noise", "both"), default="result")
        .req_normalizetristate_normalize_result()
        .req_normalizetristate_normalize_noise(),
    )

    @classmethod
    def get_item_class(cls):
        return noise.BlendFilterNoise

    def go(
        self,
        *,
        factor,
        sonar_custom_noise,
        blend_mode,
        ffilter,
        ffilter_custom,
        ffilter_scale,
        ffilter_strength,
        ffilter_threshold,
        enhance_mode,
        enhance_strength,
        affect,
        normalize_result,
        normalize_noise,
    ):
        if bleh is None:
            raise RuntimeError("bleh not available")
        import ast  # noqa: PLC0415

        ffilter_custom = ffilter_custom.strip()
        normalize_result = (
            None if normalize_result == "default" else normalize_result == "forced"
        )
        normalize_noise = (
            None if normalize_noise == "default" else normalize_noise == "forced"
        )
        if ffilter_custom:
            ffilter = ast.literal_eval(f"[{ffilter_custom}]")
        elif ffilter == "none":
            ffilter = None
        else:
            ffilter = bleh.py.latent_utils.FILTER_PRESETS[ffilter]
        return super().go(
            factor,
            noise=sonar_custom_noise.clone(),
            blend_mode=blend_mode,
            ffilter=ffilter,
            ffilter_scale=ffilter_scale,
            ffilter_strength=ffilter_strength,
            ffilter_threshold=ffilter_threshold,
            enhance_mode=enhance_mode,
            enhance_strength=enhance_strength,
            affect=affect,
            normalize_noise=self.get_normalize(normalize_noise),
            normalize_result=self.get_normalize(normalize_result),
        )


class SonarBlehOpsNoiseNode(
    SonarCustomNoiseNodeBase,
    SonarNormalizeNoiseNodeMixin,
):
    DESCRIPTION = (
        "Custom noise type that allows manipulating noise with ComfyUI-bleh ops."
    )

    INPUT_TYPES = SonarLazyInputTypes(
        lambda: NoiseNoChainInputTypes()
        .req_customnoise_sonar_custom_noise()
        .req_normalizetristate_normalize()
        .req_yaml_rules(
            tooltip="Enter rules in the bleh block ops format here.",
            placeholder="# YAML ops here",
        ),
    )

    @classmethod
    def get_item_class(cls):
        return noise.BlehOpsNoise

    def go(
        self,
        *,
        factor,
        sonar_custom_noise,
        rules,
        normalize,
    ):
        if bleh is None:
            raise RuntimeError("bleh not available")
        return super().go(
            factor,
            noise=sonar_custom_noise.clone(),
            rules=bleh.py.nodes.ops.RuleGroup.from_yaml(rules),
            normalize=normalize,
        )


restart = None


def KRestartSamplerCustomNoise_INPUT_TYPES_BUILDER():
    if restart is not None:
        get_normal_schedulers = getattr(
            restart.nodes,
            "get_supported_normal_schedulers",
            restart.nodes.get_supported_restart_schedulers,
        )
        restart_normal_schedulers = get_normal_schedulers()
        restart_schedulers = restart.nodes.get_supported_restart_schedulers()
        restart_default_segments = restart.restart_sampling.DEFAULT_SEGMENTS
    else:
        restart_default_segments = ""
        restart_normal_schedulers = restart_schedulers = ()
    return (
        SonarInputTypes()
        .req_model()
        .req_field_add_noise(("enable", "disable"), default="enable")
        .req_seed_noise_seed()
        .req_int_steps(default=20, min=1)
        .req_float_cfg(default=8.0, min=0.0)
        .req_sampler()
        .req_field_scheduler(restart_normal_schedulers)
        .req_conditioning_positive()
        .req_conditioning_negative()
        .req_latent_latent_image()
        .req_int_start_at_step(default=0, min=0)
        .req_int_end_at_step(default=10000, min=0)
        .req_field_return_with_leftover_noise(
            ("disable", "enable"),
            default="disable",
        )
        .req_string_segments(default=restart_default_segments)
        .req_field_restart_scheduler(restart_schedulers)
        .req_bool_chunked_mode(default=True)
        .opt_customnoise_custom_noise_opt(tooltip="Optional custom noise input.")
    )


class KRestartSamplerCustomNoise:
    DESCRIPTION = "Restart sampler variant that allows specifying a custom noise type for noise added by restarts."

    INPUT_TYPES = SonarLazyInputTypes(KRestartSamplerCustomNoise_INPUT_TYPES_BUILDER)

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("output", "denoised_output")
    FUNCTION = "go"
    CATEGORY = "sampling"

    @classmethod
    def go(
        cls,
        *,
        model,
        add_noise,
        noise_seed,
        steps,
        cfg,
        sampler,
        scheduler,
        positive,
        negative,
        latent_image,
        start_at_step,
        end_at_step,
        return_with_leftover_noise,
        segments,
        restart_scheduler,
        chunked_mode=False,
        custom_noise_opt=None,
    ):
        if restart is None:
            raise RuntimeError("Restart not available")
        return restart.restart_sampling.restart_sampling(
            model,
            noise_seed,
            steps,
            cfg,
            sampler,
            scheduler,
            positive,
            negative,
            latent_image,
            segments,
            restart_scheduler,
            disable_noise=add_noise == "disable",
            step_range=(start_at_step, end_at_step),
            force_full_denoise=return_with_leftover_noise != "enable",
            output_only=False,
            chunked_mode=chunked_mode,
            custom_noise=custom_noise_opt.make_noise_sampler
            if custom_noise_opt
            else None,
        )


class RestartSamplerCustomNoise:
    DESCRIPTION = "Wrapper used to make another sampler Restart compatible. Allows specifying a custom type for noise added by restarts."

    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "go"
    CATEGORY = "sampling/custom_sampling/samplers"

    INPUT_TYPES = SonarLazyInputTypes(
        lambda: SonarInputTypes()
        .req_sampler()
        .req_bool_chunked_mode(default=True)
        .opt_customnoise_custom_noise_opt(tooltip="Optional custom noise input."),
    )

    @classmethod
    def go(cls, sampler, chunked_mode, custom_noise_opt=None):
        if restart is None or not hasattr(restart.restart_sampling, "RestartSampler"):
            raise RuntimeError("Restart not available")
        restart_options = {
            "restart_chunked": chunked_mode,
            "restart_wrapped_sampler": sampler,
            "restart_custom_noise": None
            if custom_noise_opt is None
            else custom_noise_opt.make_noise_sampler,
        }
        restart_sampler = samplers.KSAMPLER(
            restart.restart_sampling.RestartSampler.sampler_function,
            extra_options=sampler.extra_options | restart_options,
            inpaint_options=sampler.inpaint_options,
        )
        return (restart_sampler,)


NODE_CLASS_MAPPINGS |= {
    "KRestartSamplerCustomNoise": KRestartSamplerCustomNoise,
    "RestartSamplerCustomNoise": RestartSamplerCustomNoise,
    "SonarBlendFilterNoise": SonarBlendFilterNoiseNode,
    "SonarBlehOpsNoise": SonarBlehOpsNoiseNode,
}


def init_integrations(integrations):
    global restart, bleh  # noqa: PLW0603
    restart = integrations.restart
    bleh = integrations.bleh


external.MODULES.register_init_handler(init_integrations)
