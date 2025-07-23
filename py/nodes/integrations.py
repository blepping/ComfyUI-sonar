from __future__ import annotations

from comfy import samplers

from .. import external, noise, utils
from .base import (
    NOISE_INPUT_TYPES_HINT,
    WILDCARD_NOISE,
    IntegratedNode,
    SonarCustomNoiseNodeBase,
    SonarNormalizeNoiseNodeMixin,
)

NODE_CLASS_MAPPINGS = {}


bleh = None


class SonarBlendFilterNoiseNode(
    SonarCustomNoiseNodeBase,
    SonarNormalizeNoiseNodeMixin,
):
    DESCRIPTION = "Custom noise type that allows blending and filtering the output of another noise generator using ComfyUI-bleh."

    @classmethod
    def INPUT_TYPES(cls):
        result = super().INPUT_TYPES(include_rescale=False, include_chain=False)
        bleh_filter_presets = (
            () if bleh is None else tuple(bleh.py.latent_utils.FILTER_PRESETS.keys())
        )
        bleh_enhance_methods = (
            () if bleh is None else ("none", *bleh.py.latent_utils.ENHANCE_METHODS)
        )
        result["required"] |= {
            "sonar_custom_noise": (
                WILDCARD_NOISE,
                {
                    "tooltip": f"Custom noise input.\n{NOISE_INPUT_TYPES_HINT}",
                },
            ),
            "blend_mode": (
                ("simple_add", *utils.BLENDING_MODES.keys()),
                {"default": "simple_add"},
            ),
            "ffilter": (bleh_filter_presets,),
            "ffilter_custom": ("STRING", {"default": ""}),
            "ffilter_scale": (
                "FLOAT",
                {
                    "default": 1.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 0.001,
                    "round": False,
                },
            ),
            "ffilter_strength": (
                "FLOAT",
                {
                    "default": 0.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 0.001,
                    "round": False,
                },
            ),
            "ffilter_threshold": (
                "INT",
                {"default": 1, "min": 1, "max": 32},
            ),
            "enhance_mode": (bleh_enhance_methods,),
            "enhance_strength": (
                "FLOAT",
                {
                    "default": 0.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 0.001,
                    "round": False,
                },
            ),
            "affect": (("result", "noise", "both"),),
            "normalize_result": (("default", "forced", "disabled"),),
            "normalize_noise": (("default", "forced", "disabled"),),
        }
        return result

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

    @classmethod
    def INPUT_TYPES(cls):
        result = super().INPUT_TYPES(include_rescale=False, include_chain=False)
        result["required"] |= {
            "sonar_custom_noise": (
                WILDCARD_NOISE,
                {
                    "tooltip": f"Custom noise input.\n{NOISE_INPUT_TYPES_HINT}",
                },
            ),
            "normalize": (
                ("default", "forced", "disabled"),
                {
                    "tooltip": "Controls whether the generated noise is normalized to 1.0 strength.",
                },
            ),
            "rules": (
                "STRING",
                {
                    "tooltip": "Enter rules in the bleh block ops format here.",
                    "placeholder": "# YAML ops here",
                    "dynamicPrompts": False,
                    "multiline": True,
                },
            ),
        }
        return result

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


class KRestartSamplerCustomNoise(metaclass=IntegratedNode):
    DESCRIPTION = "Restart sampler variant that allows specifying a custom noise type for noise added by restarts."

    @classmethod
    def INPUT_TYPES(cls):
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
        return {
            "required": {
                "model": ("MODEL",),
                "add_noise": (["enable", "disable"],),
                "noise_seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF},
                ),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 8.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.001,
                        "round": False,
                    },
                ),
                "sampler": ("SAMPLER",),
                "scheduler": (restart_normal_schedulers,),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                "return_with_leftover_noise": (["disable", "enable"],),
                "segments": (
                    "STRING",
                    {
                        "default": restart_default_segments,
                        "multiline": False,
                    },
                ),
                "restart_scheduler": (restart_schedulers,),
                "chunked_mode": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "custom_noise_opt": (
                    WILDCARD_NOISE,
                    {
                        "tooltip": f"Optional custom noise input.\n{NOISE_INPUT_TYPES_HINT}",
                    },
                ),
            },
        }

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


class RestartSamplerCustomNoise(metaclass=IntegratedNode):
    DESCRIPTION = "Wrapper used to make another sampler Restart compatible. Allows specifying a custom type for noise added by restarts."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampler": ("SAMPLER",),
                "chunked_mode": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "custom_noise_opt": (
                    WILDCARD_NOISE,
                    {
                        "tooltip": f"Optional custom noise input.\n{NOISE_INPUT_TYPES_HINT}",
                    },
                ),
            },
        }

    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "go"
    CATEGORY = "sampling/custom_sampling/samplers"

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
