from __future__ import annotations

import abc
import inspect
import math
import random
from typing import Any, Callable

import numpy as np
import torch
import yaml
from comfy import model_management, samplers

from . import external, noise, utils
from .external import IntegratedNode
from .noise import NoiseType
from .noise_generation import DistroNoiseGenerator
from .sonar import (
    GuidanceConfig,
    GuidanceType,
    HistoryType,
    SonarConfig,
    SonarDPMPPSDE,
    SonarEuler,
    SonarEulerAncestral,
    SonarGuidanceMixin,
)

try:
    from comfy_execution import validation as comfy_validation

    if not hasattr(comfy_validation, "validate_node_input"):
        raise NotImplementedError  # noqa: TRY301
    HAVE_COMFY_UNION_TYPE = comfy_validation.validate_node_input("B", "A,B")
except (ImportError, NotImplementedError):
    HAVE_COMFY_UNION_TYPE = False
except Exception as exc:  # noqa: BLE001
    HAVE_COMFY_UNION_TYPE = False
    print(
        f"** ComfyUI-sonar: Warning, caught unexpected exception trying to detect ComfyUI union type support. Disabling. Exception: {exc}",
    )

NOISE_INPUT_TYPES = frozenset(("SONAR_CUSTOM_NOISE", "OCS_NOISE"))

if not HAVE_COMFY_UNION_TYPE:

    class Wildcard(str):  # noqa: FURB189
        __slots__ = ("whitelist",)

        @classmethod
        def __new__(cls, s, *args: list, whitelist=None, **kwargs: dict):
            result = super().__new__(s, *args, **kwargs)
            result.whitelist = whitelist
            return result

        def __ne__(self, other):
            return False if self.whitelist is None else other not in self.whitelist

    WILDCARD_NOISE = Wildcard("*", whitelist=NOISE_INPUT_TYPES)
else:
    WILDCARD_NOISE = ",".join(NOISE_INPUT_TYPES)


NOISE_INPUT_TYPES_HINT = (
    f"The following input types are supported: {', '.join(NOISE_INPUT_TYPES)}"
)


class NoisyLatentLikeNode(metaclass=IntegratedNode):
    DESCRIPTION = "Allows generating noise (and optionally adding it) based on a reference latent. Note: For img2img workflows, you will generally want to enable add_to_latent as well as connecting the model and sigmas inputs."
    RETURN_TYPES = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The noisy latent image.",)
    CATEGORY = "latent/noise"

    FUNCTION = "go"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise_type": (
                    tuple(NoiseType.get_names()),
                    {
                        "default": "gaussian",
                        "tooltip": "Sets the type of noise to generate. Has no effect when the custom_noise_opt input is connected.",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "tooltip": "Seed to use for generated noise.",
                    },
                ),
                "latent": (
                    "LATENT",
                    {
                        "tooltip": "Latent used as a reference for generating noise.",
                    },
                ),
                "multiplier": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "step": 0.001,
                        "min": -10000.0,
                        "max": 10000.0,
                        "round": False,
                        "tooltip": "Multiplier for the strength of the generated noise. Performed after mul_by_sigmas_opt.",
                    },
                ),
                "add_to_latent": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Add the generated noise to the reference latent rather than adding it to an empty latent. Generally should be enabled for img2img workflows.",
                    },
                ),
                "repeat_batch": (
                    "INT",
                    {
                        "default": 1,
                        "tooltip": "Repeats the noise generation the specified number of times. For example, if set to two and your reference latent is also batch two you will get a batch of four as output.",
                    },
                ),
                "cpu_noise": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Controls whether noise will be generated on GPU or CPU. Only affects noise types that support GPU generation (maybe only Brownian).",
                    },
                ),
                "normalize": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Controls whether the generated noise is normalized to 1.0 strength before scaling. Generally should be left enabled.",
                    },
                ),
            },
            "optional": {
                "custom_noise_opt": (
                    WILDCARD_NOISE,
                    {
                        "tooltip": f"Allows connecting a custom noise chain. When connected, noise_type has no effect.\n{NOISE_INPUT_TYPES_HINT}",
                    },
                ),
                "mul_by_sigmas_opt": (
                    "SIGMAS",
                    {
                        "tooltip": "When connected, will scale the generated noise by the first sigma. Must also connect model_opt to enable.",
                    },
                ),
                "model_opt": (
                    "MODEL",
                    {
                        "tooltip": "Used when mul_by_sigmas_opt is connected, no effect otherwise.",
                    },
                ),
            },
        }

    @classmethod
    def go(  # noqa: PLR0914
        cls,
        *,
        noise_type: str,
        seed: int | None,
        latent: dict,
        multiplier: float = 1.0,
        add_to_latent=False,
        repeat_batch=1,
        cpu_noise=True,
        normalize=True,
        custom_noise_opt: object | None = None,
        mul_by_sigmas_opt: torch.Tensor | None = None,
        model_opt: object | None = None,
    ):
        model, sigmas = model_opt, mul_by_sigmas_opt
        if sigmas is not None and len(sigmas) > 0:
            if model is None:
                raise ValueError(
                    "NoisyLatentLike requires a model when sigmas are connected!",
                )
            while hasattr(model, "model"):
                model = model.model
            latent_scale_factor = model.latent_format.scale_factor
            model_sigma_max = float(model.model_sampling.sigma_max)
            first_sigma = float(sigmas[0])
            max_denoise = (
                math.isclose(model_sigma_max, first_sigma, rel_tol=1e-05)
                or first_sigma > model_sigma_max
            )
            multiplier *= (
                float(
                    torch.sqrt(1.0 + sigmas[0] ** 2.0) if max_denoise else sigmas[0],
                )
                / latent_scale_factor
            )
        if sigmas is not None and sigmas.numel() > 1:
            sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
            sigma, sigma_next = sigmas[0], sigmas[1]
        else:
            sigma_min, sigma_max, sigma, sigma_next = (None,) * 4
        latent_samples = latent["samples"]
        orig_device = latent_samples.device
        want_device = (
            torch.device("cpu") if cpu_noise else model_management.get_torch_device()
        )
        if latent_samples.device != want_device:
            latent_samples = latent_samples.detach().clone().to(want_device)
        if custom_noise_opt is not None:
            ns = custom_noise_opt.make_noise_sampler(
                latent_samples,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                seed=seed,
                cpu=cpu_noise,
                normalized=normalize,
            )
        else:
            ns = noise.get_noise_sampler(
                NoiseType[noise_type.upper()],
                latent_samples,
                sigma_min,
                sigma_max,
                seed=seed,
                cpu=cpu_noise,
                normalized=normalize,
            )
        randst = torch.random.get_rng_state()
        try:
            torch.random.manual_seed(seed)
            result = torch.cat(
                tuple(ns(sigma, sigma_next) for _ in range(repeat_batch)),
                dim=0,
            )
        finally:
            torch.random.set_rng_state(randst)
        result = utils.scale_noise(result, multiplier, normalized=True)
        if add_to_latent:
            result += latent_samples.repeat(
                *(repeat_batch if i == 0 else 1 for i in range(latent_samples.ndim)),
            ).to(result)
        result = result.to(orig_device)
        return ({"samples": result},)


class SonarCustomNoiseNodeBase(metaclass=IntegratedNode):
    DESCRIPTION = "A custom noise item."
    RETURN_TYPES = ("SONAR_CUSTOM_NOISE",)
    OUTPUT_TOOLTIPS = ("A custom noise chain.",)
    CATEGORY = "advanced/noise"
    FUNCTION = "go"

    @abc.abstractmethod
    def get_item_class(self):
        raise NotImplementedError

    @classmethod
    def INPUT_TYPES(cls, *, include_rescale=True, include_chain=True):
        result = {
            "required": {
                "factor": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -10000.0,
                        "max": 10000.0,
                        "step": 0.001,
                        "round": False,
                        "tooltip": "Scaling factor for the generated noise of this type.",
                    },
                ),
            },
            "optional": {},
        }
        if include_rescale:
            result["required"] |= {
                "rescale": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 10000.0,
                        "step": 0.001,
                        "round": False,
                        "tooltip": "When non-zero, this custom noise item and other custom noise items items connected to it will have their factor scaled to add up to the specified rescale value. When set to 0, rescaling is disabled.",
                    },
                ),
            }
        if include_chain:
            result["optional"] |= {
                "sonar_custom_noise_opt": (
                    WILDCARD_NOISE,
                    {
                        "tooltip": f"Optional input for more custom noise items.\n{NOISE_INPUT_TYPES_HINT}",
                    },
                ),
            }
        return result

    def go(
        self,
        factor=1.0,
        rescale=0.0,
        sonar_custom_noise_opt=None,
        **kwargs: dict[str, Any],
    ):
        nis = (
            sonar_custom_noise_opt.clone()
            if sonar_custom_noise_opt
            else noise.CustomNoiseChain()
        )
        if factor != 0:
            nis.add(self.get_item_class()(factor, **kwargs))
        return (nis if rescale == 0 else nis.rescaled(rescale),)


class SonarCustomNoiseNode(SonarCustomNoiseNodeBase):
    @classmethod
    def INPUT_TYPES(cls):
        result = super().INPUT_TYPES()
        result["required"] |= {
            "noise_type": (
                tuple(NoiseType.get_names()),
                {
                    "tooltip": "Sets the type of noise to generate.",
                },
            ),
        }
        return result

    @classmethod
    def get_item_class(cls):
        return noise.CustomNoiseItem


class SonarCustomNoiseAdvNode(SonarCustomNoiseNode):
    DESCRIPTION = "A custom noise item allowing advanced YAML parameter input."

    @classmethod
    def INPUT_TYPES(cls):
        result = super().INPUT_TYPES()
        result["optional"] |= {
            "yaml_parameters": (
                "STRING",
                {
                    "tooltip": "Allows specifying custom parameters via YAML. Note: When specifying paramaters this way, there is no error checking.",
                    "placeholder": "# YAML or JSON here",
                    "dynamicPrompts": False,
                    "multiline": True,
                },
            ),
        }
        return result


class SonarNormalizeNoiseNodeMixin:
    @staticmethod
    def get_normalize(val: str) -> bool | None:
        return None if val == "default" else val == "forced"


class SonarModulatedNoiseNode(SonarCustomNoiseNodeBase, SonarNormalizeNoiseNodeMixin):
    DESCRIPTION = "Custom noise type that allows modulating the output of another custom noise generator."

    @classmethod
    def INPUT_TYPES(cls):
        result = super().INPUT_TYPES(include_rescale=False, include_chain=False)
        result["required"] |= {
            "sonar_custom_noise": (
                WILDCARD_NOISE,
                {
                    "tooltip": f"Input custom noise to modulate.\n{NOISE_INPUT_TYPES_HINT}",
                },
            ),
            "modulation_type": (
                (
                    "intensity",
                    "frequency",
                    "spectral_signum",
                    "none",
                ),
                {
                    "tooltip": "Type of modulation to use.",
                },
            ),
            "dims": (
                "INT",
                {
                    "default": 3,
                    "min": 1,
                    "max": 3,
                    "tooltip": "Dimensions to modulate over. 1 - channels only, 2 - height and width, 3 - both",
                },
            ),
            "strength": (
                "FLOAT",
                {
                    "default": 2.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 0.001,
                    "round": False,
                    "tooltip": "Controls the strength of the modulation effect.",
                },
            ),
            "normalize_result": (
                ("default", "forced", "disabled"),
                {
                    "tooltip": "Controls whether the final result is normalized to 1.0 strength.",
                },
            ),
            "normalize_noise": (
                ("default", "forced", "disabled"),
                {
                    "tooltip": "Controls whether the generated noise is normalized to 1.0 strength.",
                },
            ),
            "normalize_ref": (
                "BOOLEAN",
                {
                    "default": True,
                    "tooltip": "Controls whether the reference latent (when present) is normalized to 1.0 strength.",
                },
            ),
        }
        result["optional"] |= {"ref_latent_opt": ("LATENT",)}
        return result

    @classmethod
    def get_item_class(cls):
        return noise.ModulatedNoise

    def go(
        self,
        *,
        factor,
        sonar_custom_noise,
        modulation_type,
        dims,
        strength,
        normalize_result,
        normalize_noise,
        normalize_ref,
        ref_latent_opt=None,
    ):
        if ref_latent_opt is not None:
            ref_latent_opt = ref_latent_opt["samples"].clone()
        return super().go(
            factor,
            noise=sonar_custom_noise,
            modulation_type=modulation_type,
            modulation_dims=dims,
            modulation_strength=strength,
            normalize_result=self.get_normalize(normalize_result),
            normalize_noise=self.get_normalize(normalize_noise),
            normalize_ref=self.get_normalize(normalize_ref),
            ref_latent_opt=ref_latent_opt,
        )


class SonarRepeatedNoiseNode(SonarCustomNoiseNodeBase, SonarNormalizeNoiseNodeMixin):
    DESCRIPTION = "Custom noise type that allows caching the output of other custom noise generators."

    @classmethod
    def INPUT_TYPES(cls):
        result = super().INPUT_TYPES(include_rescale=False, include_chain=False)
        result["required"] |= {
            "sonar_custom_noise": (
                WILDCARD_NOISE,
                {
                    "tooltip": f"Custom noise input for items to repeat. Note: Unlike most other custom noise nodes, this is treated like a list.\n{NOISE_INPUT_TYPES_HINT}",
                },
            ),
            "repeat_length": (
                "INT",
                {
                    "default": 8,
                    "min": 1,
                    "max": 100,
                    "tooltip": "Number of items to cache.",
                },
            ),
            "max_recycle": (
                "INT",
                {
                    "default": 1000,
                    "min": 1,
                    "max": 1000,
                    "tooltip": "Number of times an individual item will be used before it is replaced with fresh noise.",
                },
            ),
            "normalize": (
                ("default", "forced", "disabled"),
                {
                    "tooltip": "Controls whether the generated noise is normalized to 1.0 strength.",
                },
            ),
            "permute": (
                ("enabled", "disabled", "always"),
                {
                    "tooltip": "When enabled, recycled noise will be permuted by randomly flipping it, rolling the channels, etc. If set to always, the noise will be permuted the first time it's used as well.",
                },
            ),
        }
        return result

    @classmethod
    def get_item_class(cls):
        return noise.RepeatedNoise

    def go(
        self,
        *,
        factor,
        sonar_custom_noise,
        repeat_length,
        max_recycle,
        normalize,
        permute=True,
    ):
        return super().go(
            factor,
            noise=sonar_custom_noise,
            repeat_length=repeat_length,
            max_recycle=max_recycle,
            normalize=self.get_normalize(normalize),
            permute=permute,
        )


class SonarScheduledNoiseNode(SonarCustomNoiseNodeBase, SonarNormalizeNoiseNodeMixin):
    DESCRIPTION = "Custom noise type that allows scheduling the output of other custom noise generators. NOTE: If you don't connect the fallback custom noise input, no noise will be generated outside of the start_percent, end_percent range. I recommend connecting a 1.0 strength Gaussian custom noise node as the fallback."

    @classmethod
    def INPUT_TYPES(cls):
        result = super().INPUT_TYPES(include_rescale=False, include_chain=False)
        result["required"] |= {
            "model": (
                "MODEL",
                {
                    "tooltip": "The model input is required to calculate sampling percentages.",
                },
            ),
            "sonar_custom_noise": (
                WILDCARD_NOISE,
                {
                    "tooltip": f"Custom noise to use when start_percent and end_percent matches.\n{NOISE_INPUT_TYPES_HINT}",
                },
            ),
            "start_percent": (
                "FLOAT",
                {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.001,
                    "round": False,
                    "tooltip": "Time the custom noise becomes active. Note: Sampling percentage where 1.0 indicates 100%, not based on steps.",
                },
            ),
            "end_percent": (
                "FLOAT",
                {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.001,
                    "round": False,
                    "tooltip": "Time the custom noise effect ends - inclusive, so only sampling percentages greater than this will be excluded. Note: Sampling percentage where 1.0 indicates 100%, not based on steps.",
                },
            ),
            "normalize": (
                ("default", "forced", "disabled"),
                {
                    "tooltip": "Controls whether the generated noise is normalized to 1.0 strength.",
                },
            ),
        }
        result["optional"] |= {
            "fallback_sonar_custom_noise": (
                WILDCARD_NOISE,
                {
                    "tooltip": f"Optional input for noise to use when outside of the start_percent, end_percent range. NOTE: When not connected, defaults to NO NOISE which is probably not what you want.\n{NOISE_INPUT_TYPES_HINT}",
                },
            ),
        }
        return result

    @classmethod
    def get_item_class(cls):
        return noise.ScheduledNoise

    def go(
        self,
        *,
        model,
        factor,
        sonar_custom_noise,
        start_percent,
        end_percent,
        normalize,
        fallback_sonar_custom_noise=None,
    ):
        ms = model.get_model_object("model_sampling")
        start_sigma = ms.percent_to_sigma(start_percent)
        end_sigma = ms.percent_to_sigma(end_percent)
        return super().go(
            factor,
            noise=sonar_custom_noise,
            start_sigma=start_sigma,
            end_sigma=end_sigma,
            normalize=self.get_normalize(normalize),
            fallback_noise=fallback_sonar_custom_noise,
        )


class SonarCompositeNoiseNode(SonarCustomNoiseNodeBase, SonarNormalizeNoiseNodeMixin):
    DESCRIPTION = "Custom noise type that allows compositing two other custom noise generators based on a mask."

    @classmethod
    def INPUT_TYPES(cls):
        result = super().INPUT_TYPES(include_rescale=False, include_chain=False)
        result["required"] |= {
            "sonar_custom_noise_dst": (
                WILDCARD_NOISE,
                {
                    "tooltip": f"Custom noise input for noise where the mask is not set.\n{NOISE_INPUT_TYPES_HINT}",
                },
            ),
            "sonar_custom_noise_src": (
                WILDCARD_NOISE,
                {
                    "tooltip": f"Custom noise input for noise where the mask is set.\n{NOISE_INPUT_TYPES_HINT}",
                },
            ),
            "normalize_dst": (
                ("default", "forced", "disabled"),
                {
                    "tooltip": "Controls whether noise generated for dst is normalized to 1.0 strength.",
                },
            ),
            "normalize_src": (
                ("default", "forced", "disabled"),
                {
                    "tooltip": "Controls whether noise generated for src is normalized to 1.0 strength.",
                },
            ),
            "normalize_result": (
                ("default", "forced", "disabled"),
                {
                    "tooltip": "Controls whether the final result after composition is normalized to 1.0 strength.",
                },
            ),
            "mask": (
                "MASK",
                {
                    "tooltip": "Mask to use when compositing noise. Where the mask is 1.0, you will get 100% src, where it is 0.75 you will get 75% src and 25% dst. The mask will be rescaled to match the latent size if necessary.",
                },
            ),
        }
        return result

    @classmethod
    def get_item_class(cls):
        return noise.CompositeNoise

    def go(
        self,
        *,
        factor,
        sonar_custom_noise_dst,
        sonar_custom_noise_src,
        normalize_src,
        normalize_dst,
        normalize_result,
        mask,
    ):
        return super().go(
            factor,
            dst_noise=sonar_custom_noise_dst,
            src_noise=sonar_custom_noise_src,
            normalize_dst=self.get_normalize(normalize_src),
            normalize_src=self.get_normalize(normalize_dst),
            normalize_result=self.get_normalize(normalize_result),
            mask=mask,
        )


class SonarGuidedNoiseNode(SonarCustomNoiseNodeBase, SonarNormalizeNoiseNodeMixin):
    DESCRIPTION = "Custom noise type that mixes a references with another custom noise generator to guide the generation."

    @classmethod
    def INPUT_TYPES(cls):
        result = super().INPUT_TYPES(include_rescale=False, include_chain=False)
        result["required"] |= {
            "latent": (
                "LATENT",
                {
                    "tooltip": "Latent to use for guidance.",
                },
            ),
            "sonar_custom_noise": (
                WILDCARD_NOISE,
                {
                    "tooltip": f"Custom noise input to combine with the guidance.\n{NOISE_INPUT_TYPES_HINT}",
                },
            ),
            "method": (
                ("euler", "linear"),
                {
                    "tooltip": "Method to use when calculating guidance. When set to linear, will simply LERP the guidance at the specified strength. When set to Euler, will do a Euler step toward the guidance instead.",
                },
            ),
            "guidance_factor": (
                "FLOAT",
                {
                    "default": 0.0125,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 0.001,
                    "round": False,
                    "tooltip": "Strength of the guidance to apply. Generally should be a relatively slow value to avoid overpowering the generation.",
                },
            ),
            "normalize_noise": (
                ("default", "forced", "disabled"),
                {
                    "tooltip": "Controls whether the generated noise is normalized to 1.0 strength.",
                },
            ),
            "normalize_result": (
                ("default", "forced", "disabled"),
                {
                    "tooltip": "Controls whether the final result is normalized to 1.0 strength.",
                },
            ),
            "normalize_ref": (
                "BOOLEAN",
                {
                    "default": True,
                    "tooltip": "Controls whether the reference latent (when present) is normalized to 1.0 strength.",
                },
            ),
        }
        return result

    @classmethod
    def get_item_class(cls):
        return noise.GuidedNoise

    def go(
        self,
        *,
        factor,
        latent,
        sonar_custom_noise,
        normalize_noise,
        normalize_result,
        normalize_ref=True,
        method="euler",
        guidance_factor=0.5,
    ):
        return super().go(
            factor,
            ref_latent=utils.scale_noise(
                SonarGuidanceMixin.prepare_ref_latent(latent["samples"].clone()),
                normalized=normalize_ref,
            ),
            guidance_factor=guidance_factor,
            noise=sonar_custom_noise.clone(),
            method=method,
            normalize_noise=self.get_normalize(normalize_noise),
            normalize_result=self.get_normalize(normalize_result),
        )


class SonarRandomNoiseNode(SonarCustomNoiseNodeBase, SonarNormalizeNoiseNodeMixin):
    DESCRIPTION = "Custom noise type that randomly selects between other custom noise items connected to it."

    @classmethod
    def INPUT_TYPES(cls):
        result = super().INPUT_TYPES(include_rescale=False, include_chain=False)
        result["required"] |= {
            "sonar_custom_noise": (
                WILDCARD_NOISE,
                {
                    "tooltip": f"Custom noise input for noise items to randomize. Note: Unlike most other custom noise nodes, this is treated like a list.\n{NOISE_INPUT_TYPES_HINT}",
                },
            ),
            "mix_count": (
                "INT",
                {
                    "default": 1,
                    "min": 1,
                    "max": 100,
                    "tooltip": "Number of items to select each time noise is generated.",
                },
            ),
            "normalize": (
                ("default", "forced", "disabled"),
                {
                    "tooltip": "Controls whether the generated noise is normalized to 1.0 strength.",
                },
            ),
        }

        return result

    @classmethod
    def get_item_class(cls):
        return noise.RandomNoise

    def go(
        self,
        factor,
        sonar_custom_noise,
        mix_count,
        normalize,
    ):
        return super().go(
            factor,
            noise=sonar_custom_noise,
            mix_count=mix_count,
            normalize=self.get_normalize(normalize),
        )


class SonarChannelNoiseNode(SonarCustomNoiseNodeBase, SonarNormalizeNoiseNodeMixin):
    DESCRIPTION = "Custom noise type that uses a different noise generator for each channel. Note: The connected noise items are treated as a list. If you want to blend noise types, you can use something like a SonarBlendedNoise node."

    @classmethod
    def INPUT_TYPES(cls):
        result = super().INPUT_TYPES(include_rescale=False, include_chain=False)
        result["required"] |= {
            "sonar_custom_noise": (
                WILDCARD_NOISE,
                {
                    "tooltip": f"Custom noise input for noise items corresponding to each channel. SD1/2x and SDXL use 4 channels, Flux and SD3 use 16. Note: Unlike most other custom noise nodes, this is treated like a list where the noise item furthest from the node corresponds to channel 0.\n{NOISE_INPUT_TYPES_HINT}",
                },
            ),
            "insufficient_channels_mode": (
                ("wrap", "repeat", "zero"),
                {
                    "default": "wrap",
                    "tooltip": "Controls behavior for when there are less noise items connected than channels in the latent. wrap - wraps back to the first noise item, repeat - repeats the last item, zero - fills the channel with zeros (generally not recommended).",
                },
            ),
            "normalize": (
                ("default", "forced", "disabled"),
                {
                    "tooltip": "Controls whether the generated noise is normalized to 1.0 strength.",
                },
            ),
        }

        return result

    @classmethod
    def get_item_class(cls):
        return noise.ChannelNoise

    def go(
        self,
        factor,
        *,
        sonar_custom_noise,
        insufficient_channels_mode,
        normalize,
    ):
        return super().go(
            factor,
            noise=sonar_custom_noise,
            insufficient_channels_mode=insufficient_channels_mode,
            normalize=self.get_normalize(normalize),
        )


class SonarBlendedNoiseNode(SonarCustomNoiseNodeBase, SonarNormalizeNoiseNodeMixin):
    DESCRIPTION = "Custom noise type that allows blending two other noise items."

    @classmethod
    def INPUT_TYPES(cls):
        result = super().INPUT_TYPES()
        result["required"] |= {
            "noise_2_percent": (
                "FLOAT",
                {
                    "default": 0.5,
                    "step": 0.001,
                    "round": False,
                    "tooltip": "Blend strength for custom_noise_2. Note that if set to 0 then custom_noise_2 is optional (and will not be called to generate noise) and if set to 1 then custom_noise_1 will not be called to generate noise. This is worth mentioning since going from a strength of 0.000000001 to 0 could make a big difference.",
                },
            ),
            "blend_mode": (
                tuple(utils.BLENDING_MODES.keys()),
                {
                    "default": "lerp",
                    "tooltip": "Mode used for blending the two noise types. More modes will be available if ComfyUI-bleh is installed.",
                },
            ),
            "normalize": (
                ("default", "forced", "disabled"),
                {
                    "tooltip": "Controls whether the generated noise is normalized to 1.0 strength. For weird blend modes, you may want to set this to forced.",
                },
            ),
        }
        result["optional"] |= {
            "custom_noise_1": (
                WILDCARD_NOISE,
                {
                    "tooltip": f"Custom noise. Optional if noise_2 percent is 1.\n{NOISE_INPUT_TYPES_HINT}",
                },
            ),
            "custom_noise_2": (
                WILDCARD_NOISE,
                {
                    "tooltip": f"Custom noise. Optional if noise_2_percent is 0.\n{NOISE_INPUT_TYPES_HINT}",
                },
            ),
        }
        return result

    @classmethod
    def get_item_class(cls):
        return noise.BlendedNoise

    def go(
        self,
        *,
        factor,
        rescale,
        sonar_custom_noise_opt=None,
        normalize,
        noise_2_percent,
        custom_noise_1=None,
        custom_noise_2=None,
        blend_mode="lerp",
    ):
        blend_function = utils.BLENDING_MODES.get(blend_mode)
        if blend_function is None:
            raise ValueError("Unknown blend mode")
        return super().go(
            factor,
            rescale=rescale,
            sonar_custom_noise_opt=sonar_custom_noise_opt,
            blend_function=blend_function,
            normalize=self.get_normalize(normalize),
            custom_noise_1=custom_noise_1,
            custom_noise_2=custom_noise_2,
            noise_2_percent=noise_2_percent,
        )


class SonarResizedNoiseNode(SonarCustomNoiseNodeBase, SonarNormalizeNoiseNodeMixin):
    DESCRIPTION = "Custom noise type that allows resizing another noise item."

    @classmethod
    def INPUT_TYPES(cls):
        result = super().INPUT_TYPES(include_rescale=False, include_chain=False)
        result["required"] |= {
            "width": (
                "INT",
                {
                    "default": 1152,
                    "min": 16,
                    "max": 1024 * 1024 * 1024,
                    "step": 8,
                    "tooltip": "Note: This should almost always be set to a higher value than the image you're actually sampling.",
                },
            ),
            "height": (
                "INT",
                {
                    "default": 1152,
                    "min": 16,
                    "max": 1024 * 1024 * 1024,
                    "step": 8,
                    "tooltip": "Note: This should almost always be set to a higher value than the image you're actually sampling.",
                },
            ),
            "downscale_strategy": (
                ("crop", "scale"),
                {
                    "default": "crop",
                    "tooltip": "Scaling noise is something you'd pretty much only use to create weird effects. For normal workflows, leave this on crop.",
                },
            ),
            "initial_reference": (
                ("prefer_crop", "prefer_scale"),
                {
                    "default": "prefer_crop",
                    "tooltip": "The initial latent the noise sampler uses as a reference may not match the requested width/height. This setting controls whether to crop or scale. Note: Cropping can only occur when the initial reference is larger than width/height in both dimensions which is unlikely (and not recommended).",
                },
            ),
            "crop_mode": (
                (
                    "center",
                    "top_left",
                    "top_center",
                    "top_right",
                    "center_left",
                    "center_right",
                    "bottom_left",
                    "bottom_center",
                    "bottom_right",
                ),
                {
                    "default": "center",
                    "tooltip": "Note: Crops will have a bias toward the lower number when the size isn't divisible by two. For example, a center crop of size 3 from (0, 1, 2, 3, 4, 5) will result in (1, 2, 3).",
                },
            ),
            "crop_offset_horizontal": (
                "INT",
                {
                    "default": 0,
                    "step": 8,
                    "min": -8000,
                    "max": 8000,
                    "tooltip": "This offsets the cropped view by the specified size. Positive values will move it toward the right, negative values will move it toward the left. The offsets will be adjusted to to fit in the available space. For example, if you have crop_mode set to top_right then setting a positive offset isn't going to do anything: it's already as far right as it can go.",
                },
            ),
            "crop_offset_vertical": (
                "INT",
                {
                    "default": 0,
                    "step": 8,
                    "min": -8000,
                    "max": 8000,
                    "tooltip": "This offsets the cropped view by the specified size. Positive values will move it toward the bottom, negative values will move it toward the top. The offsets will be adjusted to to fit in the available space. For example, if you have crop_mode set to bottom_right then setting a positive offset isn't going to do anything: it's already as far down as it can go.",
                },
            ),
            "upscale_mode": (
                utils.UPSCALE_METHODS,
                {
                    "tooltip": "Allows setting the scaling mode when width/height is smaller than the requested size.",
                    "default": "nearest-exact",
                },
            ),
            "downscale_mode": (
                utils.UPSCALE_METHODS,
                {
                    "tooltip": "Allows setting the scaling mode when width/height is larger than the requested size and downscale_strategy is set to 'scale'.",
                    "default": "nearest-exact",
                },
            ),
            "normalize": (
                ("default", "forced", "disabled"),
                {
                    "tooltip": "Controls whether the generated noise is normalized to 1.0 strength. For weird blend modes, you may want to set this to forced.",
                },
            ),
            "custom_noise": (
                WILDCARD_NOISE,
                {
                    "tooltip": f"Custom noise.\n{NOISE_INPUT_TYPES_HINT}",
                },
            ),
        }
        return result

    @classmethod
    def get_item_class(cls):
        return noise.ResizedNoise

    def go(
        self,
        *,
        factor,
        width,
        height,
        downscale_strategy,
        initial_reference,
        crop_offset_horizontal,
        crop_offset_vertical,
        crop_mode,
        upscale_mode,
        downscale_mode,
        normalize,
        custom_noise,
    ):
        return super().go(
            factor,
            width=width,
            height=height,
            downscale_strategy=downscale_strategy,
            initial_reference=initial_reference,
            crop_offset_horizontal=crop_offset_horizontal,
            crop_offset_vertical=crop_offset_vertical,
            crop_mode=crop_mode,
            upscale_mode=upscale_mode,
            downscale_mode=downscale_mode,
            normalize=normalize,
            custom_noise=custom_noise,
        )


class SonarAdvancedPyramidNoiseNode(SonarCustomNoiseNodeBase):
    DESCRIPTION = (
        "Custom noise type that allows specifying parameters for Pyramid variants."
    )

    @classmethod
    def INPUT_TYPES(cls):
        result = super().INPUT_TYPES()
        result["required"] |= {
            "variant": (
                (
                    "highres_pyramid",
                    "pyramid",
                    "pyramid_old",
                ),
                {
                    "tooltip": "Sets the Pyramid noise variant to generate.",
                    "default": "highres_pyramid",
                },
            ),
            "iterations": (
                "INT",
                {
                    "default": -1,
                    "min": -1,
                    "max": 8,
                    "tooltip": "When set to -1 will use the variant default.",
                },
            ),
            "discount": (
                "FLOAT",
                {
                    "default": 0.0,
                    "step": 0.001,
                    "min": -1000.0,
                    "max": 1000.0,
                    "round": False,
                    "tooltip": "When set to 0 will use the variant default.",
                },
            ),
            "upscale_mode": (
                ("default", *utils.UPSCALE_METHODS),
                {
                    "tooltip": "Allows setting the scaling mode for Pyramid noise. Leave on default to use the variant default.",
                    "default": "default",
                },
            ),
        }
        return result

    @classmethod
    def get_item_class(cls):
        return noise.AdvancedPyramidNoise

    def go(
        self,
        *,
        factor,
        rescale,
        variant,
        iterations,
        discount,
        upscale_mode,
        sonar_custom_noise_opt=None,
    ):
        return super().go(
            factor,
            rescale=rescale,
            sonar_custom_noise_opt=sonar_custom_noise_opt,
            variant=variant,
            iterations=iterations if iterations != -1 else None,
            discount=discount if discount != 0 else None,
            upscale_mode=upscale_mode if upscale_mode != "default" else None,
        )


class SonarAdvanced1fNoiseNode(SonarCustomNoiseNodeBase):
    DESCRIPTION = "Custom noise type that allows specifying parameters for 1f (pink, green, etc) variants."

    @classmethod
    def INPUT_TYPES(cls):
        result = super().INPUT_TYPES()
        result["required"] |= {
            "alpha": (
                "FLOAT",
                {
                    "default": 0.25,
                    "step": 0.001,
                    "min": -1000.0,
                    "max": 1000.0,
                    "round": False,
                    "tooltip": "Similar to the advanced power noise node, positive values increase low frequencies (with colorful effects), negative values increase high frequencies.",
                },
            ),
            "k": (
                "FLOAT",
                {
                    "default": 1.0,
                    "step": 0.001,
                    "min": -1000.0,
                    "max": 1000.0,
                    "round": False,
                    "tooltip": "Currently no description of exactly what it does, it's just another knob you can try turning for a different effect.",
                },
            ),
            "vertical_factor": (
                "FLOAT",
                {
                    "default": 1.0,
                    "step": 0.001,
                    "min": -1000.0,
                    "max": 1000.0,
                    "round": False,
                    "tooltip": "Vertical frequency scaling factor.",
                },
            ),
            "horizontal_factor": (
                "FLOAT",
                {
                    "default": 1.0,
                    "step": 0.001,
                    "min": -1000.0,
                    "max": 1000.0,
                    "round": False,
                    "tooltip": "Horizontal frequency scaling factor.",
                },
            ),
            "use_sqrt": (
                "BOOLEAN",
                {
                    "default": True,
                    "tooltip": "Controls whether to sqrt when dividing the FFT. Negative hfac/wfac won't work when enabled. Turning it off seems to make the parameters have a much stronger effect.",
                },
            ),
        }
        return result

    @classmethod
    def get_item_class(cls):
        return noise.Advanced1fNoise

    def go(
        self,
        *,
        factor,
        rescale,
        alpha,
        k,
        vertical_factor,
        horizontal_factor,
        use_sqrt,
        sonar_custom_noise_opt=None,
    ):
        return super().go(
            factor,
            rescale=rescale,
            sonar_custom_noise_opt=sonar_custom_noise_opt,
            alpha=alpha,
            k=k,
            hfac=vertical_factor,
            wfac=horizontal_factor,
            use_sqrt=use_sqrt,
        )


class SonarAdvancedPowerLawNoiseNode(SonarCustomNoiseNodeBase):
    DESCRIPTION = "Custom noise type that allows specifying parameters for power law (grey, violet, etc) variants. "

    @classmethod
    def INPUT_TYPES(cls):
        result = super().INPUT_TYPES()
        result["required"] |= {
            "alpha": (
                "FLOAT",
                {
                    "default": 0.5,
                    "step": 0.001,
                    "min": -1000.0,
                    "max": 1000.0,
                    "round": False,
                    "tooltip": "Alpha parameter of the generated noise. Positive values (low frequency noise) tend to produce colorful results.",
                },
            ),
            "div_max_dims": (
                (
                    "none",
                    "non-batch",
                    "spatial",
                    "all",
                    "batch",
                    "channel",
                    "height",
                    "width",
                ),
                {
                    "default": "non-batch",
                    "tooltip": "If non-none, the noise gets divide by the maxmimu over this dimension.",
                },
            ),
            "use_div_max_abs": (
                "BOOLEAN",
                {
                    "default": True,
                    "tooltip": "Only has an effect when div_max_dims is not none. Controls whether maximization is done with the absolute values or raw values.",
                },
            ),
            "use_sign": (
                "BOOLEAN",
                {
                    "default": False,
                    "tooltip": "When set, only the sign of the initial noise is used, so -0.5, -0.2 all turn into -1, 0.5, 2, etc all turn into 1.",
                },
            ),
        }
        return result

    @classmethod
    def get_item_class(cls):
        return noise.AdvancedPowerLawNoise

    MAX_DIMS_MAP = {  # noqa: RUF012
        "none": None,
        "non-batch": (-3, -2, -1),
        "spatial": (-2, -1),
        "all": (),
        "batch": 0,
        "channel": 1,
        "height": 2,
        "width": 3,
    }

    def go(
        self,
        *,
        factor,
        rescale,
        alpha,
        div_max_dims,
        use_sign,
        use_div_max_abs,
        sonar_custom_noise_opt=None,
    ):
        return super().go(
            factor,
            rescale=rescale,
            sonar_custom_noise_opt=sonar_custom_noise_opt,
            alpha=alpha,
            div_max_dims=self.MAX_DIMS_MAP.get(div_max_dims),
            use_sign=use_sign,
            use_div_max_abs=use_div_max_abs,
        )


class SonarAdvancedCollatzNoiseNode(SonarCustomNoiseNodeBase):
    DESCRIPTION = "Custom noise type that allows specifying parameters for Collatz noise. Very experimental, also very slow."

    @classmethod
    def INPUT_TYPES(cls):
        result = super().INPUT_TYPES()
        result["required"] |= {
            "adjust_scale": (
                "BOOLEAN",
                {
                    "default": True,
                },
            ),
            "use_initial": (
                "BOOLEAN",
                {
                    "default": True,
                },
            ),
            "iteration_sign_flipping": (
                "BOOLEAN",
                {
                    "default": False,
                },
            ),
            "chain_length": (
                "STRING",
                {
                    "default": "1, 2, 3, 4",
                    "tooltip": "Comma-separated list of chain lengths. Cannot be empty. Iterations will cycle through the list and wrap.",
                },
            ),
            "iterations": ("INT", {"default": 500, "min": 1, "max": 10000}),
            "rmin": ("FLOAT", {"default": -100.0, "min": -100000.0, "max": 100000.0}),
            "rmax": ("FLOAT", {"default": 100.0, "min": -100000.0, "max": 100000.0}),
            "flatten": ("BOOLEAN", {"default": False}),
            "dims": (
                "STRING",
                {
                    "default": "-1, -2",
                    "tooltip": "Comma-separated list of dimensions. Cannot be empty. May be negative to count from the end of the list. Iterations will cycle through the list and wrap.",
                },
            ),
        }
        return result

    @classmethod
    def get_item_class(cls):
        return noise.AdvancedCollatzNoise

    def go(
        self,
        *,
        factor,
        rescale,
        adjust_scale,
        use_initial,
        iteration_sign_flipping,
        chain_length,
        iterations,
        rmin,
        rmax,
        flatten,
        dims,
        sonar_custom_noise_opt=None,
    ):
        if rmin > rmax:
            rmin, rmax = rmax, rmin
        dims = tuple(int(i) for i in dims.split(","))
        return super().go(
            factor,
            rescale=rescale,
            sonar_custom_noise_opt=sonar_custom_noise_opt,
            adjust_scale=adjust_scale,
            use_initial=use_initial,
            iteration_sign_flipping=iteration_sign_flipping,
            chain_length=tuple(int(i) for i in chain_length.split(",")),
            iterations=iterations,
            rmin=rmin,
            rmax=rmax,
            flatten=flatten,
            dims=dims,
        )


class SonarQuantileFilteredNoiseNode(SonarCustomNoiseNodeBase):
    DESCRIPTION = "Custom noise type that allows filtering noise based on the quantile"

    @classmethod
    def INPUT_TYPES(cls):
        result = super().INPUT_TYPES(include_chain=False, include_rescale=False)
        result["required"] |= {
            "custom_noise": (
                WILDCARD_NOISE,
                {
                    "tooltip": f"Custom noise type to filter.\n{NOISE_INPUT_TYPES_HINT}",
                },
            ),
            "quantile": (
                "FLOAT",
                {
                    "default": 0.85,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.001,
                    "round": False,
                    "tooltip": "When enabled, will normalize generated noise to this quantile (i.e. 0.75 means outliers >75% will be clipped). Set to 1.0 or 0.0 to disable quantile normalization. A value like 0.75 or 0.85 should be reasonable, it really depends on the input and how many of the values are extreme.",
                },
            ),
            "dim": (
                ("global", "0", "1", "2", "3", "4"),
                {
                    "default": "1",
                    "tooltip": "Controls what dimensions quantile normalization uses. Dimensions start from 0. Image latents have dimensions: batch, channel, row, column. Video latents have dimensions: batch, channel, frame, row, column.",
                },
            ),
            "flatten": (
                "BOOLEAN",
                {
                    "default": True,
                    "tooltip": "Controls whether the noise is flattened before quantile normalization. You can try disabling it but they may have a very strong row/column influence.",
                },
            ),
            "norm_factor": (
                "FLOAT",
                {
                    "default": 1.0,
                    "min": 0.00001,
                    "max": 10000.0,
                    "step": 0.001,
                    "tooltip": "Multiplier on the input noise just before it is clipped to the quantile min/max. Generally should be left at the default.",
                },
            ),
            "norm_power": (
                "FLOAT",
                {
                    "default": 0.5,
                    "min": -10000.0,
                    "max": 10000.0,
                    "step": 0.001,
                    "tooltip": "The absolute value of the noise is raised to this power after it is clipped to the quantile min/max. You can use negative values here, but anything below -0.3 will probably produce pretty strange effects. Generally should be left at the default.",
                },
            ),
            "normalize_noise": (
                "BOOLEAN",
                {
                    "default": False,
                    "tooltip": "Controls whether the noise source is normalized before quantile filtering occurs.",
                },
            ),
            "normalize": (
                ("default", "forced", "disabled"),
                {
                    "default": "disabled",
                    "tooltip": "Controls whether the generated noise is normalized to 1.0 strength after quantile filtering.",
                },
            ),
        }
        return result

    @classmethod
    def get_item_class(cls):
        return noise.QuantileFilteredNoise

    def go(
        self,
        *,
        factor: float,
        quantile: float,
        dim: str,
        flatten: bool,
        norm_power: float,
        norm_factor: float,
        normalize_noise: bool,
        normalize: str,
        custom_noise: object,
    ):
        return super().go(
            factor,
            noise=custom_noise,
            quantile=quantile,
            norm_dim=None if dim == "global" else int(dim),
            norm_flatten=flatten,
            norm_pow=norm_power,
            norm_fac=norm_factor,
            normalize=normalize,
            normalize_noise=normalize_noise,
        )


class SonarAdvancedDistroNoiseNode(SonarCustomNoiseNodeBase):
    DESCRIPTION = "Custom noise type that allows specifying parameters for Distro variants. See: https://pytorch.org/docs/stable/distributions.html"

    @classmethod
    def INPUT_TYPES(cls):
        distro_params = DistroNoiseGenerator.distro_params()
        variants = tuple(sorted(distro_params.keys()))
        combined_params = DistroNoiseGenerator.build_params()

        result = super().INPUT_TYPES()
        result["required"] |= {
            "distribution": (
                variants,
                {
                    "tooltip": "Sets the distribution used for noise generation. See: https://pytorch.org/docs/stable/distributions.html",
                    "default": "uniform",
                },
            ),
            "quantile_norm": (
                "FLOAT",
                {
                    "default": 0.85,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.001,
                    "round": False,
                    "tooltip": "When enabled, will normalize generated noise to this quantile (i.e. 0.75 means outliers >75% will be clipped). Set to 1.0 or 0.0 to disable quantile normalization. A value like 0.75 or 0.85 should be reasonable, it really depends on the distribution and how many of the values are extreme.",
                },
            ),
            "quantile_norm_mode": (
                (
                    "global",
                    "batch",
                    "channel",
                    "batch_row",
                    "batch_col",
                    "nonflat_row",
                    "nonflat_col",
                ),
                {
                    "default": "batch",
                    "tooltip": "Controls what dimensions quantile normalization uses. By default, the noise is flattened first. You can try the nonflat versions but they may have a very strong row/column influence. Only applies when quantile_norm is active.",
                },
            ),
            "result_index": (
                "STRING",
                {
                    "default": "-1",
                    "tooltip": "When noise generation returns a batch of items, it will select the specified index. Negative indexes count from the end. Values outside the valid range will be automatically adjusted. You may enter a space-separated list of values for the case where there might be multiple added batch dimensions. Excess batch dimensions are removed from the end, indexe from result_index are used in order so you may want to enter the indexes in reverse order.\nExample: If your noise has shape (1, 4, 3, 3) and two 2-sized batch dims are added resulting in (1, 4, 3, 3, 2, 2) and you wanted index 0 from the first additional batch dimension and 1 from the second you would use result_index: 1 0",
                },
            ),
        } | {
            k: ("STRING" if isinstance(v["default"], str) else v.get("_ty", "FLOAT"), v)
            for k, v in combined_params.items()
        }
        # print("RESULT:", result)
        return result

    @classmethod
    def get_item_class(cls):
        return noise.AdvancedDistroNoise

    def go(
        self,
        *,
        factor,
        rescale,
        distribution,
        quantile_norm,
        quantile_norm_mode,
        result_index,
        sonar_custom_noise_opt=None,
        **kwargs: dict[str],
    ):
        normdim, normflat = {
            "global": (None, True),
            "batch": (0, True),
            "channel": (1, True),
            "batch_row": (2, True),
            "batch_col": (3, True),
            "nonflat_row": (2, False),
            "nonflat_col": (3, False),
        }.get(quantile_norm_mode, (1, True))
        result_index = tuple(int(v) for v in result_index.split(None))
        return super().go(
            factor,
            rescale=rescale,
            sonar_custom_noise_opt=sonar_custom_noise_opt,
            distro=distribution,
            quantile_norm=quantile_norm,
            quantile_norm_dim=normdim,
            quantile_norm_flatten=normflat,
            result_index=result_index,
            **kwargs,
        )


class SonarWaveletFilteredNoiseNode(
    SonarCustomNoiseNodeBase,
    SonarNormalizeNoiseNodeMixin,
):
    DESCRIPTION = "Custom noise type that allows filtering another custom noise source with wavelets."

    @classmethod
    def INPUT_TYPES(cls):
        result = super().INPUT_TYPES()
        result["required"] |= {
            "normalize_noise": (
                "BOOLEAN",
                {
                    "default": False,
                    "tooltip": "Controls whether the noise source is normalized before wavelet filtering occurs.",
                },
            ),
            "normalize": (
                ("default", "forced", "disabled"),
                {
                    "tooltip": "Controls whether the generated noise is normalized to 1.0 strength. For weird blend modes, you may want to set this to forced.",
                },
            ),
        }
        result["optional"] |= {
            "custom_noise": (
                WILDCARD_NOISE,
                {
                    "tooltip": f"Optional: Custom noise input. If unconnected will default to Gaussian noise.\n{NOISE_INPUT_TYPES_HINT}",
                },
            ),
            "yaml_parameters": (
                "STRING",
                {
                    "tooltip": "Allows specifying custom parameters via YAML. Note: When specifying paramaters this way, there is no error checking.",
                    "placeholder": "# YAML or JSON here",
                    "dynamicPrompts": False,
                    "multiline": True,
                },
            ),
        }
        return result

    @classmethod
    def get_item_class(cls):
        return noise.WaveletFilteredNoise

    def go(
        self,
        *,
        factor,
        rescale,
        normalize,
        normalize_noise,
        custom_noise=None,
        yaml_parameters=None,
        sonar_custom_noise_opt=None,
    ):
        return super().go(
            factor,
            rescale=rescale,
            sonar_custom_noise_opt=sonar_custom_noise_opt,
            normalize=self.get_normalize(normalize),
            normalize_noise=normalize_noise,
            noise=custom_noise,
            yaml_parameters=yaml_parameters,
        )


class CustomNOISE:
    def __init__(
        self,
        custom_noise,
        seed,
        *,
        cpu_noise=True,
        normalize=True,
        multiplier=1.0,
    ):
        self.custom_noise = custom_noise
        self.seed = seed
        self.cpu_noise = cpu_noise
        self.normalize = normalize
        self.multiplier = multiplier

    def _sample_noise(self, latent_image, seed):
        result = self.custom_noise.make_noise_sampler(
            latent_image,
            None,
            None,
            seed=seed,
            cpu=self.cpu_noise,
            normalized=self.normalize,
        )(None, None).to(
            device="cpu",
            dtype=latent_image.dtype,
        )
        if result.layout != latent_image.layout:
            if latent_image.layout == torch.sparse_coo:
                return result.to_sparse()
            errstr = f"Cannot handle latent layout {type(latent_image.layout).__name__}"
            raise NotImplementedError(errstr)
        return result if self.multiplier == 1.0 else result.mul_(self.multiplier)

    def generate_noise(self, input_latent):
        latent_image = input_latent["samples"]
        batch_inds = input_latent.get("batch_index")
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        if self.multiplier == 0.0:
            return torch.zeros(
                latent_image.shape,
                dtype=latent_image.dtype,
                layout=latent_image.layout,
                device="cpu",
            )
        if batch_inds is None:
            return self._sample_noise(latent_image, self.seed)
        unique_inds, inverse_inds = np.unique(batch_inds, return_inverse=True)
        result = []
        batch_size = latent_image.shape[0]
        for idx in range(unique_inds[-1] + 1):
            noise = self._sample_noise(
                latent_image[idx % batch_size].unsqueeze(0),
                self.seed + idx,
            )
            if idx in unique_inds:
                result.append(noise)
        return torch.cat(tuple(result[i] for i in inverse_inds), axis=0)


class SonarToComfyNOISENode(metaclass=IntegratedNode):
    DESCRIPTION = "Allows converting SONAR_CUSTOM_NOISE to NOISE (used by SamplerCustomAdvanced and possibly other custom samplers). NOTE: Does not work with noise types that depend on sigma (Brownian, ScheduledNoise, etc)."
    RETURN_TYPES = ("NOISE",)
    CATEGORY = "sampling/custom_sampling/noise"
    FUNCTION = "go"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "custom_noise": (
                    WILDCARD_NOISE,
                    {
                        "tooltip": f"Custom noise type to convert.\n{NOISE_INPUT_TYPES_HINT}",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "tooltip": "Seed to use for generated noise.",
                    },
                ),
                "cpu_noise": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Controls whether noise is generated on CPU or GPU.",
                    },
                ),
                "normalize": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Controls whether generated noise is normalized to 1.0 strength.",
                    },
                ),
                "multiplier": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "step": 0.001,
                        "min": -1000.0,
                        "max": 1000.0,
                        "round": False,
                        "tooltip": "Simple multiplier applied to noise after all other scaling and normalization effects. If set to 0, no noise will be generated (same as disabling noise).",
                    },
                ),
            },
        }

    @classmethod
    def go(cls, *, custom_noise, seed, cpu_noise=True, normalize=True, multiplier=1.0):
        return (
            CustomNOISE(
                custom_noise,
                seed,
                cpu_noise=cpu_noise,
                normalize=normalize,
                multiplier=multiplier,
            ),
        )


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


class SamplerNodeConfigOverride(metaclass=IntegratedNode):
    DESCRIPTION = "Allows overriding paramaters for a SAMPLER. Only parameters that particular sampler supports will be applied, so for example setting ETA will have no effect for non-ancestral Euler."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampler": ("SAMPLER",),
                "eta": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "step": 0.01,
                        "max": 1000.0,
                        "round": False,
                        "tooltip": "Basically controls the ancestralness of the sampler. When set to 0, you will get a non-ancestral (or SDE) sampler.",
                    },
                ),
                "s_noise": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "step": 0.01,
                        "min": -1000.0,
                        "max": 1000.0,
                        "round": False,
                        "tooltip": "Multiplier for noise added during ancestral or SDE sampling.",
                    },
                ),
                "s_churn": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "step": 0.01,
                        "min": -1000.0,
                        "max": 1000.0,
                        "round": False,
                        "tooltip": "Churn was the predececessor of ETA. Only used by a few types of samplers (notably Euler non-ancestral). Not used by any ancestral or SDE samplers.",
                    },
                ),
                "r": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "step": 0.01,
                        "min": -1000.0,
                        "max": 1000.0,
                        "round": False,
                        "tooltip": "Used by dpmpp_sde.",
                    },
                ),
                "sde_solver": (
                    ("midpoint", "heun"),
                    {
                        "tooltip": "Solver used by dpmpp_2m_sde.",
                    },
                ),
                "cpu_noise": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Controls whether noise is generated on CPU or GPU.",
                    },
                ),
                "normalize": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Controls whether generated noise is normalized to 1.0 strength.",
                    },
                ),
            },
            "optional": {
                "noise_type": (
                    ("DEFAULT", *NoiseType.get_names()),
                    {
                        "default": "DEFAULT",
                        "tooltip": "Noise type used during ancestral or SDE sampling. Leave blank to use the default for the attached sampler. Only used when the custom noise input is not connected.",
                    },
                ),
                "custom_noise_opt": (
                    WILDCARD_NOISE,
                    {
                        "tooltip": f"Optional input for custom noise used during ancestral or SDE sampling. When connected, the built-in noise_type selector is ignored.\n{NOISE_INPUT_TYPES_HINT}",
                    },
                ),
                "yaml_parameters": (
                    "STRING",
                    {
                        "tooltip": "Allows specifying custom parameters via YAML. Note: When specifying paramaters this way, there is no error checking.",
                        "placeholder": "# YAML or JSON here",
                        "dynamicPrompts": False,
                        "multiline": True,
                    },
                ),
            },
        }

    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(
        self,
        *,
        sampler,
        eta,
        s_noise,
        s_churn,
        r,
        sde_solver,
        cpu_noise=True,
        noise_type=None,
        custom_noise_opt=None,
        normalize=True,
        yaml_parameters="",
    ):
        sampler_kwargs = {
            "s_noise": s_noise,
            "eta": eta,
            "s_churn": s_churn,
            "r": r,
            "solver_type": sde_solver,
        }
        if yaml_parameters:
            extra_params = yaml.safe_load(yaml_parameters)
            if extra_params is None:
                pass
            elif not isinstance(extra_params, dict):
                raise ValueError(
                    "SamplerConfigOverride: yaml_parameters must either be null or an object",
                )
            else:
                sampler_kwargs |= extra_params
        return (
            samplers.KSAMPLER(
                self.sampler_function,
                extra_options=sampler.extra_options
                | {
                    "override_sampler_cfg": {
                        "sampler": sampler,
                        "noise_type": NoiseType[noise_type.upper()]
                        if noise_type not in {None, "DEFAULT"}
                        else None,
                        "custom_noise": custom_noise_opt,
                        "sampler_kwargs": sampler_kwargs,
                        "cpu_noise": cpu_noise,
                        "normalize": normalize,
                    },
                },
                inpaint_options=sampler.inpaint_options | {},
            ),
        )

    @classmethod
    def sampler_function(
        cls,
        model,
        x,
        sigmas,
        *args: list[Any],
        override_sampler_cfg: dict[str, Any] | None = None,
        noise_sampler: Callable | None = None,
        extra_args: dict[str, Any] | None = None,
        **kwargs: dict[str, Any],
    ):
        if not override_sampler_cfg:
            raise ValueError("Override sampler config missing!")
        if extra_args is None:
            extra_args = {}
        cfg = override_sampler_cfg
        sampler, sampler_kwargs, noise_type, custom_noise, cpu, normalize = (
            cfg["sampler"],
            cfg["sampler_kwargs"],
            cfg.get("noise_type"),
            cfg.get("custom_noise"),
            cfg.get("cpu_noise", True),
            cfg.get("normalize", True),
        )
        sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
        sig = inspect.signature(sampler.sampler_function)
        params = sig.parameters
        if "noise_sampler" in params:
            seed = extra_args.get("seed")
            if custom_noise is not None:
                noise_sampler = custom_noise.make_noise_sampler(
                    x,
                    sigma_min,
                    sigma_max,
                    seed=seed,
                    cpu=cpu,
                    normalized=normalize,
                )
            elif noise_type is not None:
                noise_sampler = noise.get_noise_sampler(
                    noise_type,
                    x,
                    sigma_min,
                    sigma_max,
                    seed=seed,
                    cpu=cpu,
                    normalized=normalize,
                )
        kwargs |= {k: v for k, v in sampler_kwargs.items() if k in params}
        if "noise_sampler" in params:
            kwargs["noise_sampler"] = noise_sampler
        return sampler.sampler_function(
            model,
            x,
            sigmas,
            *args,
            extra_args=extra_args,
            **kwargs,
        )


NODE_DISPLAY_NAME_MAPPINGS = {}

NODE_CLASS_MAPPINGS = {
    "SamplerSonarEuler": SamplerNodeSonarEuler,
    "SamplerSonarEulerA": SamplerNodeSonarEulerAncestral,
    "SamplerSonarDPMPPSDE": SamplerNodeSonarDPMPPSDE,
    "SonarGuidanceConfig": GuidanceConfigNode,
    "SamplerConfigOverride": SamplerNodeConfigOverride,
    "NoisyLatentLike": NoisyLatentLikeNode,
    "SonarAdvancedPyramidNoise": SonarAdvancedPyramidNoiseNode,
    "SonarAdvanced1fNoise": SonarAdvanced1fNoiseNode,
    "SonarAdvancedPowerLawNoise": SonarAdvancedPowerLawNoiseNode,
    "SonarAdvancedCollatzNoise": SonarAdvancedCollatzNoiseNode,
    "SonarAdvancedDistroNoise": SonarAdvancedDistroNoiseNode,
    "SonarCustomNoise": SonarCustomNoiseNode,
    "SonarCustomNoiseAdv": SonarCustomNoiseAdvNode,
    "SonarCompositeNoise": SonarCompositeNoiseNode,
    "SonarModulatedNoise": SonarModulatedNoiseNode,
    "SonarRepeatedNoise": SonarRepeatedNoiseNode,
    "SonarScheduledNoise": SonarScheduledNoiseNode,
    "SonarGuidedNoise": SonarGuidedNoiseNode,
    "SonarRandomNoise": SonarRandomNoiseNode,
    "SonarChannelNoise": SonarChannelNoiseNode,
    "SonarBlendedNoise": SonarBlendedNoiseNode,
    "SonarResizedNoise": SonarResizedNoiseNode,
    "SonarWaveletFilteredNoise": SonarWaveletFilteredNoiseNode,
    "SonarQuantileFilteredNoise": SonarQuantileFilteredNoiseNode,
    "SONAR_CUSTOM_NOISE to NOISE": SonarToComfyNOISENode,
}


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
            "blend_mode": (("simple_add", *utils.BLENDING_MODES.keys()),),
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
