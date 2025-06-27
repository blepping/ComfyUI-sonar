# ruff: noqa: EM102

from __future__ import annotations

import abc
import functools
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


class SonarNoiseImageNode(metaclass=IntegratedNode):
    DESCRIPTION = "Allows adding noise to an image or generating images full of noise."
    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "image"

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
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "Image noise will be added to.",
                    },
                ),
                "noise_min": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "step": 0.001,
                        "min": -1000.0,
                        "max": 1000.0,
                        "round": False,
                        "tooltip": "Generated noise will be normalized to have values between noise_min and noise_max. If you set them both to the same value then this disables normalization.",
                    },
                ),
                "noise_max": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "step": 0.001,
                        "min": -1000.0,
                        "max": 1000.0,
                        "round": False,
                        "tooltip": "Generated noise will be normalized to have values between noise_min and noise_max. If you set them both to the same value then this disables normalization.",
                    },
                ),
                "noise_multiplier": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "step": 0.001,
                        "min": -1000.0,
                        "max": 1000.0,
                        "round": False,
                        "tooltip": "Multiplier for the strength of the generated noise. This is performed after noise_min/max scaling.",
                    },
                ),
                "channel_mode": (
                    (
                        "RGB",
                        "RGBA",
                        "R",
                        "G",
                        "B",
                        "A",
                        "RA",
                        "GA",
                        "BA",
                        "RG",
                        "RB",
                        "GB",
                        "RGA",
                        "RBA",
                        "GBA",
                    ),
                    {
                        "default": "RGB",
                        "tooltip": "RGBA will also add noise to the alpha channel as well if it exists. Only used for 3 or 4 channel images, for other numbers of channels (i.e. one channel) then all channels will be targeted.",
                    },
                ),
                "blend_mode": (
                    ("simple_add", *utils.BLENDING_MODES.keys()),
                    {
                        "default": "simple_add",
                        "tooltip": "Controls how the generated noise is combined with the image. simple_add just adds it and blend_strength is ignored in that case.",
                    },
                ),
                "blend_strength": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "step": 0.001,
                        "min": -1000.0,
                        "max": 1000.0,
                        "round": False,
                        "tooltip": "Multiplier for the strength of the generated noise.",
                    },
                ),
                "overflow_mode": (
                    ("clamp", "rescale"),
                    {
                        "default": "clamp",
                        "tooltip": "When set to clamp, values above/below 0, 1 will be set to those values. When set to rescale, the image values will be rescaled such that the minimum value is 0 and the maximum is 1.",
                    },
                ),
                "greyscale_mode": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "When enabled, generated noise will be averaged so the same amount value is added to all specified channels.",
                    },
                ),
                "pure_noise_mode": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "When enabled, the original image is only used for its shape and you will be adding noise to an image full of zeros (black), suitable for creating pure noise images.",
                    },
                ),
                "dtype": (
                    ("default", "float32", "float64", "float16", "bfloat16"),
                    {
                        "default": "default",
                        "tooltip": "When set to default it will use the same type as the input tensor (probably float32). You can manually set the dtype if you want, though it likely isn't going to matter. Using dtypes with limited range (float16, bfloat16) isn't recommended.",
                    },
                ),
                "cpu_noise": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Controls whether noise will be generated on GPU or CPU.",
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
            },
        }

    @classmethod
    def go(  # noqa: PLR0914
        cls,
        *,
        noise_type: str,
        seed: int,
        image: torch.Tensor,
        noise_multiplier: float,
        noise_min: float,
        noise_max: float,
        channel_mode: str,
        blend_mode: str,
        blend_strength: float,
        overflow_mode: str,
        greyscale_mode: bool,
        dtype: str,
        pure_noise_mode: bool,
        cpu_noise: bool,
        normalize: bool,
        custom_noise_opt: object | None = None,
    ):
        sigma_min, sigma_max, sigma, sigma_next = (None,) * 4
        orig_image = image = (
            torch.zeros_like(image) if pure_noise_mode else image.detach().clone()
        )
        if image.ndim == 3:
            image = image.unsqueeze(0)
        elif image.ndim != 4:
            raise ValueError(
                f"Expected image tensor with 3 or 4 dimensions, got {image.ndim}",
            )
        blend_function = (
            utils.BLENDING_MODES[blend_mode]
            if blend_mode != "simple_add"
            else lambda a, b, _t: a + b
        )
        if noise_min > noise_max:
            noise_min, noise_max = noise_max, noise_min
        image = image.movedim(-1, 1)
        channels = image.shape[1]
        channel_map = {"R": 0, "B": 1, "G": 2, "A": 3}
        channel_mode = channel_mode.upper()
        if channels == 3 or channels == 4:  # noqa: PLR1714
            channel_targets = tuple(
                channel_map[c]
                for c in "RGBA"
                if c in channel_mode and channel_map[c] < channels
            )
        else:
            channel_targets = tuple(range(channels))
        want_device = (
            torch.device("cpu") if cpu_noise else model_management.get_torch_device()
        )
        image = image.to(
            device=want_device,
            dtype={
                "float32": torch.float32,
                "float64": torch.float64,
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
            }.get(
                dtype,
                image.dtype,
            ),
        )
        pyrandst = random.getstate()
        randst = torch.random.get_rng_state()
        try:
            random.seed(seed)
            torch.random.manual_seed(seed)
            if custom_noise_opt is not None:
                ns = custom_noise_opt.make_noise_sampler(
                    image,
                    sigma_min=sigma_min,
                    sigma_max=sigma_max,
                    seed=seed,
                    cpu=cpu_noise,
                    normalized=normalize,
                )
            else:
                ns = noise.get_noise_sampler(
                    NoiseType[noise_type.upper()],
                    image,
                    sigma_min,
                    sigma_max,
                    seed=seed,
                    cpu=cpu_noise,
                    normalized=normalize,
                )
            result = ns(sigma, sigma_next)
        finally:
            torch.random.set_rng_state(randst)
            random.setstate(pyrandst)
        del ns
        result = utils.scale_noise(result, normalized=True)
        if greyscale_mode:
            result = result.mean(dim=1, keepdim=True).expand(image.shape).contiguous()
        if noise_max != 0 and noise_min != noise_max:  # noqa: PLR1714
            result = utils.normalize_to_scale(result, noise_min, noise_max)
        result *= noise_multiplier
        image[:, channel_targets, ...] = blend_function(
            image[:, channel_targets, ...],
            result[:, channel_targets, ...],
            blend_strength,
        )
        if overflow_mode == "rescale":
            image = utils.normalize_to_scale(image, 0.0, 1.0)
        else:
            image = image.clip_(0, 1)
        image = image.movedim(1, -1).to(
            device=orig_image.device,
            dtype=orig_image.dtype,
        )
        return (image,)


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
    DESCRIPTION = "Custom noise type that allows specifying parameters for Collatz noise. Very experimental, also very slow. It might just about work as initial noise with non-ancestral sampling but if you get weird results I recommend mixing it with other noise types or possibly using ancestral/SDE sampling."

    @classmethod
    def INPUT_TYPES(cls):
        result = super().INPUT_TYPES()
        result["required"] |= {
            "adjust_scale": (
                "BOOLEAN",
                {
                    "default": False,
                    "tooltip": "When enabled, the output will be normalized to values between -1 and 1 using the last two dimensions (if there are four or more), otherwise dimensions after the first.",
                },
            ),
            "chain_length": (
                "STRING",
                {
                    "default": "1, 1, 2, 2, 3, 3",
                    "tooltip": "Comma-separated list of chain lengths. Cannot be empty. Iterations will cycle through the list and wrap. Controls the length of Collatz chains. Note: Using a high chain length may be very slow, especially if combined with many iterations.",
                },
            ),
            "chain_offset": (
                "INT",
                {
                    "default": 5,
                    "min": 0,
                    "max": 10000,
                    "tooltip": "Uses values starting at the specified offset. Note: This entails generating chains of length chain_length + chain_offset, which may be quite slow if you use high values.",
                },
            ),
            "iterations": (
                "INT",
                {
                    "default": 10,
                    "min": 1,
                    "max": 10000,
                    "tooltip": "Number of iterations to run. Warning: Collatz noise (my implementation, anyway) is EXTREMELY slow.",
                },
            ),
            "iteration_sign_flipping": (
                "BOOLEAN",
                {
                    "default": True,
                    "tooltip": "Controls whether we cycle between flipping the sign on the output from each iteration. May average out weirdness... Or make stuff weirder.",
                },
            ),
            "rmin": (
                "FLOAT",
                {
                    "default": -8000.0,
                    "min": -100000.0,
                    "max": 100000.0,
                    "tooltip": "Minimum value a chain can start with. Going as low as -9500 should be safe with float32.",
                },
            ),
            "rmax": (
                "FLOAT",
                {
                    "default": 8000.0,
                    "min": -100000.0,
                    "max": 100000.0,
                    "tooltip": "Maximum value a chain can start with. I don't recommend going over 9500 if you are using the float32 dtype here as that is where the Collatz chain starts to reach values that can't be accurately represented.",
                },
            ),
            "dims": (
                "STRING",
                {
                    "default": "-1, -1, -2, -2",
                    "tooltip": "Comma-separated list of dimensions. Cannot be empty. May be negative to count from the end of the list. Iterations will cycle through the list and wrap.",
                },
            ),
            "flatten": (
                "BOOLEAN",
                {
                    "default": False,
                    "tooltip": "Controls whether dimensions past the current one selected from the dims parameter will get flattened.",
                },
            ),
            "output_mode": (
                (
                    "values",
                    "ratios",
                    "mults",
                    "adds",
                    "seed_x_mults",
                    "seed_x_adds",
                    "noise_x_ratios",
                    "noise_x_mults",
                    "noise_x_adds",
                ),
                {
                    "default": "values",
                },
            ),
            "quantile": (
                "FLOAT",
                {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "tooltip": "The initial output of each iteration will be run through quantile normalization. Setting the parameter to 0 or 1 will disable quantile normalization.",
                },
            ),
            "quantile_strategy": (
                tuple(utils.quantile_handlers.keys()),
                {
                    "default": "clamp",
                    "tooltip": "Determines how to treat outliers. zero and reverse_zero modes are only useful if you're going to do something like add the result to some other noise. zero will return zero for anything outside the quantile range, reverse_zero only _keeps_ the outliers and zeros everything else.",
                },
            ),
            "noise_dtype": (
                ("float32", "float64", "float16", "bfloat16"),
                {
                    "default": "float32",
                    "tooltip": "Generally should be left at the default. Only float32 and float64 will work if you have quantile normalization enabled.",
                },
            ),
            "even_multiplier": (
                "FLOAT",
                {
                    "default": 0.5,
                    "min": -10000.0,
                    "max": 1000.0,
                    "tooltip": "Multiplier to use when the previous link in the chain is even. Collatz uses 0.5 (divides by two) here.",
                },
            ),
            "even_addition": (
                "FLOAT",
                {
                    "default": 0.0,
                    "min": -10000.0,
                    "max": 1000.0,
                    "tooltip": "Value to add when the previous link in the chain is even. Collatz uses 0 here.",
                },
            ),
            "odd_multiplier": (
                "FLOAT",
                {
                    "default": 3.0,
                    "min": -10000.0,
                    "max": 1000.0,
                    "tooltip": "Multiplier to use when the previous link in the chain is odd. Collatz uses 3 here.",
                },
            ),
            "odd_addition": (
                "FLOAT",
                {
                    "default": 1.0,
                    "min": -10000.0,
                    "max": 1000.0,
                    "tooltip": "Value to add when the previous link in the chain is odd. Collatz uses 1 here.",
                },
            ),
            "integer_math": (
                "BOOLEAN",
                {
                    "default": True,
                    "tooltip": "Controls whether the results during chain generation get truncated to an integer value or not. Should be enabled if you actually want to generate accurate Collatz chains.",
                },
            ),
            "add_preserves_sign": (
                "BOOLEAN",
                {
                    "default": True,
                    "tooltip": "Controls whether additions use the same sign as the item they're being added to.",
                },
            ),
            "break_loops": (
                "BOOLEAN",
                {
                    "default": True,
                    "tooltip": "Controls whether the chain resets back to the seed value once it reaches 1 or 0. Generally should be left enabled, otherwise the chain will oscillate between only a few values for the rest of the length (at least with the Collatz rules).",
                },
            ),
            "seed_mode": (
                ("default", "force_odd", "force_even"),
                {
                    "default": "default",
                    "tooltip": "Default mode just uses whatever the original seed value was. force_odd/force_even will force it to the specified parity by adding one if it doesn't match. Starting from odd seeds might result in longer chains. Enabling the force modes may cause the initial seeds to exceed rmax by one.",
                },
            ),
        }
        result["optional"] |= {
            "seed_custom_noise": (
                WILDCARD_NOISE,
                {
                    "tooltip": f"Optional custom noise to use for initial values for Collatz chains. May be slow as it will generate noise according to the original input size and then crop it. Does this noise type have enough warnings about it being slow? Yeah. Connecting something here will probably make it even slower!\n{NOISE_INPUT_TYPES_HINT}",
                },
            ),
            "mix_custom_noise": (
                WILDCARD_NOISE,
                {
                    "tooltip": f"Optional custom noise to use with the output modes starting with 'noise'.\n{NOISE_INPUT_TYPES_HINT}",
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
        factor: float,
        rescale: float,
        adjust_scale: bool,
        iteration_sign_flipping: bool,
        chain_length: int,
        iterations: int,
        rmin: float,
        rmax: float,
        flatten: bool,
        dims: str,
        output_mode: str,
        noise_dtype: str,
        quantile: float,
        quantile_strategy: str,
        integer_math: bool,
        add_preserves_sign: bool,
        even_multiplier: float,
        even_addition: float,
        odd_multiplier: float,
        odd_addition: float,
        chain_offset: int,
        seed_mode: str,
        break_loops: bool,
        seed_custom_noise: object | None = None,
        mix_custom_noise: object | None = None,
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
            iteration_sign_flipping=iteration_sign_flipping,
            chain_length=tuple(int(i) for i in chain_length.split(",")),
            iterations=iterations,
            rmin=rmin,
            rmax=rmax,
            flatten=flatten,
            dims=dims,
            output_mode=output_mode,
            quantile=quantile,
            quantile_strategy=quantile_strategy,
            integer_math=integer_math,
            add_preserves_sign=add_preserves_sign,
            even_multiplier=even_multiplier,
            even_addition=even_addition,
            odd_multiplier=odd_multiplier,
            odd_addition=odd_addition,
            chain_offset=chain_offset,
            break_loops=break_loops,
            seed_mode=seed_mode,
            noise_dtype={
                "float32": torch.float32,
                "float64": torch.float64,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }.get(noise_dtype, torch.float32),
            seed_custom_noise=seed_custom_noise,
            mix_custom_noise=mix_custom_noise,
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
            "strategy": (
                tuple(utils.quantile_handlers.keys()),
                {
                    "default": "clamp",
                    "tooltip": "Determines how to treat outliers. zero and reverse_zero modes are only useful if you're going to do something like add the result to some other noise. zero will return zero for anything outside the quantile range, reverse_zero only _keeps_ the outliers and zeros everything else.",
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
        strategy: str,
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
            strategy=strategy,
            normalize_noise=normalize_noise,
        )


class SonarShuffledNoiseNode(SonarCustomNoiseNodeBase):
    DESCRIPTION = "Custom noise type that allows shuffling noise along some dimension"

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
            "dims": (
                "STRING",
                {
                    "default": "-1",
                    "tooltip": "Comma separated list of dimensions to shuffle. May be negative to count from the end.",
                },
            ),
            "flatten": (
                "BOOLEAN",
                {
                    "default": False,
                    "tooltip": "Controls whether to flatten starting from the dimension before the shuffle operation. May be slow as this requires flattening and then reshaping the tensor back to the correct shape. Flattening will occur between the lowest and highest dimension in the list, other dimensions will be ignored. If they are the same, then it will just flatten from the lowest dimension.",
                },
            ),
            "percentage": (
                "FLOAT",
                {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "tooltip": "Percentage of elements to shuffle in the specified dimensions.",
                },
            ),
        }
        return result

    @classmethod
    def get_item_class(cls):
        return noise.ShuffledNoise

    def go(
        self,
        *,
        factor: float,
        dims: str,
        flatten: bool,
        percentage: float,
        custom_noise: object,
    ):
        dims = dims.strip()
        dims = () if not dims else tuple(int(i) for i in dims.split(","))
        return super().go(
            factor,
            noise=custom_noise,
            dims=dims,
            flatten=flatten,
            percentage=percentage,
        )


class SonarPatternBreakNoiseNode(SonarCustomNoiseNodeBase):
    DESCRIPTION = "Custom noise type that allows breaking patterns in the noise with configurable strength"

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
            "detail_level": (
                "FLOAT",
                {
                    "default": 0.0,
                    "min": -10000.0,
                    "max": 10000.0,
                    "tooltip": "Controls the detail level of the noise when break_pattern is non-zero. No effect when strength is 0.",
                },
            ),
            "blend_mode": (
                tuple(utils.BLENDING_MODES.keys()),
                {
                    "default": "lerp",
                    "tooltip": "Function to use for blending original noise with pattern broken noise. If you have ComfyUI-bleh then you will have access to many more blend modes.",
                },
            ),
            "percentage": (
                "FLOAT",
                {
                    "default": 1.0,
                    "min": -10000.0,
                    "max": 10000.0,
                    "tooltip": "Percentage pattern-broken noise to mix with the original noise. Going outside of 0.0 through 1.0 is unlikely to work well with normal blend modes.",
                },
            ),
            "restore_scale": (
                "BOOLEAN",
                {
                    "default": True,
                    "tooltip": "Controls whether the original min/max values get preserved. Not sure which is better, it is slightly slower to do this though.",
                },
            ),
        }
        return result

    @classmethod
    def get_item_class(cls):
        return noise.PatternBreakNoise

    def go(
        self,
        *,
        factor: float,
        blend_mode: str,
        detail_level: float,
        percentage: float,
        restore_scale: bool,
        custom_noise: object,
    ):
        return super().go(
            factor,
            noise=custom_noise,
            blend_mode=blend_mode,
            detail_level=detail_level,
            percentage=percentage,
            restore_scale=restore_scale,
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


class SonarWaveletNoiseNode(
    SonarCustomNoiseNodeBase,
    SonarNormalizeNoiseNodeMixin,
):
    DESCRIPTION = "Custom noise type that allows generating wavelet noise. Very simple explanation of how a single octave works:\n1) Generate some noise.\n2) Scale it down 50%.\n3) Scale it back up to the original size.\n4) Subtract the scaled noise from the original noise.\nScaling the noise down and then back up blurs it, so this is essentially sharpening the noise."

    @classmethod
    def INPUT_TYPES(cls):
        result = super().INPUT_TYPES()
        result["required"] |= {
            "octaves": (
                "INT",
                {
                    "default": 4,
                    "min": -100,
                    "max": 100,
                    "tooltip": "Number of octaves to generate. You can use a negative number here to run the octaves in reverse order though it may produce weird results/not work very well.",
                },
            ),
            "octave_height_factor": (
                "FLOAT",
                {
                    "default": 0.5,
                    "min": 0.001,
                    "max": 10000.0,
                    "tooltip": "Wavelet noise works by scaling noise by this factor in each octave, then scaling it back up to the original size. After that, the scaled noise is subtracted from the original noise.",
                },
            ),
            "octave_width_factor": (
                "FLOAT",
                {
                    "default": 0.5,
                    "min": 0.001,
                    "max": 10000.0,
                    "tooltip": "Wavelet noise works by scaling noise by this factor in each octave, then scaling it back up to the original size. After that, the scaled noise is subtracted from the original noise.",
                },
            ),
            "octave_scale_mode": (
                utils.UPSCALE_METHODS,
                {
                    "tooltip": "Scaling mode used within each octave to produce the scaled noise. By default this will be scaling down that octave's noise.",
                    "default": "adaptive_avg_pool2d",
                },
            ),
            "octave_rescale_mode": (
                utils.UPSCALE_METHODS,
                {
                    "tooltip": "Scaling mode used within each octave to scale the noise back up to that octave's original size.",
                    "default": "bilinear",
                },
            ),
            "post_octave_rescale_mode": (
                utils.UPSCALE_METHODS,
                {
                    "tooltip": "Scaling mode used to scale the output of an octave back up to the actual latent size.",
                    "default": "bilinear",
                },
            ),
            "initial_amplitude": (
                "FLOAT",
                {
                    "default": 1.0,
                    "min": -10000.0,
                    "max": 10000.0,
                    "tooltip": "Basically the strength an octave gets added to the total. This will be scaled by persistance after each octave.",
                },
            ),
            "persistence": (
                "FLOAT",
                {
                    "default": 0.5,
                    "min": -10000.0,
                    "max": 10000.0,
                    "tooltip": "Multiplier applied to amplitude after each octave. 0.5 means the first octave uses initial_amplitude, the second uses half of that and so on.",
                },
            ),
            "height_factor": (
                "FLOAT",
                {
                    "default": 2.0,
                    "min": 0.001,
                    "max": 10000.0,
                    "tooltip": "Scaling factor for height, calculated after each octave. 2.0 means divide by two. Note: It's possible to use values below 1 here but be careful as it's very easy to reach absurd latent sizes with only a few octaves.",
                },
            ),
            "width_factor": (
                "FLOAT",
                {
                    "tooltip": "Scaling factor for width, calculated after each octave. 2.0 means divide by two. Note: It's possible to use values below 1 here but be careful as it's very easy to reach absurd latent sizes with only a few octaves.",
                    "default": 2.0,
                    "min": 0.001,
                    "max": 10000.0,
                },
            ),
            "update_blend": (
                "FLOAT",
                {
                    "tooltip": "Controls how original_noise - scaled_noise is blended with original_noise. The default is to use 100% original_noise - scaled_noise.",
                    "default": 1.0,
                    "min": -10000.0,
                    "max": 10000.0,
                },
            ),
            "update_blend_mode": (
                ("simple_add", *utils.BLENDING_MODES.keys()),
                {
                    "default": "lerp",
                    "tooltip": "Controls how the enhanced noise from each octave is blended with that octave's raw noise. With normal wavelet noise there's no blending and you use 100% enhanced noise.",
                },
            ),
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
                    "tooltip": f"Optional: Custom noise input. If unconnected will default to Gaussian noise. Note: When connected, the noise for all octaves will be generated at the maximum scale and then cropped which may be slow.\n{NOISE_INPUT_TYPES_HINT}",
                },
            ),
        }
        return result

    @classmethod
    def get_item_class(cls):
        return noise.AdvancedWaveletNoise

    def go(
        self,
        *,
        factor,
        rescale,
        normalize,
        octaves: int,
        octave_height_factor: float,
        octave_width_factor: float,
        octave_scale_mode: str,
        octave_rescale_mode: str,
        post_octave_rescale_mode: str,
        initial_amplitude: float,
        persistence: float,
        height_factor: float,
        width_factor: float,
        update_blend: float,
        update_blend_mode: str,
        normalize_noise: bool,
        custom_noise=None,
        sonar_custom_noise_opt=None,
    ):
        if persistence == 0 or initial_amplitude == 0 or octaves == 0:
            raise ValueError(
                "Persistence, initial amplitude and octaves must be non-zero",
            )
        return super().go(
            factor,
            rescale=rescale,
            sonar_custom_noise_opt=sonar_custom_noise_opt,
            octaves=octaves,
            octave_height_factor=octave_height_factor,
            octave_width_factor=octave_width_factor,
            octave_scale_mode=octave_scale_mode,
            octave_rescale_mode=octave_rescale_mode,
            post_octave_rescale_mode=post_octave_rescale_mode,
            initial_amplitude=initial_amplitude,
            persistence=persistence,
            height_factor=height_factor,
            width_factor=width_factor,
            update_blend=update_blend,
            update_blend_function=utils.BLENDING_MODES[update_blend_mode],
            normalize=self.get_normalize(normalize),
            normalize_noise=normalize_noise,
            custom_noise=custom_noise,
        )


class SonarRippleFilteredNoiseNode(
    SonarCustomNoiseNodeBase,
    SonarNormalizeNoiseNodeMixin,
):
    DESCRIPTION = (
        "Custom noise filter that allows applying scaling based on a wave (sin or cos)."
    )

    @classmethod
    def INPUT_TYPES(cls):
        result = super().INPUT_TYPES()
        result["required"] |= {
            "custom_noise": (
                WILDCARD_NOISE,
                {
                    "tooltip": f"Custom noise input. \n{NOISE_INPUT_TYPES_HINT}",
                },
            ),
            "mode": (
                ("sin", "cos", "sin_copysign", "cos_copysign"),
                {
                    "default": "cos",
                    "tooltip": "Function to use for rippling. The copysign variations are not recommended, they will force the noise to the sign of the wave (whether it's above or below the midline) which has an extremely strong effect. If you want to try it, use something like a 1:16 ratio or higher with normal noise.",
                },
            ),
            "dim": (
                "INT",
                {
                    "default": -1,
                    "min": -100,
                    "max": 100,
                    "tooltip": "Dimension to use for the ripple effect. Negative dimensions count from the end where -1 is the last dimension.",
                },
            ),
            "flatten": (
                "BOOLEAN",
                {
                    "default": False,
                    "tooltip": "When enabled, the noise will be flattened starting from (and including) the specified dimension.",
                },
            ),
            "offset": (
                "FLOAT",
                {
                    "default": 0.0,
                    "min": -10000,
                    "max": 10000.0,
                    "tooltip": "Simple addition to the base value used for the wave.",
                },
            ),
            "roll": (
                "FLOAT",
                {
                    "default": 0.0,
                    "min": -10000,
                    "max": 10000.0,
                    "tooltip": "Rolls the wave by this many elements each time the noise generator is called. Negative values roll backward.",
                },
            ),
            "amplitude_high": (
                "FLOAT",
                {
                    "default": 0.25,
                    "min": -10000,
                    "max": 10000.0,
                    "tooltip": "Scale for noise at the highest point of the wave. This adds to the base value (respecting sign). For example, if set to 0.25 you will get noise * 1.25 at that point. It's also possible to use negative values, -0.25 will result in noise * -1.25.",
                },
            ),
            "amplitude_low": (
                "FLOAT",
                {
                    "default": 0.15,
                    "min": -10000,
                    "max": 10000.0,
                    "tooltip": "Scale for noise at the lowest point of the wave. This subtracts from the base value (respecting sign). For example, if set to 0.25 you will get noise * 0.75 at that point. It's also possible to use negative values, -0.25 will result in noise * -0.75.",
                },
            ),
            "period": (
                "FLOAT",
                {
                    "default": 3.0,
                    "min": -10000,
                    "max": 10000.0,
                    "tooltip": "Number of oscillations along the specified dimension.",
                },
            ),
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
        return result

    @classmethod
    def get_item_class(cls):
        return noise.RippleFilteredNoise

    def go(
        self,
        *,
        factor,
        rescale,
        normalize,
        mode: str,
        dim: int,
        flatten: bool,
        offset: float,
        amplitude_high: float,
        amplitude_low: float,
        period: float,
        roll: float,
        normalize_noise: bool,
        custom_noise=None,
        sonar_custom_noise_opt=None,
    ):
        return super().go(
            factor,
            rescale=rescale,
            sonar_custom_noise_opt=sonar_custom_noise_opt,
            mode=mode,
            dim=dim,
            flatten=flatten,
            offset=offset,
            amplitude_high=amplitude_high,
            amplitude_low=amplitude_low,
            period=period,
            roll=roll,
            normalize=self.get_normalize(normalize),
            normalize_noise=normalize_noise,
            noise=custom_noise,
        )


class SonarApplyLatentOperationCFG(metaclass=IntegratedNode):
    DESCRIPTION = "Allows applying a LATENT_OPERATION during sampling. ComfyUI has a few that are builtin and this node pack also includes: SonarLatentOperationQuantileFilter."
    RETURN_TYPES = ("MODEL",)
    CATEGORY = "latent/advanced/operations"

    FUNCTION = "go"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "mode": (
                    (
                        "cond_sub_uncond",
                        "denoised_sub_uncond",
                        "uncond_sub_cond",
                        "denoised",
                        "cond",
                        "uncond",
                        "model_input",
                    ),
                    {
                        "default": "cond_sub_uncond",
                        "tooltip": "cond_sub_uncond is what ComfyUI's latent operations use. The non-sub_uncond modes likely won't work with pred_flip mode enabled. If you have anything but the denoised options selected, this will use pre-CFG, otherwise it will use post-CFG (unless you are using model_input).",
                    },
                ),
                "pred_flip_mode": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Lets you try to apply the latent operation to the noise prediction rather than the image prediction. Doesn't work properly with the non-sub_uncond modes. No real reason it should be better, just something you can try. Note: The noise prediction gets scaled by the sigma first, in case that's useful information.",
                    },
                ),
                "require_uncond": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "When enabled, the operation will be skipped if uncond is unavailable. This will also happen if you choose a mode that requires uncond.",
                    },
                ),
                "start_sigma": (
                    "FLOAT",
                    {
                        "default": -1.0,
                        "min": -1.0,
                        "max": 9999.0,
                        "tooltip": "Sigma when the effect becomes active. You can set a negative value here to use whatever the model's maximum sigma is.",
                    },
                ),
                "end_sigma": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 9999.0,
                    },
                ),
                "blend_mode": (
                    tuple(utils.BLENDING_MODES.keys()),
                    {
                        "default": "lerp",
                        "tooltip": "Controls how the output of the latent operation is blended with the original result.",
                    },
                ),
                "blend_strength": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "step": 0.001,
                        "min": -1000.0,
                        "max": 1000.0,
                        "round": False,
                        "tooltip": "Strength of the blend. For a normal blend mode like LERP, 1.0 means use 100% of the output from the latent operation, 0.0 means use none of it and only the original value. Note: Blending is applied to the final result of the operations, in other words operation_2 sees a full unblended result from operation_1.",
                    },
                ),
                "blend_scale_mode": (
                    (
                        "none",
                        "reverse_sampling",
                        "sampling",
                        "reverse_enabled_range",
                        "enabled_range",
                        "sampling_sin",
                        "enabled_range_sin",
                    ),
                    {
                        "default": "reverse_sampling",
                        "tooltip": "Can be used to scale the blend strength over time. Basically works like blend_strength * scale_factor (see below)\nnone: Just uses the blend_strength you have set.\nreverse_sampling: The opposite of the model sampling percent, so if you're making a new generation, the beginning of sampling will be 1.0 and the end will be 0.0. The recommended option as applying these operations usually works better toward the beginning of sampling.\nsampling: Same as reverse_sampling, except the beginning will be 0.0 and the end will be 1.0.\nreverse_enabled_range: Flipped percentage of the range between start_sigma and end_sigma.\nenabled_range: Percentage of the range between start_sigma and end_sigma.\nsampling_sin: Uses the sampling percentage with the sine function such that blend_strength will hit the peak value in the middle of the range.\nenabled_range_sin: Similar to sampling_sin except it applies to the percentage of the enabled range.",
                    },
                ),
                "blend_scale_offset": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -1.0,
                        "max": 1.0,
                        "tooltip": "Only applies when blend_scale_mode is not none. Adds the offset to the calculated percentage and then clamps it to be between blend_scale_min and blend_scale_max.",
                    },
                ),
                "blend_scale_min": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "tooltip": "Only applies when blend_scale_mode is not none. Minimum value for the blend scale percentage.",
                    },
                ),
                "blend_scale_max": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "tooltip": "Only applies when blend_scale_mode is not none. Maximum value for the blend scale percentage.",
                    },
                ),
            },
            "optional": {
                "operation_1": (
                    "LATENT_OPERATION",
                    {
                        "tooltip": "Optional LATENT_OPERATION. The operations will be applied in sequence.",
                    },
                ),
                "operation_2": (
                    "LATENT_OPERATION",
                    {
                        "tooltip": "Optional LATENT_OPERATION. The operations will be applied in sequence.",
                    },
                ),
                "operation_3": (
                    "LATENT_OPERATION",
                    {
                        "tooltip": "Optional LATENT_OPERATION. The operations will be applied in sequence.",
                    },
                ),
                "operation_4": (
                    "LATENT_OPERATION",
                    {
                        "tooltip": "Optional LATENT_OPERATION. The operations will be applied in sequence.",
                    },
                ),
                "operation_5": (
                    "LATENT_OPERATION",
                    {
                        "tooltip": "Optional LATENT_OPERATION. The operations will be applied in sequence.",
                    },
                ),
            },
        }

    @staticmethod
    def get_blend_scaling(
        *,
        model_sampling: object,
        scale_mode: str,
        sigma: float,
        sigma_t_max: torch.Tensor,
        start_sigma: float,
        end_sigma: float,
        offset: float,
        min_pct: float,
        max_pct: float,
    ) -> float | torch.Tensor:
        if scale_mode == "none":
            return 1.0
        if scale_mode in {"sampling", "sampling_sin", "reverse_sampling"}:
            rev_sampling_pct = (
                (model_sampling.timestep(sigma_t_max) / 999).clamp(0, 1).detach().item()
            )
            result = (
                1.0 - rev_sampling_pct if scale_mode == "sampling" else rev_sampling_pct
            )
        elif scale_mode in {
            "enabled_range",
            "enabled_range_sin",
            "reverse_enabled_range",
        }:
            rev_range_pct = (sigma - end_sigma) / (start_sigma - end_sigma)
            result = (
                1.0 - rev_range_pct if scale_mode == "enabled_range" else rev_range_pct
            )
        else:
            raise ValueError("Bad blend_scale_mode")
        if scale_mode.endswith("_sin"):
            result = math.sin(result * math.pi)
        return max(min_pct, min(result + offset, max_pct))

    @classmethod
    def go(
        cls,
        *,
        model,
        mode: str,
        pred_flip_mode: bool,
        require_uncond: bool,
        start_sigma: float,
        end_sigma: float,
        blend_mode: str,
        blend_strength: float,
        blend_scale_mode: str,
        blend_scale_offset: float,
        blend_scale_min: float,
        blend_scale_max: float,
        operation_1=None,
        operation_2=None,
        operation_3=None,
        operation_4=None,
        operation_5=None,
    ) -> tuple:
        if mode == "model_input":
            if require_uncond:
                raise ValueError(
                    "require_uncond does not make sense for the model_input mode.",
                )
            if pred_flip_mode:
                raise ValueError(
                    "pred_flip does not make sense for the model_input mode.",
                )
        model = model.clone()
        operations = tuple(
            o
            for o in (operation_1, operation_2, operation_3, operation_4, operation_5)
            if o is not None
        )
        if not operations:
            return (model,)
        ms = model.get_model_object("model_sampling")
        post_cfg_mode = mode in {"denoised", "denoised_sub_uncond"}
        blend_function = utils.BLENDING_MODES[blend_mode]
        sigma_max, sigma_min = (
            ms.sigma_max.detach().item(),
            ms.sigma_min.detach().item(),
        )
        if start_sigma < 0:
            start_sigma = sigma_max
        start_sigma = max(sigma_min, min(sigma_max, start_sigma))
        end_sigma = max(sigma_min, min(sigma_max, end_sigma))
        if end_sigma > start_sigma:
            start_sigma, end_sigma = end_sigma, start_sigma
        if start_sigma == end_sigma:
            blend_scale_mode = "none"
        orig_mode = mode

        def patch(args: dict) -> torch.Tensor:
            nonlocal mode

            x = args["input"]
            sigma_t = args["sigma"]
            sigma_t_max = sigma_t.max()
            if sigma_t.numel() > 1:
                shape_pad = (1,) * (x.ndim - sigma_t.ndim)
                sigma_t = sigma_t.reshape(sigma_t.shape[0], *shape_pad)
            sigma = sigma_t_max.detach().item()
            enabled = end_sigma <= sigma <= start_sigma
            conds_out = args.get("conds_out", ())
            uncond = (
                args.get("uncond_denoised")
                if post_cfg_mode
                else (conds_out[1] if len(conds_out) > 1 else None)
            )
            if uncond is None and (
                require_uncond
                or mode in {"uncond", "uncond_sub_cond", "denoised_sub_uncond"}
            ):
                enabled = False
            if not enabled:
                if mode == "model_input":
                    return x
                return args["denoised"] if post_cfg_mode else conds_out
            cond = conds_out[0] if not post_cfg_mode and len(conds_out) else None
            if uncond is None and mode.endswith("_sub_uncond"):
                mode = orig_mode.split("_", 1)[0]
            else:
                mode = orig_mode
            if mode == "model_input":
                t1 = x
                t2 = None
            elif mode in {"cond", "cond_sub_uncond"}:
                t1 = cond
                t2 = uncond if mode == "cond_sub_uncond" else None
            elif mode in {"uncond", "uncond_sub_cond"}:
                t1 = uncond
                t2 = cond if mode == "uncond_sub_cond" else None
            else:
                t1 = args["denoised"]
                t2 = uncond if mode == "denoised_sub_uncond" else None
            t1_orig = t1
            if pred_flip_mode:
                t1 = (x - t1) / sigma_t
                if t2 is not None:
                    t2 = (x - t2) / sigma_t
            curr_blend = blend_strength * cls.get_blend_scaling(
                scale_mode=blend_scale_mode,
                offset=blend_scale_offset,
                min_pct=blend_scale_min,
                max_pct=blend_scale_max,
                model_sampling=args["model"].model_sampling,
                start_sigma=start_sigma,
                end_sigma=end_sigma,
                sigma=max(sigma_min, min(sigma, sigma_max)),
                sigma_t_max=sigma_t_max.clamp(sigma_min, sigma_max),
            )
            result = t1 - t2 if t2 is not None else t1.clone()
            for operation in operations:
                result = operation(result)
            if t2 is not None:
                result += t2
            if pred_flip_mode:
                result = x - sigma_t * result
            if curr_blend != 1:
                result = blend_function(t1_orig, result, curr_blend)
            if post_cfg_mode or mode == "model_input":
                return result
            conds_out = conds_out.copy()
            conds_out[0 if mode.startswith("cond") else 1] = result
            return conds_out

        if post_cfg_mode:
            model.set_model_sampler_post_cfg_function(patch)
        elif mode == "model_input":

            def patch_wrapper(apply_model, args: dict) -> torch.Tensor:
                timestep = args["timestep"]
                patch_args = args | {"sigma": timestep, "model": model.model}
                return apply_model(patch(patch_args), timestep, **args["c"])

            model.set_model_unet_function_wrapper(patch_wrapper)
        else:
            model.set_model_sampler_pre_cfg_function(patch)
        return (model,)


class SonarLatentOperationQuantileFilter(SonarQuantileFilteredNoiseNode):
    DESCRIPTION = "Allows applying a quantile normalization function to the latent during sampling. Can be used with Sonar SonarApplyLatentOperationCFG. The just copies most of the parameters from the other quantile normalization node where it talks to 'noise', this will apply to whatever you're applying the latent operation to (denoised, uncond, etc)."
    RETURN_TYPES = ("LATENT_OPERATION",)
    CATEGORY = "latent/advanced/operations"

    @classmethod
    def INPUT_TYPES(cls):
        result = super().INPUT_TYPES()
        result.pop("optional", None)
        reqparams = result["required"]
        for k in ("custom_noise", "normalize", "normalize_noise", "factor"):
            reqparams.pop(k, None)
        return result

    @classmethod
    def go(
        cls,
        *,
        quantile: float,
        dim: str,
        flatten: bool,
        norm_power: float,
        norm_factor: float,
        strategy: str,
    ):
        def operation(latent: torch.Tensor, **_kwargs: dict) -> torch.Tensor:
            return utils.quantile_normalize(
                latent,
                quantile=quantile,
                dim=None if dim == "global" else int(dim),
                flatten=flatten,
                nq_fac=norm_factor,
                pow_fac=norm_power,
                strategy=strategy,
            )

        return (operation,)


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
        sampler_function = functools.update_wrapper(
            functools.partial(
                self.sampler_function,
                override_sampler_cfg={
                    "sampler": sampler,
                    "noise_type": NoiseType[noise_type.upper()]
                    if noise_type not in {None, "DEFAULT"}
                    else None,
                    "custom_noise": custom_noise_opt,
                    "sampler_kwargs": sampler_kwargs,
                    "cpu_noise": cpu_noise,
                    "normalize": normalize,
                },
            ),
            sampler.sampler_function,
        )
        return (
            samplers.KSAMPLER(
                sampler_function,
                extra_options=sampler.extra_options.copy(),
                inpaint_options=sampler.inpaint_options.copy(),
            ),
        )

    @staticmethod
    def sampler_function(
        model,
        x,
        sigmas,
        *args: list[Any],
        override_sampler_cfg: dict[str, Any] | None = None,
        noise_sampler: Callable | None = None,
        extra_args: dict[str, Any] | None = None,
        **kwargs: dict[str, Any],
    ) -> torch.Tensor:
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
    "SonarNoiseImage": SonarNoiseImageNode,
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
    "SonarShuffledNoise": SonarShuffledNoiseNode,
    "SonarPatternBreakNoise": SonarPatternBreakNoiseNode,
    "SonarChannelNoise": SonarChannelNoiseNode,
    "SonarBlendedNoise": SonarBlendedNoiseNode,
    "SonarResizedNoise": SonarResizedNoiseNode,
    "SonarWaveletNoise": SonarWaveletNoiseNode,
    "SonarWaveletFilteredNoise": SonarWaveletFilteredNoiseNode,
    "SonarRippleFilteredNoise": SonarRippleFilteredNoiseNode,
    "SonarQuantileFilteredNoise": SonarQuantileFilteredNoiseNode,
    "SONAR_CUSTOM_NOISE to NOISE": SonarToComfyNOISENode,
    "SonarApplyLatentOperationCFG": SonarApplyLatentOperationCFG,
    "SonarLatentOperationQuantileFilter": SonarLatentOperationQuantileFilter,
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
