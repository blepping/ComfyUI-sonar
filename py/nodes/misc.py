# ruff: noqa: TID252

from __future__ import annotations

import functools
import inspect
import math
import random
from typing import Any, Callable

import numpy as np
import torch
import yaml
from comfy import model_management, samplers

from .. import noise, utils
from ..external import IntegratedNode
from ..noise import NoiseType
from .base import (
    NOISE_INPUT_TYPES_HINT,
    WILDCARD_NOISE,
    SonarCustomNoiseNodeBase,
    SonarNormalizeNoiseNodeMixin,
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
                    tuple(noise.NoiseType.get_names()),
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
            errstr = (
                f"Expected image tensor with 3 or 4 dimensions, got {image.ndim}",
            )
            raise ValueError(errstr)
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


class SonarSplitNoiseChainNode(SonarCustomNoiseNodeBase, SonarNormalizeNoiseNodeMixin):
    DESCRIPTION = "Custom noise type that allows splitting off a new chain. This can be useful if you want a link in the chain to be a blended type."

    @classmethod
    def INPUT_TYPES(cls):
        result = super().INPUT_TYPES()
        result["required"] |= {
            "normalize": (
                ("default", "forced", "disabled"),
                {
                    "tooltip": "Controls whether the generated noise is normalized to 1.0 strength.",
                },
            ),
        }
        result["optional"] |= {
            "custom_noise": (
                WILDCARD_NOISE,
                {"tooltip": f"Custom noise. \n{NOISE_INPUT_TYPES_HINT}"},
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
        custom_noise=None,
    ):
        return super().go(
            factor,
            rescale=rescale,
            sonar_custom_noise_opt=sonar_custom_noise_opt,
            blend_function=lambda a, _b, _t: a,
            normalize=self.get_normalize(normalize),
            custom_noise_1=custom_noise,
            custom_noise_2=None,
            noise_2_percent=0.0,
        )


NODE_CLASS_MAPPINGS = {
    "NoisyLatentLike": NoisyLatentLikeNode,
    "SamplerConfigOverride": SamplerNodeConfigOverride,
    "SONAR_CUSTOM_NOISE to NOISE": SonarToComfyNOISENode,
    "SonarNoiseImage": SonarNoiseImageNode,
    "SonarSplitNoiseChain": SonarSplitNoiseChainNode,
}
