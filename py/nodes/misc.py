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
from tqdm import tqdm

from .. import noise, utils
from ..external import IntegratedNode
from ..noise import NoiseType
from ..wavelet_cfg import WaveletCFG, WCFGRules
from .base import (
    NoiseChainInputTypes,
    SonarCustomNoiseNodeBase,
    SonarInputTypes,
    SonarLazyInputTypes,
    SonarNormalizeNoiseNodeMixin,
)


class NoisyLatentLikeNode(metaclass=IntegratedNode):
    DESCRIPTION = "Allows generating noise (and optionally adding it) based on a reference latent. Note: For img2img workflows, you will generally want to enable add_to_latent as well as connecting the model and sigmas inputs."
    RETURN_TYPES = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The noisy latent image.",)
    CATEGORY = "latent/noise"

    FUNCTION = "go"

    INPUT_TYPES = SonarLazyInputTypes(
        lambda: SonarInputTypes()
        .req_selectnoise_noise_type(
            tooltip="Sets the type of noise to generate. Has no effect when the custom_noise_opt input is connected.",
        )
        .req_seed()
        .req_latent(tooltip="Latent used as a reference for generating noise.")
        .req_float_multiplier(
            default=1.0,
            tooltip="Multiplier for the strength of the generated noise. Performed after mul_by_sigmas_opt.",
        )
        .req_bool_add_to_latent(
            tooltip="Add the generated noise to the reference latent rather than adding it to an empty latent. Generally should be enabled for img2img workflows.",
        )
        .req_int_repeat_batch(
            default=1,
            min=1,
            tooltip="Repeats the noise generation the specified number of times. For example, if set to two and your reference latent is also batch two you will get a batch of four as output.",
        )
        .req_bool_cpu_noise(
            default=True,
            tooltip="Controls whether noise will be generated on GPU or CPU. Only affects noise types that support GPU generation (maybe only Brownian).",
        )
        .req_bool_normalize(
            default=True,
            tooltip="Controls whether the generated noise is normalized to 1.0 strength before scaling. Generally should be left enabled.",
        )
        .opt_customnoise_custom_noise_opt()
        .opt_sigmas_mul_by_sigmas_opt(
            tooltip="When connected, will scale the generated noise by the first sigma. Must also connect model_opt to enable.",
        )
        .opt_model_model_opt(
            tooltip="Used when mul_by_sigmas_opt is connected, no effect otherwise.",
        ),
    )

    @classmethod
    def go(
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

    INPUT_TYPES = SonarLazyInputTypes(
        lambda: SonarInputTypes()
        .req_selectnoise_noise_type(
            tooltip="Sets the type of noise to generate. Has no effect when the custom_noise_opt input is connected.",
        )
        .req_seed()
        .req_image(tooltip="Image noise will be added to.")
        .req_float_noise_min(
            default=0.0,
            tooltip="Generated noise will be normalized to have values between noise_min and noise_max. If you set them both to the same value then this disables normalization.",
        )
        .req_float_noise_max(
            default=1.0,
            tooltip="Generated noise will be normalized to have values between noise_min and noise_max. If you set them both to the same value then this disables normalization.",
        )
        .req_float_noise_multiplier(
            default=0.5,
            tooltip="Multiplier for the strength of the generated noise. This is performed after noise_min/max scaling.",
        )
        .req_field_channel_mode(
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
            default="RGB",
            tooltip="RGBA will also add noise to the alpha channel as well if it exists. Only used for 3 or 4 channel images, for other numbers of channels (i.e. one channel) then all channels will be targeted.",
        )
        .req_selectblend(
            insert_modes=("simple_add",),
            default="simple_add",
            tooltip="Controls how the generated noise is combined with the image. simple_add just adds it and blend_strength is ignored in that case.",
        )
        .req_float_blend_strength(
            default=0.5,
            tooltip="Multiplier for the strength of the generated noise.",
        )
        .req_field_overflow_mode(
            ("clamp", "rescale"),
            default="clamp",
            tooltip="When set to clamp, values above/below 0, 1 will be set to those values. When set to rescale, the image values will be rescaled such that the minimum value is 0 and the maximum is 1.",
        )
        .req_bool_greyscale_mode(
            tooltip="When set to clamp, values above/below 0, 1 will be set to those values. When set to rescale, the image values will be rescaled such that the minimum value is 0 and the maximum is 1.",
        )
        .req_bool_pure_noise_mode(
            tooltip="When enabled, the original image is only used for its shape and you will be adding noise to an image full of zeros (black), suitable for creating pure noise images.",
        )
        .req_field_dtype(
            ("default", "float32", "float64", "float16", "bfloat16"),
            default="default",
            tooltip="When set to default it will use the same type as the input tensor (probably float32). You can manually set the dtype if you want, though it likely isn't going to matter. Using dtypes with limited range (float16, bfloat16) isn't recommended.",
        )
        .req_bool_cpu_noise(
            default=True,
            tooltip="Controls whether noise will be generated on GPU or CPU.",
        )
        .req_bool_normalize(
            default=True,
            tooltip="Controls whether the generated noise is normalized to 1.0 strength before scaling. Generally should be left enabled.",
        )
        .opt_customnoise_custom_noise_opt(
            tooltip="Allows connecting a custom noise chain. When connected, noise_type has no effect.",
        ),
    )

    @classmethod
    def go(
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

    INPUT_TYPES = SonarLazyInputTypes(
        lambda: SonarInputTypes()
        .req_customnoise_custom_noise(
            tooltip="Custom noise type to convert.",
        )
        .req_seed(tooltip="Seed to use for generated noise.")
        .req_bool_cpu_noise(
            default=True,
            tooltip="Controls whether noise is generated on CPU or GPU.",
        )
        .req_bool_normalize(
            default=True,
            tooltip="Controls whether generated noise is normalized to 1.0 strength.",
        )
        .req_float_multiplier(
            default=1.0,
            tooltip="Simple multiplier applied to noise after all other scaling and normalization effects. If set to 0, no noise will be generated (same as disabling noise).",
        ),
    )

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

    INPUT_TYPES = SonarLazyInputTypes(
        lambda: SonarInputTypes()
        .req_sampler()
        .req_float_eta(
            default=1.0,
            tooltip="Basically controls the ancestralness of the sampler. When set to 0, you will get a non-ancestral (or SDE) sampler.",
        )
        .req_float_s_noise(
            default=1.0,
            tooltip="Multiplier for noise added during ancestral or SDE sampling.",
        )
        .req_float_s_churn(
            default=0.0,
            tooltip="Churn was the predececessor of ETA. Only used by a few types of samplers (notably Euler non-ancestral). Not used by any ancestral or SDE samplers.",
        )
        .req_float_r(
            default=0.5,
            tooltip="Used by dpmpp_sde (and perhaps a few other SDE samplers).",
        )
        .req_field_sde_solver(
            ("midpoint", "heun"),
            tooltip="Solver used by dpmpp_2m_sde.",
        )
        .req_bool_cpu_noise(
            default=True,
            tooltip="Controls whether noise is generated on CPU or GPU.",
        )
        .req_bool_normalize(
            default=True,
            tooltip="Controls whether generated noise is normalized to 1.0 strength.",
        )
        .opt_selectnoise_noise_type(
            insert_types=("DEFAULT",),
            default="DEFAULT",
            tooltip="Noise type used during ancestral or SDE sampling. DEFAULT will use the default for the attached sampler. Only used when the custom noise input is not connected.",
        )
        .opt_customnoise_custom_noise_opt(
            tooltip="Optional input for custom noise used during ancestral or SDE sampling. When connected, the built-in noise_type selector is ignored.",
        )
        .opt_yaml(),
    )

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

    INPUT_TYPES = SonarLazyInputTypes(
        lambda: NoiseChainInputTypes()
        .req_normalizetristate_normalize(
            tooltip="Controls whether the generated noise is normalized to 1.0 strength.",
        )
        .opt_customnoise_custom_noise(),
    )

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


class SonarWaveletCFGNode(metaclass=IntegratedNode):
    DESCRIPTION = "Wavelet CFG function that allows you to apply different CFG strength to different frequencies."
    CATEGORY = "model_patches"
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "go"

    _yaml_placeholder = """# YAML or JSON here.
# I recommend reading the documentation at https://github.com/blepping/ComfyUI-sonar/docs/waveletcfg.md
# For wavelet information, see: https://pytorch-wavelets.readthedocs.io/en/latest/index.html

# You may override the fields from the node like start_sigma here.

# This section is basically the CFG scale. (All scales sections use the same format.)
difference:
    # Scale for the low frequency components.
    yl_scale: 5.0

    # Scale (or scales) for high frequency components.
    # This can be scalar or a list or list of lists.
    # List example:
    #  yh_scales:
    #      - [1, 2, 3]
    #      - fill
    #      - 5
    # You can separately apply a scale to items equal to the wavelet level. Levels go from fine to coarse.
    # If the item is a list, the three items correspond to horizontal, vertical, diagonal for DWT. (DTCWT has 6.)
    # You can have one "fill" item, this will replicate the item before it however many times is necessary to
    # match the wavelet level.
    yh_scales: 3.0

    # You can optionally include a scales_end block with yl_scale/yh_scales.
    # to interpolate from the toplevel scales (can also be in a scales_start blockx if you prefer).

    # scales_end:
    #     yl_scale: 1.0
    #     yh_scales: 1.0

    # The following scheduling parameters only apply if scales_end exists.

    # One of linear, logarithmic, exponential, half_cosine, sine
    # Sine mode will hit the peak scales_after values in the middle of the range.
    schedule: linear

    # One of: sampling, enabled_sampling, sigmas, enabled_sigmas, step, enabled_steps
    schedule_mode: sampling

    # When enabled, flips the schedule percentage. This happens before the schedule is applied
    # or any offset/multiplier stuff. If you want to flip the final result you can do something like
    # schedule_offset_after: -1.0 and schedule_multiplier_after: -1.0
    reverse_schedule: false

    # Added to the percentage before the schedule function is applied.
    schedule_offset: 0.0

    # Applied to the percentage before the schedule function (but after the offset).
    schedule_multiplier: 1.0

    # Added to the percentage after the schedule function is applied.
    schedule_offset_after: 0.0

    # Applied to the percentage after the schedule function (but after the offset).
    schedule_multiplier_after: 1.0

    # Min/max for the final calculated percent. Must be between 0 and 1.
    schedule_min: 0.0
    schedule_max: 1.0

    # If you're a crazy person, you can use non-standard blend modes for interpolating
    # the scales. Not recommended.
    blend_mode: lerp


# Wavelet type
wave: db4

# Wavelet level
level: 5

### Start of advanced options

# Mode used for padding
padding_mode: symmetric

# Mutually exclusive with DTCWT mode.
use_1d_dwt: false

# Enables DTCWT mode.
use_dtcwt: false

# Configuration for DTCWT, only relevant when enabled.
biort: near_sym_a
qshift: qshift_a

# It's also possible to set these wavelet options with an "inv_"
# prefix: mode, biort, qshift, wave, padding_mode

# One of: noise_norm, noise, denoised
# Normal CFG uses denoised mode. noise_norm divides by the current sigma, noise just uses the raw noise prediction.
target_mode: denoised

# Can be used to scale cond before the difference is calculated.
cond:
    yl_scale: 1.0
    yh_scales: 1.0

# Can be used to scale uncond before the difference is calculated.
uncond:
    yl_scale: 1.0
    yh_scales: 1.0

# Can be used to scale the final result after blending.
final:
    yl_scale: 1.0
    yh_scales: 1.0

# Uses float64 for the wavelets/scaling/blending operations.
# It doesn't seem to hurt performance much, but you can disable it if you want.
high_precision_mode: true

# Inject is just addition which is usually what you want. The normal CFG function is:
# uncond + (cond - uncond) * cfg_scale
difference_blend_mode: inject
difference_blend_strength: 1.0

# Per-rule value, can be enabled to spam your console with information when
# rules activate, dump exactly what high/low scales are used, etc.
verbose: false

# You may include a rules block which is a list of these configuration definitions.
# Include start_sigma/end_sigma parameters. The first matching definition will be used.
# rules:
#     - start_sigma: -1.0
"""

    INPUT_TYPES = SonarLazyInputTypes(
        lambda _yaml_placeholder=_yaml_placeholder: SonarInputTypes()
        .req_model()
        .req_float_start_sigma(
            default=-1.0,
            min=-1.0,
            tooltip="First sigma wavelet CFG will be used.",
        )
        .req_float_end_sigma(
            default=0.0,
            min=0.0,
            tooltip="Last sigma wavelet CFG will be used.",
        )
        .req_field_fallback_mode(
            ("existing", "own"),
            default="existing",
            tooltip="Existing mode uses whatever CFG function existed set when this model patch was applied. Own mode does the CFG calculation on its own. The scale will be whatever you set in your guider or sampler.",
        )
        .req_selectblend_blend_mode(
            tooltip="Controls how the result from wavelet CFG is blended with normal CFG. The default of LERP with strength 1.0 uses 100% wavelet CFG.",
        )
        .req_float_blend_strength(
            default=1.0,
            tooltip="Controls how the result from wavelet CFG is blended with normal CFG. The default of LERP with strength 1.0 uses 100% wavelet CFG.",
        )
        .req_yaml(default=_yaml_placeholder)
        .opt_field_operation_cond(
            "LATENT_OPERATION",
            tooltip="Optional latent operation that will be applied to cond. Note: Latent operations only apply if a rule matches.",
        )
        .opt_field_operation_uncond(
            "LATENT_OPERATION",
            tooltip="Optional latent operation that will be applied to uncond. Note: Latent operations only apply if a rule matches.",
        )
        .opt_field_operation_fallback_cfg(
            "LATENT_OPERATION",
            tooltip="Optional latent operation that will be applied to the fallback (non-wavelet) CFG result. Note: Latent operations only apply if a rule matches.",
        )
        .opt_field_operation_wavelet_cfg(
            "LATENT_OPERATION",
            tooltip="Optional latent operation that will be applied to wavelet CFG result. Note: Latent operations only apply if a rule matches.",
        )
        .opt_field_operation_result(
            "LATENT_OPERATION",
            tooltip="Optional latent operation that will be applied to the final result, after wavelet and normal CFG are potentially blended. Note: Latent operations only apply if a rule matches.",
        ),
    )

    @classmethod
    def go(
        cls,
        *,
        model: object,
        start_sigma: float,
        end_sigma: float,
        fallback_mode: str,
        blend_mode: str,
        blend_strength: float,
        yaml_parameters: str,
        operation_cond: Callable | None = None,
        operation_uncond: Callable | None = None,
        operation_fallback_cfg: Callable | None = None,
        operation_wavelet_cfg: Callable | None = None,
        operation_result: Callable | None = None,
        _override_rules_dict: dict | None = None,
    ) -> tuple[object]:
        if start_sigma < 0:
            start_sigma = math.inf
        if _override_rules_dict is not None:
            wavelet_params = _override_rules_dict.copy()
        else:
            wavelet_params = yaml.safe_load(yaml_parameters)
        rules = WCFGRules.build(
            **(
                {
                    "start_sigma": start_sigma,
                    "end_sigma": end_sigma,
                    "fallback_existing": fallback_mode == "existing",
                    "blend_mode": blend_mode,
                    "blend_strength": blend_strength,
                }
                | wavelet_params
            ),
        )
        if len(rules) and rules[0].verbose:
            tqdm.write(f"\nWCFG: Using rules: {rules}\n")
        model = model.clone()
        model.set_model_sampler_cfg_function(
            WaveletCFG(
                existing_cfg=model.model_options.get("sampler_cfg_function"),
                rules=rules,
                operation_cond=operation_cond,
                operation_uncond=operation_uncond,
                operation_fallback_cfg=operation_fallback_cfg,
                operation_wavelet_cfg=operation_wavelet_cfg,
                operation_result=operation_result,
            ),
        )
        return (model,)


NODE_CLASS_MAPPINGS = {
    "NoisyLatentLike": NoisyLatentLikeNode,
    "SamplerConfigOverride": SamplerNodeConfigOverride,
    "SONAR_CUSTOM_NOISE to NOISE": SonarToComfyNOISENode,
    "SonarNoiseImage": SonarNoiseImageNode,
    "SonarSplitNoiseChain": SonarSplitNoiseChainNode,
    "SonarWaveletCFG": SonarWaveletCFGNode,
}
