# ruff: noqa: TID252

from __future__ import annotations

from .. import noise, utils
from ..latent_ops import SonarLatentOperation
from ..sonar import SonarGuidanceMixin
from .base import (
    NOISE_INPUT_TYPES_HINT,
    WILDCARD_NOISE,
    SonarCustomNoiseNodeBase,
    SonarNormalizeNoiseNodeMixin,
)


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
        result["optional"] = {
            "sonar_custom_noise": (
                WILDCARD_NOISE,
                {
                    "tooltip": f"Optional custom noise input to combine with the guidance. If you don't attach something here your reference will be combined with zeros.\n{NOISE_INPUT_TYPES_HINT}",
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
        normalize_noise,
        normalize_result,
        normalize_ref=True,
        method="euler",
        guidance_factor=0.5,
        sonar_custom_noise=None,
    ):
        return super().go(
            factor,
            ref_latent=utils.scale_noise(
                SonarGuidanceMixin.prepare_ref_latent(latent["samples"].clone()),
                normalized=normalize_ref,
            ),
            guidance_factor=guidance_factor,
            noise=sonar_custom_noise.clone()
            if sonar_custom_noise is not None
            else None,
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
                    "tooltip": "Controls whether the generated noise is normalized to 1.0 strength.",
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


class SonarWaveletFilteredNoiseNode(
    SonarCustomNoiseNodeBase,
    SonarNormalizeNoiseNodeMixin,
):
    DESCRIPTION = "Custom noise type that allows filtering another custom noise source with wavelets."
    _yaml_placeholder = """# YAML or JSON here. YAML example:
# Wavelet type to use.
wave: haar

# Wavelet level
level: 3

# Wavelet padding mode.
mode: periodization

# Enables DTCWT mode.
use_dtcwt: false

# Configuration for DTCWT, only relevant when enabled.
biort: near_sym_a
qshift: qshift_a

# Does the inverse operation with the high/low parts separately
# (the other is zeroed) and then adds the results. Might work
# better for noise.
two_step_inverse: false

# It's also possible to set these options with an "inv_"
# prefix: mode, biort, qshift, wave, mode

# Scale for the low frequency side.
yl_scale: 1.0

# Scales for the high frequency side. Can be a scalar
# or a list or list of lists.
yh_scales: 1.0
"""

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
                    "default": "default",
                    "tooltip": "Controls whether the generated noise is normalized to 1.0 strength.",
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
            "custom_noise_high": (
                WILDCARD_NOISE,
                {
                    "tooltip": f"Optional: Custom noise input. If unconnected will use the same noise generator as custom_noise. However, if you do connect it this noise will be used for the high-frequency side of the wavelet.\n{NOISE_INPUT_TYPES_HINT}",
                },
            ),
            "yaml_parameters": (
                "STRING",
                {
                    "tooltip": "Allows specifying custom parameters via YAML. Note: When specifying paramaters this way, there is no error checking.",
                    "placeholder": cls._yaml_placeholder,
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
        custom_noise_high=None,
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
            noise_high=custom_noise_high
            if custom_noise_high is not None
            else custom_noise,
            yaml_parameters=yaml_parameters,
        )


class SonarScatternetFilteredNoiseNode(
    SonarCustomNoiseNodeBase,
    SonarNormalizeNoiseNodeMixin,
):
    DESCRIPTION = "Custom noise type that allows filtering noise using a scatternet (basically wavelets). Requires the pytorch_wavelets package to be installed in your Python environment. Can be used to do stuff like take the higher frequency components of a very low-frequency noise type such as Pyramid. Currently only works with 4D latents."

    @classmethod
    def INPUT_TYPES(cls):
        result = super().INPUT_TYPES()
        result["required"] |= {
            "padding_mode": (
                "STRING",
                {
                    "default": "symmetric",
                    "tooltip": "This is just passed to the pytorch_wavelets scatternet constructor. Valid padding modes that I know of (second order only supports symmetric and zero): symmetric, reflect, zero, periodization, constant, replicate, periodic",
                },
            ),
            "use_symmetric_filter": (
                "BOOLEAN",
                {
                    "default": False,
                    "tooltip": "Slower, but possibly higher quality.",
                },
            ),
            "magbias": (
                "FLOAT",
                {
                    "default": 1e-02,
                    "min": -1000.0,
                    "max": 1000.0,
                    "tooltip": "Magnitude bias. Changing it doesn't seem to affect anything, but you can try.",
                },
            ),
            "output_offset": (
                "FLOAT",
                {
                    "default": 0.0,
                    "min": -100000.0,
                    "max": 100000.0,
                    "tooltip": "Controls where the output starts. The beginning is the low frequency bands, the end is high frequencies. If less than 1 (positive or negative) it will be treated as a percentage into the dimension. Negative values count from the end.",
                },
            ),
            "output_mode": (
                ("channels_adjusted", "flat_adjusted", "channels", "flat"),
                {
                    "default": "channels_adjusted",
                    "tooltip": "The normal scatternet reduces the spatial dimensions 2x, the second order one 4x. The adjusted modes will generate larger noise (in the spatial dimensions) to compensate, this is slower but gives you a lot more room to work with. Modes that start with channels will index along the channel dimension, otherwise the indexing will be flat (after the batch dimension). Note: I recommend channels_adjusted mode, it's very possible the offset indexing math is wrong for other modes.",
                },
            ),
            "scatternet_order": (
                "INT",
                {
                    "default": 1,
                    "min": -3,
                    "max": 3,
                    "tooltip": "Each order increases the number of channels exponentially. You can use a primitive node to bypass the limit of 3 here if you're a crazy person, the code will handle any value but you're very likely to die of old age or run out of VRAM or both if you go above 3 (and even that is stretching it). You can set this to 0 to disable scatternet filtering quickly. Negative values are the same as positive ones here with one exception: there's a specialized 2nd order scatternet which will be used by default for order 2, however it may not support the normal parameters (like padding modes). Use -2 here if you just want to stack two normal scatternet layers instead.",
                },
            ),
            "per_channel_scatternet": (
                "BOOLEAN",
                {
                    "default": False,
                    "tooltip": "Runs the scatternet on each channel separately. May be very slow. Models like SDXL use 4 channels, models like Flux have 16. Enabling this may help with non-adjusted output modes.",
                },
            ),
            "normalize_noise": (
                "BOOLEAN",
                {
                    "default": False,
                    "tooltip": "Controls whether the noise source is normalized before scatternet filtering occurs.",
                },
            ),
            "normalize": (
                ("default", "forced", "disabled"),
                {
                    "default": "default",
                    "tooltip": "Controls whether the generated noise is normalized to 1.0 strength.",
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
        }
        return result

    @classmethod
    def get_item_class(cls):
        return noise.ScatternetFilteredNoise

    def go(
        self,
        *,
        factor: float,
        rescale: float,
        padding_mode: str,
        use_symmetric_filter: bool,
        magbias: float,
        output_offset: float,
        output_mode: str,
        scatternet_order: int,
        per_channel_scatternet: bool,
        normalize: str,
        normalize_noise: bool,
        custom_noise: object | None = None,
    ):
        return super().go(
            factor,
            rescale=rescale,
            noise=custom_noise,
            padding_mode=padding_mode,
            use_symmetric_filter=use_symmetric_filter,
            magbias=magbias,
            output_offset=output_offset,
            output_mode=output_mode,
            scatternet_order=scatternet_order,
            per_channel_scatternet=per_channel_scatternet,
            normalize_noise=normalize_noise,
            normalize=normalize,
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
                    "tooltip": "Controls whether the generated noise is normalized to 1.0 strength.",
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


class SonarNormalizeNoiseToScaleNode(
    SonarCustomNoiseNodeBase,
    SonarNormalizeNoiseNodeMixin,
):
    DESCRIPTION = "Custom noise type that allows precisely controling noise normalization. The default range of -4.5 to 4.5 is roughly what you'd get from 10,000 items of Gaussian noise."

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
            "min_negative_value": (
                "FLOAT",
                {
                    "default": -4.5,
                    "min": -10000.0,
                    "max": 10000.0,
                    "tooltip": "In simple mode, this is just the lowest value in the range (and can be positive, despite the name). In advanced mode, this controls the minimum negative value. If you set it to 0 or higher then normalization will leave negative values alone.",
                },
            ),
            "max_negative_value": (
                "FLOAT",
                {
                    "default": 0.0,
                    "min": -10000.0,
                    "max": 10000.0,
                    "tooltip": "Not used in simple mode. In advanced mode, this controls the maximum negative value. If you set it to 0 or higher, a maximum negative value will be automatically determined from negative value closest (but not equal to) zero.",
                },
            ),
            "min_positive_value": (
                "FLOAT",
                {
                    "default": 0.0,
                    "min": -10000.0,
                    "max": 10000.0,
                    "tooltip": "Not used in simple mode. In advanced mode, this controls the minmum positive value. If you set it to 0 or lower, a minimum positive value will be automatically determined from positive value closest (but not equal to) zero.",
                },
            ),
            "max_positive_value": (
                "FLOAT",
                {
                    "default": 4.5,
                    "min": -10000.0,
                    "max": 10000.0,
                    "tooltip": "In simple mode, this is just the highest value in the range (and can be negative, despite the name). In advanced mode, this controls the maximum positive value. If you set it to 0 or lower then normalization will leave positive values alone.",
                },
            ),
            "mode": (
                ("simple", "advanced"),
                {
                    "default": "simple",
                    "tooltip": "There are several modes:\nsimple: The noise will be rebalanced to be in between min_negative_value and max_positive_value. Though it sounds weird, you don't need to respect the positive/negative in the names. It is just treated as a simple range.\nadvanced: Positive and negative values in the noise are separately rebalanced to be between the specified ranges. If you set max_negative_value to something positive or min_positive_value to something negative this will automatically determine whatever the closest value to zero is for each sign. Additionally, if you set max_positive_value to something negative or min_negative_value to something positive then values for that sign will be left alone.",
                },
            ),
            "dims": (
                "STRING",
                {
                    "default": "-3, -2, -1",
                    "tooltip": "A comma separated list of dimensions which can be negative to count from the end. This behaves differently in advanced mode: If left blank, normalization will be global. If set to anything, normalization will be over each batch item separately. The actual values of the dimensions are ignored in advanced mode currently.",
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
                    "default": "disabled",
                    "tooltip": "Controls whether the generated noise is normalized to 1.0 strength.",
                },
            ),
        }
        return result

    @classmethod
    def get_item_class(cls):
        return noise.NormalizeToScaleNoise

    def go(
        self,
        *,
        factor,
        rescale,
        normalize,
        min_negative_value: float,
        max_negative_value: float,
        min_positive_value: float,
        max_positive_value: float,
        mode: str,
        dims: str,
        normalize_noise: bool,
        custom_noise=None,
        sonar_custom_noise_opt=None,
    ):
        return super().go(
            factor,
            rescale=rescale,
            sonar_custom_noise_opt=sonar_custom_noise_opt,
            min_negative_value=min_negative_value,
            max_negative_value=max_negative_value,
            min_positive_value=min_positive_value,
            max_positive_value=max_positive_value,
            mode=mode,
            dims=() if not dims.strip() else tuple(int(i) for i in dims.split(",")),
            normalize=self.get_normalize(normalize),
            normalize_noise=normalize_noise,
            noise=custom_noise,
        )


class SonarPerDimNoiseNode(
    SonarCustomNoiseNodeBase,
    SonarNormalizeNoiseNodeMixin,
):
    DESCRIPTION = "Custom noise type that allows calling the noise sampler multiple times along a dimension. Can be useful for stuff like moving slices of 3D Perlin noise into the batch dimension."

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
            "dim": (
                "INT",
                {
                    "default": 0,
                    "min": -100,
                    "max": 100,
                    "tooltip": "Dimension to use. The default usually corresponds to the batch. Be careful using dimensions above 1 as those tend to be spatial and you might end up calling a slow noise sampler hundreds of times.",
                },
            ),
            "shrink_dim": (
                "BOOLEAN",
                {
                    "default": False,
                    "tooltip": "When enabled, the reference latent will be chunk_size in the specified dimension. When disabled, noise will be generated according to the initial latent size and then sliced along the specified dimension. Enabling it should be considerably faster/more memory efficient but may not work well for some noise types.",
                },
            ),
            "chunk_size": (
                "INT",
                {
                    "default": 1,
                    "min": 1,
                    "max": 10000,
                    "tooltip": "Can be used to control how many times the noise sampler is called. For example, if you have dim=0, chunk_size=2 and are dealing with a batch of 4, this will call the noise sampler twice, taking the first two items from the first call and the last two items from the second call.",
                },
            ),
            # "offset": (
            #     "INT",
            #     {
            #         "default": 0,
            #         "min": -10000,
            #         "max": 10000,
            #     },
            # ),
            "normalize_noise": (
                "BOOLEAN",
                {
                    "default": False,
                    "tooltip": "Controls whether the noise source is normalized initially.",
                },
            ),
            "normalize": (
                ("default", "forced", "disabled"),
                {
                    "default": "disabled",
                    "tooltip": "Controls whether the generated noise is normalized to 1.0 strength.",
                },
            ),
        }
        return result

    @classmethod
    def get_item_class(cls):
        return noise.PerDimNoise

    def go(
        self,
        *,
        factor,
        rescale,
        normalize,
        dim: int,
        shrink_dim: bool,
        chunk_size: int,
        # offset: int,
        normalize_noise: bool,
        custom_noise=None,
        sonar_custom_noise_opt=None,
    ):
        return super().go(
            factor,
            rescale=rescale,
            sonar_custom_noise_opt=sonar_custom_noise_opt,
            dim=dim,
            shrink_dim=shrink_dim,
            # offset=offset,
            chunk_size=chunk_size,
            normalize=self.get_normalize(normalize),
            normalize_noise=normalize_noise,
            noise=custom_noise,
        )


class SonarLatentOperationFilteredNoiseNode(
    SonarCustomNoiseNodeBase,
    SonarNormalizeNoiseNodeMixin,
):
    DESCRIPTION = "Custom noise type that allows filtering noise with a LATENT_OPERATION. If you connect more than one, the operations will be run in sequence."

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
            "normalize_noise": (
                "BOOLEAN",
                {
                    "default": False,
                    "tooltip": "Controls whether the noise source is normalized initially.",
                },
            ),
            "normalize": (
                ("default", "forced", "disabled"),
                {
                    "default": "disabled",
                    "tooltip": "Controls whether the generated noise is normalized to 1.0 strength.",
                },
            ),
        }
        result["optional"] |= {
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
        }
        return result

    @classmethod
    def get_item_class(cls):
        return noise.LatentOperationFilteredNoise

    def go(
        self,
        *,
        factor,
        rescale,
        normalize,
        normalize_noise: bool,
        custom_noise=None,
        sonar_custom_noise_opt=None,
        operation_1=None,
        operation_2=None,
        operation_3=None,
        operation_4=None,
        operation_5=None,
    ):
        operations = tuple(
            o if isinstance(o, SonarLatentOperation) else SonarLatentOperation(op=o)
            for o in (operation_1, operation_2, operation_3, operation_4, operation_5)
            if o is not None
        )
        return super().go(
            factor,
            rescale=rescale,
            sonar_custom_noise_opt=sonar_custom_noise_opt,
            operations=operations,
            normalize=self.get_normalize(normalize),
            normalize_noise=normalize_noise,
            noise=custom_noise,
        )


NODE_CLASS_MAPPINGS = {
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
    "SonarWaveletFilteredNoise": SonarWaveletFilteredNoiseNode,
    "SonarRippleFilteredNoise": SonarRippleFilteredNoiseNode,
    "SonarQuantileFilteredNoise": SonarQuantileFilteredNoiseNode,
    "SonarNormalizeNoiseToScale": SonarNormalizeNoiseToScaleNode,
    "SonarPerDimNoise": SonarPerDimNoiseNode,
    "SonarScatternetFilteredNoise": SonarScatternetFilteredNoiseNode,
    "SonarLatentOperationFilteredNoise": SonarLatentOperationFilteredNoiseNode,
}
