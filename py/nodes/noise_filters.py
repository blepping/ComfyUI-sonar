from __future__ import annotations

import torch
from comfy import model_management

from .. import noise, utils
from ..latent_ops import SonarLatentOperation
from ..sonar import SonarGuidanceMixin
from .base import (
    NoiseChainInputTypes,
    NoiseNoChainInputTypes,
    SonarCustomNoiseNodeBase,
    SonarLazyInputTypes,
    SonarNormalizeNoiseNodeMixin,
)


class SonarModulatedNoiseNode(SonarCustomNoiseNodeBase, SonarNormalizeNoiseNodeMixin):
    DESCRIPTION = "Custom noise type that allows modulating the output of another custom noise generator."

    INPUT_TYPES = SonarLazyInputTypes(
        lambda: NoiseNoChainInputTypes()
        .req_customnoise_sonar_custom_noise(tooltip="Custom noise type to modulate.")
        .req_field_modulation_type(
            (
                "intensity",
                "frequency",
                "spectral_signum",
                "none",
            ),
            tooltip="Type of modulation to use.",
        )
        .req_int_dims(
            default=3,
            min=1,
            max=3,
            tooltip="Dimensions to modulate over. 1 - channels only, 2 - height and width, 3 - both",
        )
        .req_float_strength(
            default=2.0,
            min=-100.0,
            max=100.0,
            tooltip="Controls the strength of the modulation effect.",
        )
        .req_normalizetristate_normalize_result(
            tooltip="Controls whether the final result is normalized to 1.0 strength.",
        )
        .req_normalizetristate_normalize_noise(
            tooltip="Controls whether the generated noise is normalized to 1.0 strength.",
        )
        .req_bool_normalize_ref(
            default=True,
            tooltip="Controls whether the reference latent (when present) is normalized to 1.0 strength.",
        )
        .opt_latent_ref_latent_opt(),
    )

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

    INPUT_TYPES = SonarLazyInputTypes(
        lambda: NoiseNoChainInputTypes()
        .req_customnoise_sonar_custom_noise(tooltip="Custom noise type to modulate.")
        .req_int_repeat_length(
            default=8,
            min=1,
            max=100,
            tooltip="Number of items to cache.",
        )
        .req_int_max_recycle(
            default=1000,
            min=1,
            max=1000,
            tooltip="Number of times an individual item will be used before it is replaced with fresh noise.",
        )
        .req_normalizetristate_normalize(
            tooltip="Controls whether the generated noise is normalized to 1.0 strength.",
        )
        .req_field_permute(
            ("enabled", "disabled", "always"),
            default="enabled",
            tooltip="When enabled, recycled noise will be permuted by randomly flipping it, rolling the channels, etc. If set to always, the noise will be permuted the first time it's used as well.",
        ),
    )

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

    INPUT_TYPES = SonarLazyInputTypes(
        lambda: NoiseNoChainInputTypes()
        .req_model(
            tooltip="The model input is required to calculate sampling percentages.",
        )
        .req_customnoise_sonar_custom_noise(
            tooltip="Custom noise to use when start_percent and end_percent matches.",
        )
        .req_float_start_percent(
            default=0.0,
            min=0.0,
            max=1.0,
            tooltip="Time the custom noise becomes active. Note: Sampling percentage where 1.0 indicates 100%, not based on steps.",
        )
        .req_float_end_percent(
            default=1.0,
            min=0.0,
            max=1.0,
            tooltip="Time the custom noise effect ends - inclusive, so only sampling percentages greater than this will be excluded. Note: Sampling percentage where 1.0 indicates 100%, not based on steps.",
        )
        .req_normalizetristate_normalize(
            tooltip="Controls whether the generated noise is normalized to 1.0 strength.",
        )
        .opt_customnoise_fallback_sonar_custom_noise(
            tooltip="Optional input for noise to use when outside of the start_percent, end_percent range. NOTE: When not connected, defaults to NO NOISE which is probably not what you want.",
        ),
    )

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

    INPUT_TYPES = SonarLazyInputTypes(
        lambda: NoiseNoChainInputTypes()
        .req_customnoise_sonar_custom_noise_dst(
            tooltip="Custom noise input for noise where the mask is not set.",
        )
        .req_customnoise_sonar_custom_noise_src(
            tooltip="Custom noise input for noise where the mask is set.",
        )
        .req_normalizetristate_normalize_dst(
            tooltip="Controls whether noise generated for dst is normalized to 1.0 strength.",
        )
        .req_normalizetristate_normalize_src(
            tooltip="Controls whether noise generated for src is normalized to 1.0 strength.",
        )
        .req_normalizetristate_normalize_result(
            tooltip="Controls whether the final result after composition is normalized to 1.0 strength.",
        )
        .req_field_mask(
            "MASK",
            tooltip="Mask to use when compositing noise. Where the mask is 1.0, you will get 100% src, where it is 0.75 you will get 75% src and 25% dst. The mask will be rescaled to match the latent size if necessary.",
        ),
    )

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

    INPUT_TYPES = SonarLazyInputTypes(
        lambda: NoiseNoChainInputTypes()
        .req_latent(
            tooltip="Latent to use for guidance.",
        )
        .req_field_method(
            ("euler", "linear"),
            default="euler",
            tooltip="Method to use when calculating guidance. When set to linear, will simply LERP the guidance at the specified strength. When set to Euler, will do a Euler step toward the guidance instead.",
        )
        .req_float_guidance_factor(
            default=0.0125,
            min=-100.0,
            max=100.0,
            tooltip="Strength of the guidance to apply. Generally should be a relatively slow value to avoid overpowering the generation.",
        )
        .req_normalizetristate_normalize_noise(
            tooltip="Controls whether the generated noise is normalized to 1.0 strength.",
        )
        .req_normalizetristate_normalize_result(
            tooltip="Controls whether the final result is normalized to 1.0 strength.",
        )
        .req_bool_normalize_ref(
            default=True,
            tooltip="Controls whether the reference latent (when present) is normalized to 1.0 strength.",
        )
        .opt_customnoise_sonar_custom_noise(
            tooltip="Optional custom noise input to combine with the guidance. If you don't attach something here your reference will be combined with zeros.",
        ),
    )

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

    INPUT_TYPES = SonarLazyInputTypes(
        lambda: NoiseNoChainInputTypes()
        .req_customnoise_sonar_custom_noise(
            tooltip="Custom noise input for noise items to randomize. Note: Unlike most other custom noise nodes, this is treated like a list.",
        )
        .req_int_mix_count(
            default=1,
            min=1,
            max=100,
            tooltip="Number of items to select each time noise is generated.",
        )
        .req_normalizetristate_normalize(
            tooltip="Controls whether the generated noise is normalized to 1.0 strength.",
        ),
    )

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

    INPUT_TYPES = SonarLazyInputTypes(
        lambda: NoiseNoChainInputTypes()
        .req_customnoise_sonar_custom_noise(
            tooltip="Custom noise input for noise items corresponding to each channel. SD1/2x and SDXL use 4 channels, Flux and SD3 use 16. Note: Unlike most other custom noise nodes, this is treated like a list where the noise item furthest from the node corresponds to channel 0.",
        )
        .req_field_insufficient_channels_mode(
            ("wrap", "repeat", "zero"),
            default="wrap",
            tooltip="Controls behavior for when there are less noise items connected than channels in the latent. wrap - wraps back to the first noise item, repeat - repeats the last item, zero - fills the channel with zeros (generally not recommended).",
        )
        .req_int_mix_count(
            default=1,
            min=1,
            max=100,
            tooltip="Number of items to select each time noise is generated.",
        )
        .req_normalizetristate_normalize(
            tooltip="Controls whether the generated noise is normalized to 1.0 strength.",
        ),
    )

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

    INPUT_TYPES = SonarLazyInputTypes(
        lambda: NoiseChainInputTypes()
        .req_float_noise_2_percent(
            default=0.5,
            tooltip="Blend strength for custom_noise_2. Note that if set to 0 then custom_noise_2 is optional (and will not be called to generate noise) and if set to 1 then custom_noise_1 will not be called to generate noise. This only applies when custom_noise_mask is not connected. This is worth mentioning since going from a strength of 0.000000001 to 0 could make a big difference. Important: When custom_noise_mask is connected, this value will be added to the mask and then the mask will be clamped to 0 through 1. In other words, you could use this to ensure the mask ranges between 0.5 and 1.0 by setting it to 0.5 or ensure it ranges between 0 and 0.5 by setting it to -0.5.",
        )
        .req_selectblend(
            tooltip="Mode used for blending the two noise types. More modes will be available if ComfyUI-bleh is installed.",
        )
        .req_normalizetristate_normalize(
            tooltip="Controls whether the generated noise is normalized to 1.0 strength. For weird blend modes, you may want to set this to forced.",
        )
        .opt_customnoise_custom_noise_1(
            tooltip="Custom noise. Optional if noise_2_percent is 1 and custom_noise_mask is not connected..",
        )
        .opt_customnoise_custom_noise_2(
            tooltip="Custom noise. Optional if noise_2_percent is 0 and custom_noise_mask is not connected..",
        )
        .opt_customnoise_custom_noise_mask(
            tooltip="Custom noise. If connected, this will be used instead of noise_2_percent to determine the blend ratio. Noise generated by this will be normalized to a 0 through 1 scale. When connected, both custom noise inputs are mandatory.",
        ),
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
        noise_2_percent,
        custom_noise_1=None,
        custom_noise_2=None,
        custom_noise_mask=None,
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
            custom_noise_mask=custom_noise_mask,
            noise_2_percent=noise_2_percent,
        )


class SonarResizedNoiseNode(SonarCustomNoiseNodeBase, SonarNormalizeNoiseNodeMixin):
    DESCRIPTION = "Custom noise type that allows resizing another noise item."

    INPUT_TYPES = SonarLazyInputTypes(
        lambda: NoiseNoChainInputTypes()
        .req_int_width(
            default=1152,
            min=16,
            max=1024 * 1024 * 1024,
            step=8,
            tooltip="Note: This should almost always be set to a higher value than the image you're actually sampling.",
        )
        .req_int_height(
            default=1152,
            min=16,
            max=1024 * 1024 * 1024,
            step=8,
            tooltip="Note: This should almost always be set to a higher value than the image you're actually sampling.",
        )
        .req_field_downscale_strategy(
            ("crop", "scale"),
            default="crop",
            tooltip="Scaling noise is something you'd pretty much only use to create weird effects. For normal workflows, leave this on crop.",
        )
        .req_field_initial_reference(
            ("prefer_crop", "prefer_scale"),
            default="prefer_crop",
            tooltip="The initial latent the noise sampler uses as a reference may not match the requested width/height. This setting controls whether to crop or scale. Note: Cropping can only occur when the initial reference is larger than width/height in both dimensions which is unlikely (and not recommended).",
        )
        .req_field_crop_mode(
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
            default="center",
            tooltip="Note: Crops will have a bias toward the lower number when the size isn't divisible by two. For example, a center crop of size 3 from (0, 1, 2, 3, 4, 5) will result in (1, 2, 3).",
        )
        .req_int_crop_offset_horizontal(
            default=0,
            step=8,
            min=-8000,
            max=8000,
            tooltip="This offsets the cropped view by the specified size. Positive values will move it toward the right, negative values will move it toward the left. The offsets will be adjusted to to fit in the available space. For example, if you have crop_mode set to top_right then setting a positive offset isn't going to do anything: it's already as far right as it can go.",
        )
        .req_int_crop_offset_vertical(
            default=0,
            step=8,
            min=-8000,
            max=8000,
            tooltip="This offsets the cropped view by the specified size. Positive values will move it toward the bottom, negative values will move it toward the top. The offsets will be adjusted to to fit in the available space. For example, if you have crop_mode set to bottom_right then setting a positive offset isn't going to do anything: it's already as far down as it can go.",
        )
        .req_selectscalemode_upscale_mode(
            tooltip="Allows setting the scaling mode when width/height is smaller than the requested size.",
            default="nearest-exact",
        )
        .req_selectscalemode_downscale_mode(
            tooltip="Allows setting the scaling mode when width/height is larger than the requested size and downscale_strategy is set to 'scale'.",
            default="nearest-exact",
        )
        .req_normalizetristate_normalize(
            tooltip="Controls whether the generated noise is normalized to 1.0 strength.",
        )
        .req_customnoise_custom_noise(),
    )

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
            width=float(width),
            height=float(height),
            spatial_compression=8,
            spatial_mode="absolute",
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


class SonarResizedNoiseAdvNode(SonarCustomNoiseNodeBase, SonarNormalizeNoiseNodeMixin):
    DESCRIPTION = "Custom noise type that allows resizing another noise item. Advanced version of the SonarResizedNoise node."

    INPUT_TYPES = SonarLazyInputTypes(
        lambda: NoiseNoChainInputTypes()
        .req_float_width(
            default=32.0,
            min=0.0,
            tooltip="Note: In absolute mode, this should almost always be set to a higher value than the image you're actually sampling.",
        )
        .req_float_height(
            default=32.0,
            min=0.0,
            tooltip="Note: In absolute mode, this should almost always be set to a higher value than the image you're actually sampling.",
        )
        .req_field_spatial_mode(
            ("relative", "percentage", "absolute"),
            default="relative",
            tooltip="In relative mode, the sizes control padding. In percentage mode, the values will be interpreted as percentages of the origal size where 1.0 would be 100%, 0.5 would be 50% and so on. In absolute mode, this controls the absolute size.",
        )
        .req_int_spatial_compression(
            min=1,
            default=8,
            tooltip="Most image models use 8x spatial compression. When spatial mode is absolute, the sizes will be multiplied by this value. It is ignored in percentage mode.",
        )
        .req_field_downscale_strategy(
            ("crop", "scale"),
            default="crop",
            tooltip="Scaling noise is something you'd pretty much only use to create weird effects. For normal workflows, leave this on crop.",
        )
        .req_field_initial_reference(
            ("prefer_crop", "prefer_scale"),
            default="prefer_crop",
            tooltip="The initial latent the noise sampler uses as a reference may not match the requested width/height. This setting controls whether to crop or scale. Note: Cropping can only occur when the initial reference is larger than width/height in both dimensions which is unlikely (and not recommended).",
        )
        .req_field_crop_mode(
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
            default="center",
            tooltip="Note: Crops will have a bias toward the lower number when the size isn't divisible by two. For example, a center crop of size 3 from (0, 1, 2, 3, 4, 5) will result in (1, 2, 3).",
        )
        .req_int_crop_offset_horizontal(
            default=0,
            tooltip="This offsets the cropped view by the specified size. Positive values will move it toward the right, negative values will move it toward the left. The offsets will be adjusted to to fit in the available space. For example, if you have crop_mode set to top_right then setting a positive offset isn't going to do anything: it's already as far right as it can go.",
        )
        .req_int_crop_offset_vertical(
            default=0,
            tooltip="This offsets the cropped view by the specified size. Positive values will move it toward the bottom, negative values will move it toward the top. The offsets will be adjusted to to fit in the available space. For example, if you have crop_mode set to bottom_right then setting a positive offset isn't going to do anything: it's already as far down as it can go.",
        )
        .req_selectscalemode_upscale_mode(
            tooltip="Allows setting the scaling mode when width/height is smaller than the requested size.",
            default="nearest-exact",
        )
        .req_selectscalemode_downscale_mode(
            tooltip="Allows setting the scaling mode when width/height is larger than the requested size and downscale_strategy is set to 'scale'.",
            default="nearest-exact",
        )
        .req_normalizetristate_normalize(
            tooltip="Controls whether the generated noise is normalized to 1.0 strength.",
        )
        .req_customnoise_custom_noise(),
    )

    @classmethod
    def get_item_class(cls):
        return noise.ResizedNoise

    def go(
        self,
        *,
        factor: float,
        width: float,
        height: float,
        spatial_mode: str,
        spatial_compression: int,
        downscale_strategy: str,
        initial_reference: str,
        crop_offset_horizontal: int,
        crop_offset_vertical: int,
        crop_mode: str,
        upscale_mode: str,
        downscale_mode: str,
        normalize: str,
        custom_noise,
    ):
        return super().go(
            factor,
            width=width,
            height=height,
            spatial_compression=spatial_compression,
            spatial_mode=spatial_mode,
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

    INPUT_TYPES = SonarLazyInputTypes(
        lambda: NoiseNoChainInputTypes()
        .req_customnoise_custom_noise(
            tooltip="Custom noise type to filter.",
        )
        .req_float_quantile(
            default=0.85,
            min=-1.0,
            max=1.0,
            step=0.001,
            round=False,
            tooltip="When enabled, will normalize generated noise to this quantile (i.e. 0.75 means outliers >75% will be clipped). Set to 1.0 or 0.0 to disable quantile normalization. A value like 0.75 or 0.85 should be reasonable, it really depends on the input and how many of the values are extreme. (Experimental) You can also use a negative quantile to consider values closest to 0 to be 'extreme'.",
        )
        .req_field_dim(
            ("global", "0", "1", "2", "3", "4"),
            default="1",
            tooltip="Controls what dimensions quantile normalization uses. Dimensions start from 0. Image latents have dimensions: batch, channel, row, column. Video latents have dimensions: batch, channel, frame, row, column.",
        )
        .req_bool_flatten(
            default=True,
            tooltip="Controls whether the noise is flattened before quantile normalization. You can try disabling it but they may have a very strong row/column influence.",
        )
        .req_float_norm_factor(
            default=1.0,
            min=0.00001,
            max=10000.0,
            step=0.001,
            tooltip="Multiplier on the input noise just before it is clipped to the quantile min/max. Generally should be left at the default.",
        )
        .req_float_norm_power(
            default=0.5,
            min=-10000.0,
            max=10000.0,
            step=0.001,
            tooltip="The absolute value of the noise is raised to this power after it is clipped to the quantile min/max. You can use negative values here, but anything below -0.3 will probably produce pretty strange effects. Generally should be left at the default.",
        )
        .req_bool_normalize_noise(
            default=False,
            tooltip="Controls whether the noise source is normalized before quantile filtering occurs.",
        )
        .req_normalizetristate_normalize(
            default="disabled",
            tooltip="Controls whether the generated noise is normalized to 1.0 strength after quantile filtering.",
        )
        .req_field_strategy(
            tuple(utils.quantile_handlers.keys()),
            default="clamp",
            tooltip="Determines how to treat outliers. zero and reverse_zero modes are only useful if you're going to do something like add the result to some other noise. zero will return zero for anything outside the quantile range, reverse_zero only _keeps_ the outliers and zeros everything else.",
        ),
    )

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
    DESCRIPTION = (
        "Custom noise type that allows shuffling noise along dimensions you specify."
    )

    INPUT_TYPES = SonarLazyInputTypes(
        lambda: NoiseNoChainInputTypes()
        .req_customnoise_custom_noise(tooltip="Custom noise type to filter.")
        .req_string_dims(
            default="1,-2,-1",
            tooltip="Comma separated list of dimensions to shuffle. May be negative to count from the end.",
        )
        .req_string_percentages(
            default="1.0,0.25,0.25",
            tooltip="Comma separated list of percentages (0.0 to 1.0 for 100%) of elements to shuffle. Paired with the list of dimensions and wrap if it's shorter. For example, if you specified three dimensions and two percentages, the third dimension in the list would use the first percentage again.",
        )
        .req_bool_fork_rng(
            default=True,
            tooltip="When enabled, the RNG state will be forked to generate the shuffle values.",
        )
        .req_bool_no_identity(
            default=True,
            tooltip="When enabled, ensures shuffle never ends up selecting the original element.",
        ),
    )

    @classmethod
    def get_item_class(cls):
        return noise.ShuffledNoise

    def go(
        self,
        *,
        factor: float,
        dims: str,
        percentages: str,
        fork_rng: bool,
        no_identity: bool,
        custom_noise: object,
    ):
        dims = dims.strip()
        dims = () if not dims else tuple(int(i) for i in dims.split(","))
        percentages = percentages.strip()
        percentages = (
            () if not percentages else tuple(float(p) for p in percentages.split(","))
        )
        return super().go(
            factor,
            noise=custom_noise,
            dims=dims,
            percentages=percentages,
            fork_rng=fork_rng,
            no_identity=no_identity,
        )


class SonarPatternBreakNoiseNode(SonarCustomNoiseNodeBase):
    DESCRIPTION = "Custom noise type that allows breaking patterns in the noise with configurable strength"

    INPUT_TYPES = SonarLazyInputTypes(
        lambda: NoiseNoChainInputTypes()
        .req_customnoise_custom_noise(tooltip="Custom noise type to filter.")
        .req_float_detail_level(
            default=0.0,
            tooltip="Controls the detail level of the noise when break_pattern is non-zero. No effect when strength is 0.",
        )
        .req_selectblend(
            tooltip="Function to use for blending original noise with pattern broken noise. If you have ComfyUI-bleh then you will have access to many more blend modes.",
        )
        .req_float_percentage(
            default=1.0,
            min=0.0,
            max=1.0,
            tooltip="Percentage of elements to shuffle in the specified dimensions.",
        )
        .req_bool_restore_scale(
            default=True,
            tooltip="Controls whether the original min/max values get preserved. Not sure which is better, it is slightly slower to do this though.",
        ),
    )

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

    INPUT_TYPES = SonarLazyInputTypes(
        lambda _yaml_placeholder=_yaml_placeholder: NoiseChainInputTypes()
        .req_bool_normalize_noise(
            default=False,
            tooltip="Controls whether the noise source is normalized before wavelet filtering occurs.",
        )
        .req_normalizetristate_normalize(
            tooltip="Controls whether the generated noise is normalized to 1.0 strength.",
        )
        .opt_customnoise_custom_noise(
            tooltip="Optional: Custom noise input. If unconnected will default to Gaussian noise.",
        )
        .opt_customnoise_custom_noise_high(
            tooltip="Optional: Custom noise input. If unconnected will use the same noise generator as custom_noise. However, if you do connect it this noise will be used for the high-frequency side of the wavelet.",
        )
        .opt_yaml(placeholder=_yaml_placeholder),
    )

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

    INPUT_TYPES = SonarLazyInputTypes(
        lambda: NoiseChainInputTypes()
        .req_string_padding_mode(
            default="symmetric",
            tooltip="This is just passed to the pytorch_wavelets scatternet constructor. Valid padding modes that I know of (second order only supports symmetric and zero): symmetric, reflect, zero, periodization, constant, replicate, periodic",
        )
        .req_bool_use_symmetric_filter(
            default=False,
            tooltip="Slower, but possibly higher quality.",
        )
        .req_float_magbias(
            default=1e-02,
            min=-1000.0,
            max=1000.0,
            tooltip="Magnitude bias. Changing it doesn't seem to affect anything, but you can try.",
        )
        .req_float_output_offset(
            default=0.0,
            min=-100000.0,
            max=100000.0,
            tooltip="Controls where the output starts. The beginning is the low frequency bands, the end is high frequencies. If less than 1 (positive or negative) it will be treated as a percentage into the dimension. Negative values count from the end.",
        )
        .req_field_output_mode(
            (
                "channels_adjusted",
                "flat_adjusted",
                "channels",
                "flat",
                "channels_scaled",
                "flat_scaled",
            ),
            default="channels_adjusted",
            tooltip="The normal scatternet reduces the spatial dimensions 2x, the second order one 4x. The adjusted modes will generate larger noise (in the spatial dimensions) to compensate, this is slower but gives you a lot more room to work with. The scaled modes will just scale the noise to compensate (likely doesn't work well). Modes that start with channels will index along the channel dimension, otherwise the indexing will be flat (after the batch dimension). Note: I recommend channels_adjusted mode, it's very possible the offset indexing math is wrong for other modes.",
        )
        .req_int_scatternet_order(
            default=1,
            min=-3,
            max=3,
            tooltip="Each order increases the number of channels exponentially. You can use a primitive node to bypass the limit of 3 here if you're a crazy person, the code will handle any value but you're very likely to die of old age or run out of VRAM or both if you go above 3 (and even that is stretching it). You can set this to 0 to disable scatternet filtering quickly. Negative values are the same as positive ones here with one exception: there's a specialized 2nd order scatternet which will be used by default for order 2, however it may not support the normal parameters (like padding modes). Use -2 here if you just want to stack two normal scatternet layers instead.",
        )
        .req_bool_per_channel_scatternet(
            default=False,
            tooltip="Runs the scatternet on each channel separately. May be very slow. Models like SDXL use 4 channels, models like Flux have 16. Enabling this may help with non-adjusted output modes.",
        )
        .req_bool_normalize_noise(
            default=False,
            tooltip="Controls whether the noise source is normalized before scatternet filtering occurs.",
        )
        .req_normalizetristate_normalize(
            tooltip="Controls whether the generated noise is normalized to 1.0 strength.",
        )
        .opt_customnoise_custom_noise(
            tooltip="Optional: Custom noise input. If unconnected will default to Gaussian noise.",
        ),
    )

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

    INPUT_TYPES = SonarLazyInputTypes(
        lambda: NoiseChainInputTypes()
        .req_customnoise_custom_noise()
        .req_field_mode(
            ("sin", "cos", "sin_copysign", "cos_copysign"),
            default="cos",
            tooltip="Function to use for rippling. The copysign variations are not recommended, they will force the noise to the sign of the wave (whether it's above or below the midline) which has an extremely strong effect. If you want to try it, use something like a 1:16 ratio or higher with normal noise.",
        )
        .req_int_dim(
            default=-1,
            min=-100,
            max=100,
            tooltip="Dimension to use for the ripple effect. Negative dimensions count from the end where -1 is the last dimension.",
        )
        .req_bool_flatten(
            default=False,
            tooltip="When enabled, the noise will be flattened starting from (and including) the specified dimension.",
        )
        .req_float_offset(
            default=0.0,
            min=-10000,
            max=10000.0,
            tooltip="Simple addition to the base value used for the wave.",
        )
        .req_float_roll(
            default=0.0,
            min=-10000,
            max=10000.0,
            tooltip="Rolls the wave by this many elements each time the noise generator is called. Negative values roll backward.",
        )
        .req_float_amplitude_high(
            default=0.25,
            min=-10000,
            max=10000.0,
            tooltip="Scale for noise at the highest point of the wave. This adds to the base value (respecting sign). For example, if set to 0.25 you will get noise * 1.25 at that point. It's also possible to use negative values, -0.25 will result in noise * -1.25.",
        )
        .req_float_amplitude_low(
            default=0.15,
            min=-10000,
            max=10000.0,
            tooltip="Scale for noise at the lowest point of the wave. This subtracts from the base value (respecting sign). For example, if set to 0.25 you will get noise * 0.75 at that point. It's also possible to use negative values, -0.25 will result in noise * -0.75.",
        )
        .req_float_period(
            default=3.0,
            min=-10000,
            max=10000.0,
            tooltip="Number of oscillations along the specified dimension.",
        )
        .req_bool_normalize_noise(
            default=False,
            tooltip="Controls whether the noise source is normalized before wavelet filtering occurs.",
        )
        .req_normalizetristate_normalize(
            tooltip="Controls whether the generated noise is normalized to 1.0 strength.",
        ),
    )

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

    INPUT_TYPES = SonarLazyInputTypes(
        lambda: NoiseChainInputTypes()
        .req_customnoise_custom_noise()
        .req_float_min_negative_value(
            default=-4.5,
            min=-10000.0,
            max=10000.0,
            tooltip="In simple mode, this is just the lowest value in the range (and can be positive, despite the name). In advanced mode, this controls the minimum negative value. If you set it to 0 or higher then normalization will leave negative values alone.",
        )
        .req_float_max_negative_value(
            default=0.0,
            min=-10000.0,
            max=10000.0,
            tooltip="Not used in simple mode. In advanced mode, this controls the maximum negative value. If you set it to 0 or higher, a maximum negative value will be automatically determined from negative value closest (but not equal to) zero.",
        )
        .req_float_min_positive_value(
            default=0.0,
            min=-10000.0,
            max=10000.0,
            tooltip="Not used in simple mode. In advanced mode, this controls the minmum positive value. If you set it to 0 or lower, a minimum positive value will be automatically determined from positive value closest (but not equal to) zero.",
        )
        .req_float_max_positive_value(
            default=4.5,
            min=-10000.0,
            max=10000.0,
            tooltip="In simple mode, this is just the highest value in the range (and can be negative, despite the name). In advanced mode, this controls the maximum positive value. If you set it to 0 or lower then normalization will leave positive values alone.",
        )
        .req_field_mode(
            ("simple", "advanced"),
            default="simple",
            tooltip="There are several modes:\nsimple: The noise will be rebalanced to be in between min_negative_value and max_positive_value. Though it sounds weird, you don't need to respect the positive/negative in the names. It is just treated as a simple range.\nadvanced: Positive and negative values in the noise are separately rebalanced to be between the specified ranges. If you set max_negative_value to something positive or min_positive_value to something negative this will automatically determine whatever the closest value to zero is for each sign. Additionally, if you set max_positive_value to something negative or min_negative_value to something positive then values for that sign will be left alone.",
        )
        .req_string_dims(
            default="-3, -2, -1",
            tooltip="A comma separated list of dimensions which can be negative to count from the end. This behaves differently in advanced mode: If left blank, normalization will be global. If set to anything, normalization will be over each batch item separately. The actual values of the dimensions are ignored in advanced mode currently.",
        )
        .req_string_std_dims(
            default="-3, -2, -1",
            tooltip="A comma separated list of dimensions which can be negative to count from the end.",
        )
        .req_float_std_multiplier(
            default=1.0,
            min=-10000.0,
            max=10000.0,
            tooltip="Multiplier on the distance of the std from 1.0. The noise will be divided by this. You can set it to 1.0 to skip the division. When enabled, the division occurs before the final normalize and scaling and after the min/max value parameters are applied.",
        )
        .req_string_mean_dims(
            default="-3, -2, -1",
            tooltip="A comma separated list of dimensions which can be negative to count from the end.",
        )
        .req_float_mean_multiplier(
            default=1.0,
            min=-10000.0,
            max=10000.0,
            tooltip="Multiplier on the mean of the noise. The mean will be subtracted from the noise if it's not 0. This occurs before the final normalize and scaling and after the min/max value parameters are applied.",
        )
        .req_bool_normalize_noise(
            default=False,
            tooltip="Controls whether the noise source is normalized immediately after generation.",
        )
        .req_normalizetristate_normalize(
            default="disabled",
            tooltip="Controls whether the generated noise is normalized to 1.0 strength. Enabling this does the same thing as the default mean/std settings.",
        ),
    )

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
        std_dims: str,
        std_multiplier: float,
        mean_dims: str,
        mean_multiplier: float,
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
            std_dims=()
            if not std_dims.strip()
            else tuple(int(i) for i in dims.split(",")),
            std_multiplier=std_multiplier,
            mean_dims=()
            if not mean_dims.strip()
            else tuple(int(i) for i in dims.split(",")),
            mean_multiplier=mean_multiplier,
            normalize=self.get_normalize(normalize),
            normalize_noise=normalize_noise,
            noise=custom_noise,
        )


class SonarPerDimNoiseNode(
    SonarCustomNoiseNodeBase,
    SonarNormalizeNoiseNodeMixin,
):
    DESCRIPTION = "Custom noise type that allows calling the noise sampler multiple times along a dimension. Can be useful for stuff like moving slices of 3D Perlin noise into the batch dimension."

    INPUT_TYPES = SonarLazyInputTypes(
        lambda: NoiseChainInputTypes()
        .req_customnoise_custom_noise()
        .req_int_dim(
            default=0,
            min=-100,
            max=100,
            tooltip="Dimension to use. The default usually corresponds to the batch. Be careful using dimensions above 1 as those tend to be spatial and you might end up calling a slow noise sampler hundreds of times.",
        )
        .req_bool_shrink_dim(
            default=False,
            tooltip="When enabled, the reference latent will be chunk_size in the specified dimension. When disabled, noise will be generated according to the initial latent size and then sliced along the specified dimension. Enabling it should be considerably faster/more memory efficient but may not work well for some noise types.",
        )
        .req_int_chunk_size(
            default=1,
            min=1,
            max=10000,
            tooltip="Can be used to control how many times the noise sampler is called. For example, if you have dim=0, chunk_size=2 and are dealing with a batch of 4, this will call the noise sampler twice, taking the first two items from the first call and the last two items from the second call.",
        )
        .req_bool_normalize_noise(
            default=False,
            tooltip="Controls whether the noise source is normalized initially.",
        )
        .req_normalizetristate_normalize(
            default="disabled",
            tooltip="Controls whether the generated noise is normalized to 1.0 strength.",
        ),
    )

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
            offset=0,
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

    INPUT_TYPES = SonarLazyInputTypes(
        lambda: NoiseChainInputTypes()
        .req_customnoise_custom_noise()
        .req_bool_normalize_noise(
            default=False,
            tooltip="Controls whether the noise source is normalized initially.",
        )
        .req_normalizetristate_normalize(
            default="disabled",
            tooltip="Controls whether the generated noise is normalized to 1.0 strength.",
        )
        .opt_field_operation_1(
            "LATENT_OPERATION",
            tooltip="Optional LATENT_OPERATION. The operations will be applied in sequence.",
        )
        .opt_field_operation_2(
            "LATENT_OPERATION",
            tooltip="Optional LATENT_OPERATION. The operations will be applied in sequence.",
        )
        .opt_field_operation_3(
            "LATENT_OPERATION",
            tooltip="Optional LATENT_OPERATION. The operations will be applied in sequence.",
        )
        .opt_field_operation_4(
            "LATENT_OPERATION",
            tooltip="Optional LATENT_OPERATION. The operations will be applied in sequence.",
        )
        .opt_field_operation_5(
            "LATENT_OPERATION",
            tooltip="Optional LATENT_OPERATION. The operations will be applied in sequence.",
        ),
    )

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


class SonarCustomNoiseParametersNode(
    SonarCustomNoiseNodeBase,
    SonarNormalizeNoiseNodeMixin,
):
    DESCRIPTION = "Custom noise type that allows setting parameters like dtype or forking the RNG."

    INPUT_TYPES = SonarLazyInputTypes(
        lambda: NoiseNoChainInputTypes()
        .req_customnoise_custom_noise()
        .req_int_rng_state_offset(
            default=0,
            min=0,
            tooltip="In other words, seed. Avoiding using the word seed here to suppress ComfyUI's annoying default behavior. If you want stuff like auto-increment you can connect an INT primitive node.",
        )
        .req_field_rng_offset_mode(
            ("disabled", "override", "add"),
            default="disabled",
            tooltip="Controls the seed passed to the noise sampler and also seeding when rng_mode is set to separate. Most noise samplers don't care about the seed so this generally will only have an effect in when rng_mode is set to separate.",
        )
        .req_field_rng_mode(
            ("default", "separate", "fork"),
            default="default",
            tooltip="default mode doesn't do anything special. separate mode creates a generator and saves/restores the state when generating noise (also includes the Python random module). fork uses the existing RNG state (for both Torch and Python random module) but restores it to whatever it was before the custom noise was called.",
        )
        .req_bool_frames_to_channels(
            tooltip="Only applicable for 5D latents (video models). Will move the frame dimension into channels, may be necessary if a noise type can't deal with 5D latents directly. It's safe to enable this for all models.",
        )
        .req_bool_ensure_square_aspect_ratio(
            tooltip="Will rearrange the height/width sizes to be square, padding with zeros if necessary. May help some noise types work better with extreme aspect ratios, can also deal with 3D (1 spatial dimension) latents.",
        )
        .req_bool_fix_invalid(
            tooltip="Replaces any NaNs or infinite values with 0.",
        )
        .req_field_override_dtype(
            (
                "default",
                "float64",
                "float32",
                "float16",
                "bfloat16",
                "float8_e4m3fn",
                "float8_e4m3fnuz",
                "float8_e5m2",
                "float8_e5m2fnuz",
                "float8_e8m0fnu",
                "int64",
                "int32",
                "int16",
                "int8",
            ),
            default="default",
            tooltip="Can be used to override the dtype the noise is generated with. Not all noise generators support all types. I don't recommend using the int or float8 types. Probably the most useful override is float64.",
        )
        .req_field_override_device(
            ("default", "cpu", "gpu"),
            default="default",
            tooltip="default just uses whatever device normally would be used. gpu will use ComfyUI's default GPU device and also toggle the cpu_noise flag off. cpu will use the CPU device and toggle the cpu_noise flag on.",
        )
        .req_normalizetristate_normalize(),
    )

    @classmethod
    def get_item_class(cls):
        return noise.CustomNoiseParametersNoise

    def go(
        self,
        *,
        factor,
        rng_state_offset: int,
        rng_offset_mode: str,
        rng_mode: str,
        frames_to_channels: bool,
        ensure_square_aspect_ratio: bool,
        fix_invalid: bool,
        override_dtype: str,
        override_device: str,
        normalize: str,
        custom_noise: object,
    ):
        valid_dtypes = {
            "default",
            "float64",
            "float32",
            "float16",
            "bfloat16",
            "float8_e4m3fn",
            "float8_e4m3fnuz",
            "float8_e5m2",
            "float8_e5m2fnuz",
            "float8_e8m0fnu",
            "int64",
            "int32",
            "int16",
            "int8",
        }
        dt = getattr(torch, override_dtype, None)
        if override_dtype not in valid_dtypes or (
            override_dtype != "default" and dt is None
        ):
            raise ValueError("Bad dtype, may not be supported by your PyTorch version")
        if override_device == "default":
            device = None
        elif override_device == "cpu":
            device = "cpu"
        elif override_device == "gpu":
            device = model_management.get_torch_device()
        return super().go(
            factor,
            rng_state_offset=rng_state_offset,
            rng_offset_mode=rng_offset_mode,
            rng_mode=rng_mode,
            frames_to_channels=frames_to_channels,
            ensure_square_aspect_ratio=ensure_square_aspect_ratio,
            fix_invalid=fix_invalid,
            override_dtype=dt,
            override_device=device,
            normalize=normalize,
            noise=custom_noise,
        )


NODE_CLASS_MAPPINGS = {
    "SonarBlendedNoise": SonarBlendedNoiseNode,
    "SonarChannelNoise": SonarChannelNoiseNode,
    "SonarCompositeNoise": SonarCompositeNoiseNode,
    "SonarCustomNoiseParameters": SonarCustomNoiseParametersNode,
    "SonarGuidedNoise": SonarGuidedNoiseNode,
    "SonarLatentOperationFilteredNoise": SonarLatentOperationFilteredNoiseNode,
    "SonarModulatedNoise": SonarModulatedNoiseNode,
    "SonarNormalizeNoiseToScale": SonarNormalizeNoiseToScaleNode,
    "SonarPatternBreakNoise": SonarPatternBreakNoiseNode,
    "SonarPerDimNoise": SonarPerDimNoiseNode,
    "SonarQuantileFilteredNoise": SonarQuantileFilteredNoiseNode,
    "SonarRandomNoise": SonarRandomNoiseNode,
    "SonarRepeatedNoise": SonarRepeatedNoiseNode,
    "SonarResizedNoise": SonarResizedNoiseNode,
    "SonarResizedNoiseAdv": SonarResizedNoiseAdvNode,
    "SonarRippleFilteredNoise": SonarRippleFilteredNoiseNode,
    "SonarScatternetFilteredNoise": SonarScatternetFilteredNoiseNode,
    "SonarScheduledNoise": SonarScheduledNoiseNode,
    "SonarShuffledNoise": SonarShuffledNoiseNode,
    "SonarWaveletFilteredNoise": SonarWaveletFilteredNoiseNode,
}
