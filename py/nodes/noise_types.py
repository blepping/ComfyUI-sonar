from __future__ import annotations

import torch

from .. import noise, utils
from ..noise_generation import DistroNoiseGenerator, VoronoiNoiseGenerator
from .base import (
    NoiseChainInputTypes,
    SonarCustomNoiseNodeBase,
    SonarLazyInputTypes,
    SonarNormalizeNoiseNodeMixin,
)


class SonarAdvancedPyramidNoiseNode(SonarCustomNoiseNodeBase):
    DESCRIPTION = (
        "Custom noise type that allows specifying parameters for Pyramid variants."
    )

    INPUT_TYPES = SonarLazyInputTypes(
        lambda: NoiseChainInputTypes()
        .req_field_variant(
            (
                "highres_pyramid",
                "pyramid",
                "pyramid_old",
            ),
            default="highres_pyramid",
            tooltip="Sets the Pyramid noise variant to generate.",
        )
        .req_int_iterations(
            default=-1,
            min=-1,
            max=8,
            tooltip="When set to -1 will use the variant default.",
        )
        .req_float_discount(
            default=0.0,
            tooltip="When set to 0 will use the variant default.",
        )
        .req_selectscalemode_upscale_mode(
            insert_modes=("default",),
            default="default",
            tooltip="Allows setting the scaling mode for Pyramid noise. Leave on default to use the variant default.",
        ),
    )

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

    INPUT_TYPES = SonarLazyInputTypes(
        lambda: NoiseChainInputTypes()
        .req_float_alpha(
            default=0.25,
            tooltip="Similar to the advanced power noise node, positive values increase low frequencies (with colorful effects), negative values increase high frequencies.",
        )
        .req_float_k(
            default=1.0,
            tooltip="Currently no description of exactly what it does, it's just another knob you can try turning for a different effect.",
        )
        .req_float_vertical_factor(
            default=1.0,
            tooltip="Vertical frequency scaling factor.",
        )
        .req_float_horizontal_factor(
            default=1.0,
            tooltip="Horizontal frequency scaling factor.",
        )
        .req_bool_use_sqrt(
            default=True,
            tooltip="Controls whether to sqrt when dividing the FFT. Negative hfac/wfac won't work when enabled. Turning it off seems to make the parameters have a much stronger effect.",
        ),
    )

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
    DESCRIPTION = "Custom noise type that allows specifying parameters for power law (grey, violet, etc) variants."

    INPUT_TYPES = SonarLazyInputTypes(
        lambda: NoiseChainInputTypes()
        .req_float_alpha(
            default=0.5,
            tooltip="Similar to the advanced power noise node, positive values increase low frequencies (with colorful effects), negative values increase high frequencies.",
        )
        .req_field_div_max_dims(
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
            default="non-batch",
            tooltip="If non-none, the noise gets divide by the maximum over this dimension.",
        )
        .req_bool_use_div_max_abs(
            default=True,
            tooltip="Only has an effect when div_max_dims is not none. Controls whether maximization is done with the absolute values or raw values.",
        )
        .req_bool_use_sign(
            tooltip="When set, only the sign of the initial noise is used, so -0.5, -0.2 all turn into -1, 0.5, 2, etc all turn into 1.",
        ),
    )

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

    INPUT_TYPES = SonarLazyInputTypes(
        lambda: NoiseChainInputTypes()
        .req_bool_adjust_scale(
            default=False,
            tooltip="When enabled, the output will be normalized to values between -1 and 1 using the last two dimensions (if there are four or more), otherwise dimensions after the first.",
        )
        .req_string_chain_length(
            default="1, 1, 2, 2, 3, 3",
            tooltip="Comma-separated list of chain lengths. Cannot be empty. Iterations will cycle through the list and wrap. Controls the length of Collatz chains. Note: Using a high chain length may be very slow, especially if combined with many iterations.",
        )
        .req_int_chain_offset(
            default=5,
            min=0,
            max=10000,
            tooltip="Uses values starting at the specified offset. Note: This entails generating chains of length chain_length + chain_offset, which may be quite slow if you use high values.",
        )
        .req_int_iterations(
            default=10,
            min=1,
            max=10000,
            tooltip="Number of iterations to run. Warning: Collatz noise (my implementation, anyway) is EXTREMELY slow.",
        )
        .req_bool_iteration_sign_flipping(
            default=True,
            tooltip="Controls whether we cycle between flipping the sign on the output from each iteration. May average out weirdness... Or make stuff weirder.",
        )
        .req_float_rmin(
            default=-8000.0,
            tooltip="Minimum value a chain can start with. Going as low as -9500 should be safe with float32.",
        )
        .req_float_rmax(
            default=8000.0,
            tooltip="Maximum value a chain can start with. I don't recommend going over 9500 if you are using the float32 dtype here as that is where the Collatz chain starts to reach values that can't be accurately represented.",
        )
        .req_string_dims(
            default="-1, -1, -2, -2",
            tooltip="Comma-separated list of dimensions. Cannot be empty. May be negative to count from the end of the list. Iterations will cycle through the list and wrap.",
        )
        .req_bool_flatten(
            tooltip="Controls whether dimensions past the current one selected from the dims parameter will get flattened.",
        )
        .req_field_output_mode(
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
            default="values",
        )
        .req_float_quantile(
            default=0.5,
            min=0.0,
            max=1.0,
            tooltip="The initial output of each iteration will be run through quantile normalization. Setting the parameter to 0 or 1 will disable quantile normalization.",
        )
        .req_field_quantile_strategy(
            tuple(utils.quantile_handlers.keys()),
            default="clamp",
            tooltip="Determines how to treat outliers. zero and reverse_zero modes are only useful if you're going to do something like add the result to some other noise. zero will return zero for anything outside the quantile range, reverse_zero only _keeps_ the outliers and zeros everything else.",
        )
        .req_field_noise_dtype(
            ("float32", "float64", "float16", "bfloat16"),
            default="float32",
            tooltip="Generally should be left at the default. Only float32 and float64 will work if you have quantile normalization enabled.",
        )
        .req_float_even_multiplier(
            default=0.5,
            tooltip="Multiplier to use when the previous link in the chain is even. Collatz uses 0.5 (divides by two) here.",
        )
        .req_float_even_addition(
            default=0.0,
            tooltip="Value to add when the previous link in the chain is even. Collatz uses 0 here.",
        )
        .req_float_odd_multiplier(
            default=3.0,
            tooltip="Multiplier to use when the previous link in the chain is odd. Collatz uses 3 here.",
        )
        .req_float_odd_addition(
            default=1.0,
            tooltip="Value to add when the previous link in the chain is odd. Collatz uses 1 here.",
        )
        .req_bool_integer_math(
            default=True,
            tooltip="Controls whether the results during chain generation get truncated to an integer value or not. Should be enabled if you actually want to generate accurate Collatz chains.",
        )
        .req_bool_add_preserves_sign(
            default=True,
            tooltip="Controls whether additions use the same sign as the item they're being added to.",
        )
        .req_bool_break_loops(
            default=True,
            tooltip="Controls whether the chain resets back to the seed value once it reaches 1 or 0. Generally should be left enabled, otherwise the chain will oscillate between only a few values for the rest of the length (at least with the Collatz rules).",
        )
        .req_field_seed_mode(
            ("default", "force_odd", "force_even"),
            default="default",
            tooltip="Default mode just uses whatever the original seed value was. force_odd/force_even will force it to the specified parity by adding one if it doesn't match. Starting from odd seeds might result in longer chains. Enabling the force modes may cause the initial seeds to exceed rmax by one.",
        )
        .opt_customnoise_seed_custom_noise(
            tooltip="Optional custom noise to use for initial values for Collatz chains. May be slow as it will generate noise according to the original input size and then crop it. Does this noise type have enough warnings about it being slow? Yeah. Connecting something here will probably make it even slower!",
        )
        .opt_customnoise_mix_custom_noise(
            tooltip="Optional custom noise to use with the output modes starting with 'noise'.",
        ),
    )

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
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.001,
                    "round": False,
                    "tooltip": "When enabled, will normalize generated noise to this quantile (i.e. 0.75 means outliers >75% will be clipped). Set to 1.0 or 0.0 to disable quantile normalization. A value like 0.75 or 0.85 should be reasonable, it really depends on the distribution and how many of the values are extreme. (Experimental) You can use a negative quantile to consider the values closest to zero as extreme.",
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


class SonarWaveletNoiseNode(
    SonarCustomNoiseNodeBase,
    SonarNormalizeNoiseNodeMixin,
):
    DESCRIPTION = "Custom noise type that allows generating wavelet noise. Very simple explanation of how a single octave works:\n1) Generate some noise.\n2) Scale it down 50%.\n3) Scale it back up to the original size.\n4) Subtract the scaled noise from the original noise.\nScaling the noise down and then back up blurs it, so this is essentially sharpening the noise."

    INPUT_TYPES = SonarLazyInputTypes(
        lambda: NoiseChainInputTypes()
        .req_int_octaves(
            default=4,
            min=-100,
            max=100,
            tooltip="Number of octaves to generate. You can use a negative number here to run the octaves in reverse order though it may produce weird results/not work very well.",
        )
        .req_float_octave_height_factor(
            default=0.5,
            min=0.001,
            tooltip="Wavelet noise works by scaling noise by this factor in each octave, then scaling it back up to the original size. After that, the scaled noise is subtracted from the original noise.",
        )
        .req_float_octave_width_factor(
            default=0.5,
            min=0.001,
            tooltip="Wavelet noise works by scaling noise by this factor in each octave, then scaling it back up to the original size. After that, the scaled noise is subtracted from the original noise.",
        )
        .req_selectscalemode_octave_scale_mode(
            default="adaptive_avg_pool2d",
            tooltip="Scaling mode used within each octave to produce the scaled noise. By default this will be scaling down that octave's noise.",
        )
        .req_selectscalemode_octave_rescale_mode(
            default="bilinear",
            tooltip="Scaling mode used within each octave to scale the noise back up to that octave's original size.",
        )
        .req_selectscalemode_post_octave_rescale_mode(
            default="bilinear",
            tooltip="Scaling mode used to scale the output of an octave back up to the actual latent size.",
        )
        .req_float_initial_amplitude(
            default=1.0,
            tooltip="Basically the strength an octave gets added to the total. This will be scaled by persistance after each octave.",
        )
        .req_float_persistence(
            default=0.5,
            tooltip="Multiplier applied to amplitude after each octave. 0.5 means the first octave uses initial_amplitude, the second uses half of that and so on.",
        )
        .req_float_height_factor(
            default=2.0,
            min=0.001,
            tooltip="Scaling factor for height, calculated after each octave. 2.0 means divide by two. Note: It's possible to use values below 1 here but be careful as it's very easy to reach absurd latent sizes with only a few octaves.",
        )
        .req_float_width_factor(
            tooltip="Scaling factor for width, calculated after each octave. 2.0 means divide by two. Note: It's possible to use values below 1 here but be careful as it's very easy to reach absurd latent sizes with only a few octaves.",
            default=2.0,
            min=0.001,
        )
        .req_float_update_blend(
            tooltip="Controls how original_noise - scaled_noise is blended with original_noise. The default is to use 100% original_noise - scaled_noise.",
            default=1.0,
        )
        .req_selectblend_update_blend_mode(
            insert_modes=("simple_add",),
            default="lerp",
            tooltip="Controls how the enhanced noise from each octave is blended with that octave's raw noise. With normal wavelet noise there's no blending and you use 100% enhanced noise.",
        )
        .req_bool_normalize_noise(
            tooltip="Controls whether the noise source is normalized before wavelet filtering occurs.",
        )
        .req_normalizetristate_normalize()
        .opt_customnoise_custom_noise(
            tooltip="Optional: Custom noise input. If unconnected will default to Gaussian noise. Note: When connected, the noise for all octaves will be generated at the maximum scale and then cropped which may be slow.",
        ),
    )

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


class SonarAdvancedVoronoiNoiseNode(SonarCustomNoiseNodeBase):
    DESCRIPTION = "Voronoi noise is a very weird noise type. The default settings are just borderline usable with SDXL at a 20% ratio with normal Gaussian noise. I recommend reading the section on this noise type in the project documentation (under advanced noise types) as there are too many features to describe in the node itself."

    INPUT_TYPES = SonarLazyInputTypes(
        lambda _pretty_distance_modes=", ".join(  # noqa: B008
            sorted(VoronoiNoiseGenerator.voronoi_distance_modes),  # noqa: B008
        ),
        _pretty_result_modes=", ".join(  # noqa: B008
            sorted(VoronoiNoiseGenerator.voronoi_result_modes),  # noqa: B008
        ): NoiseChainInputTypes()
        .req_string_n_points(
            default="256",
            tooltip="Controls the number of features points in the generated noise. Higher generally results in more detail/better results but is slower. May be a comma separated list for each octave (only applicable when octave mode is set to new_features). 2 is the minimum value.",
        )
        .req_string_distance_mode(
            default="euclidean",
            placeholder=f"One of: {_pretty_distance_modes}",
            tooltip="Distance modes. You can specify a comma-separated list of items which will be used for each octave.\n"
            "You can specify an average of multiple distance modes by separating the names with +.\n"
            "Some modes can take arguments. Example syntax: modename:argname=value:argname=value\n"
            "All modes support scaling their output with dscale (which defaults to 1).\n"
            f"Possible distance modes: {_pretty_distance_modes}",
        )
        .req_float_z_initial(
            default=0.0,
            tooltip="Initial value for z (depth).",
        )
        .req_float_z_increment(
            default=1.0,
            tooltip="Amount z (depth) is incremented when applicable.",
        )
        .req_float_z_max(
            default=9999.0,
            tooltip="Maximum difference from the intial value. At that point, z_max_mode will apply. When set to 0, z_increment has no effect and you will get different noise each time you call the noise sampler.",
        )
        .req_field_z_max_mode(
            (
                "reset",
                "wrap",
                "bounce",
            ),
            default="reset",
            tooltip="Controls what happens when the z_max limit is hit (see tooltip for z_max). Reset will reset the feature points and z to the initial values. Wrap will reset z to the initial value. Bounce will flip the sign on the increment and do an increment.",
        )
        .req_string_result_mode(
            default="diff2",
            placeholder=f"One of: {_pretty_result_modes}",
            tooltip="Result modes. You can specify a comma-separated list of items which will be used for each octave.\n"
            "You can specify an average of multiple result modes by separating the names with +.\n"
            "Some modes can take arguments. Example syntax: modename:argname=value:argname=value\n"
            "All modes support scaling their output with rscale (which defaults to 1).\n"
            f"Possible result modes: {_pretty_result_modes}",
        )
        .req_field_octave_mode(
            (
                "same_features",
                "new_features",
                "same_invert_odd",
                "same_invert_even",
                "same_roll_chan_up",
                "same_roll_chan_down",
                "same_roll_dir_up",
                "same_roll_dir_down",
            ),
            default="new_features",
            tooltip="Only relevant when generating multiple octaves. Controls whether octaves share a set of feature points or if they are different for each octave (note that this is slower). Modes starting with 'same' will use the same feature points per octave but may transform them.",
        )
        .req_int_octaves(
            default=3,
            min=1,
            tooltip="Number of octaves of noise to generate.",
        )
        .req_float_gain(default=0.75)
        .req_float_lacunarity(default=2.0)
        .req_float_initial_amplitude(default=1.0)
        .req_float_initial_scale(default=1.0)
        .req_normalizetristate_normalize()
        .opt_customnoise(
            "custom_noise",
            tooltip="Optional input if you want to use some other noise type for the initial feature points. Won't work well with noise types that care about the content of the latent (I think only spectral modulation) or manage their own seed (I believe this only applies to Brownian or if you're using the custom noise parameters node to override seeds/fork the RNG).",
        ),
    )

    @classmethod
    def get_item_class(cls):
        return noise.AdvancedVoronoiNoise

    def go(
        self,
        *,
        factor: float,
        rescale: float,
        n_points: str,
        distance_mode: str,
        z_initial: float,
        z_increment: float,
        z_max: float,
        z_max_mode: str,
        result_mode: str,
        octave_mode: str,
        octaves: int,
        gain: float,
        lacunarity: float,
        initial_amplitude: float,
        initial_scale: float,
        normalize: str,
        custom_noise=None,
        sonar_custom_noise_opt=None,
    ):
        n_points = tuple(int(v) for v in n_points.split(","))
        distance_mode = tuple(v.strip() for v in distance_mode.split(","))
        result_mode = tuple(v.strip() for v in result_mode.split(","))
        return super().go(
            factor,
            rescale=rescale,
            sonar_custom_noise_opt=sonar_custom_noise_opt,
            n_points=n_points,
            distance_mode=distance_mode,
            z_initial=z_initial,
            z_increment=z_increment,
            z_max=z_max,
            z_max_mode=z_max_mode,
            result_mode=result_mode,
            octave_mode=octave_mode,
            octaves=octaves,
            gain=gain,
            lacunarity=lacunarity,
            initial_amplitude=initial_amplitude,
            initial_scale=initial_scale,
            custom_noise=custom_noise,
            normalize=normalize,
        )


NODE_CLASS_MAPPINGS = {
    "SonarAdvancedPyramidNoise": SonarAdvancedPyramidNoiseNode,
    "SonarAdvanced1fNoise": SonarAdvanced1fNoiseNode,
    "SonarAdvancedPowerLawNoise": SonarAdvancedPowerLawNoiseNode,
    "SonarAdvancedCollatzNoise": SonarAdvancedCollatzNoiseNode,
    "SonarAdvancedDistroNoise": SonarAdvancedDistroNoiseNode,
    "SonarAdvancedVoronoiNoise": SonarAdvancedVoronoiNoiseNode,
    "SonarWaveletNoise": SonarWaveletNoiseNode,
}
