from __future__ import annotations

import functools
import math
from typing import TYPE_CHECKING

from .. import utils
from ..external import IntegratedNode
from ..latent_ops import (
    SonarLatentOperation,
    SonarLatentOperationAdvanced,
    SonarLatentOperationNoise,
    SonarLatentOperationSetSeed,
)
from .base import SonarInputTypes, SonarLazyInputTypes
from .noise_filters import SonarQuantileFilteredNoiseNode

if TYPE_CHECKING:
    import torch


class SonarApplyLatentOperationCFG(metaclass=IntegratedNode):
    DESCRIPTION = "Allows applying a LATENT_OPERATION during sampling. ComfyUI has a few that are builtin and this node pack also includes: SonarLatentOperationQuantileFilter."
    RETURN_TYPES = ("MODEL",)
    CATEGORY = "latent/advanced/operations"

    FUNCTION = "go"

    INPUT_TYPES = SonarLazyInputTypes(
        lambda: SonarInputTypes()
        .req_model()
        .req_field_mode(
            (
                "cond_sub_uncond",
                "denoised_sub_uncond",
                "uncond_sub_cond",
                "denoised",
                "cond",
                "uncond",
                "model_input",
            ),
            default="cond_sub_uncond",
            tooltip="cond_sub_uncond is what ComfyUI's latent operations use. The non-sub_uncond modes likely won't work with pred_flip mode enabled. If you have anything but the denoised options selected, this will use pre-CFG, otherwise it will use post-CFG (unless you are using model_input).",
        )
        .req_bool_pred_flip_mode(
            tooltip="Lets you try to apply the latent operation to the noise prediction rather than the image prediction. Doesn't work properly with the non-sub_uncond modes. No real reason it should be better, just something you can try. Note: The noise prediction gets scaled by the sigma first, in case that's useful information.",
        )
        .req_bool_require_uncond(
            tooltip="When enabled, the operation will be skipped if uncond is unavailable. This will also happen if you choose a mode that requires uncond.",
        )
        .req_float_start_sigma(
            default=-1.0,
            min=-1.0,
            tooltip="First sigma the effect becomes active. You can set a negative value here to use whatever the model's maximum sigma is.",
        )
        .req_float_end_sigma(
            default=0.0,
            min=0.0,
            tooltip="Last sigma the effect is active.",
        )
        .req_selectblend_blend_mode(
            tooltip="Controls how the output of the latent operation is blended with the original result.",
        )
        .req_float_blend_strength(
            default=0.5,
            tooltip="Strength of the blend. For a normal blend mode like LERP, 1.0 means use 100% of the output from the latent operation, 0.0 means use none of it and only the original value. Note: Blending is applied to the final result of the operations unless you enable immediate_blend, in other words operation_2 sees a full unblended result from operation_1.",
        )
        .req_field_blend_scale_mode(
            (
                "none",
                "reverse_sampling",
                "sampling",
                "reverse_enabled_range",
                "enabled_range",
                "sampling_sin",
                "enabled_range_sin",
            ),
            default="reverse_sampling",
            tooltip="Can be used to scale the blend strength over time. Basically works like blend_strength * scale_factor (see below)\nnone: Just uses the blend_strength you have set.\nreverse_sampling: The opposite of the model sampling percent, so if you're making a new generation, the beginning of sampling will be 1.0 and the end will be 0.0. The recommended option as applying these operations usually works better toward the beginning of sampling.\nsampling: Same as reverse_sampling, except the beginning will be 0.0 and the end will be 1.0.\nreverse_enabled_range: Flipped percentage of the range between start_sigma and end_sigma.\nenabled_range: Percentage of the range between start_sigma and end_sigma.\nsampling_sin: Uses the sampling percentage with the sine function such that blend_strength will hit the peak value in the middle of the range.\nenabled_range_sin: Similar to sampling_sin except it applies to the percentage of the enabled range.",
        )
        .req_float_blend_scale_offset(
            default=0.0,
            min=-1.0,
            max=1.0,
            tooltip="Only applies when blend_scale_mode is not none. Adds the offset to the calculated percentage and then clamps it to be between blend_scale_min and blend_scale_max.",
        )
        .req_float_blend_scale_min(
            default=0.0,
            tooltip="Only applies when blend_scale_mode is not none. Minimum value for the blend scale percentage. Many blend modes don't tolerate negative values here.",
        )
        .req_float_blend_scale_max(
            default=1.0,
            tooltip="Only applies when blend_scale_mode is not none. Maximum value for the blend scale percentage. Many blend modes don't tolerate values over 1.0 here.",
        )
        .req_bool_immediate_blend(
            tooltip="You can enable this to do blending immediately after each latent operation is called. Mainly affects the case where you have multiple latent operations connected.",
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
        immediate_blend: bool,
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
            SonarLatentOperation(op=o)
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
            cond_scale = args.get("cond_scale")
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
                curr_result = operation(
                    result,
                    sigma=sigma,
                    t2=t2,
                    cond=cond,
                    uncond=uncond,
                    cond_scale=cond_scale,
                    raw_args=args,
                )
                result = (
                    blend_function(result, curr_result, curr_blend)
                    if immediate_blend
                    else curr_result
                )
            if t2 is not None:
                result += t2
            if pred_flip_mode:
                result = x - sigma_t * result
            if not immediate_blend:
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
        qnorm_filter = functools.partial(
            utils.quantile_normalize,
            quantile=quantile,
            dim=None if dim == "global" else int(dim),
            flatten=flatten,
            nq_fac=norm_factor,
            pow_fac=norm_power,
            strategy=strategy,
        )

        return (SonarLatentOperation(op=lambda latent: qnorm_filter(latent)),)  # noqa: PLW0108


class SonarLatentOperationAdvancedNode(metaclass=IntegratedNode):
    DESCRIPTION = "Allows scheduling and other advanced features for latent operations. If you attach the optional extra LATENT_OPERATIONS, they will be called in sequence _before_ blending or output scaling."
    RETURN_TYPES = ("LATENT_OPERATION",)
    CATEGORY = "latent/advanced/operations"

    FUNCTION = "go"

    INPUT_TYPES = SonarLazyInputTypes(
        lambda: SonarInputTypes()
        .req_field_operation(
            "LATENT_OPERATION",
            tooltip="Latent operation to apply.",
        )
        .req_float_start_sigma(
            default=-1.0,
            min=-1.0,
            tooltip="First sigma the effect becomes active. You can set a negative value here to use whatever the model's maximum sigma is.",
        )
        .req_float_end_sigma(
            default=0.0,
            min=0.0,
            tooltip="Last sigma the effect is active.",
        )
        .req_float_input_multiplier(
            default=1.0,
            tooltip="Flat multiplier on the input to the latent operation. The multiplied input is *not* used when calculating the difference, it is only passed to the operation.",
        )
        .req_float_output_multiplier(
            default=1.0,
            tooltip="Flat multiplier on the output from the latent operation. Occurs before blending or calculating the difference.",
        )
        .req_float_difference_multiplier(
            default=1.0,
            tooltip="Flat multiplier on the difference or change from the original that the operation performed. Occurs after output_multiplier and before blending applies.",
        )
        .req_selectblend_blend_mode(
            default="inject",
            tooltip="Controls how the change from the operation is combined with the input. The default of inject just adds it scaled by the blend strength. With 1.0 blend strength, this is just using the output from the operation with no change.",
        )
        .req_float_blend_strength(
            default=0.5,
            tooltip="Strength of the blend.",
        )
        .opt_field_operation_alt(
            "LATENT_OPERATION",
            tooltip="Optional alternative operation that will be used when the primary one isn't enabled. May be useful in a case when you want one operation between sigma 1.0 and 0.5 and then a difference operation for lower sigmas which is kind of annoying to specify manually (you'd need to do something like configure another operation to start at 0.499999 or something).",
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
    def go(
        cls,
        *,
        operation,
        start_sigma: float,
        end_sigma: float,
        input_multiplier: float,
        output_multiplier: float,
        difference_multiplier: float,
        blend_mode: str,
        blend_strength: float,
        operation_alt=None,
        operation_2=None,
        operation_3=None,
        operation_4=None,
        operation_5=None,
    ) -> tuple[SonarLatentOperationAdvanced]:
        operations = tuple(
            o if isinstance(o, SonarLatentOperation) else SonarLatentOperation(op=o)
            for o in (operation, operation_2, operation_3, operation_4, operation_5)
            if o is not None
        )
        if operation_alt is not None and not isinstance(
            operation_alt,
            SonarLatentOperation,
        ):
            operation_alt = SonarLatentOperation(op=operation_alt)
        return (
            SonarLatentOperationAdvanced(
                ops=operations,
                op_alt=operation_alt,
                start_sigma=start_sigma,
                end_sigma=end_sigma,
                input_multiplier=input_multiplier,
                output_multiplier=output_multiplier,
                difference_multiplier=difference_multiplier,
                blend_mode=blend_mode,
                blend_strength=blend_strength,
            ),
        )


class SonarLatentOperationNoiseNode(metaclass=IntegratedNode):
    DESCRIPTION = "Latent operation that allows injecting noise."
    RETURN_TYPES = ("LATENT_OPERATION",)
    CATEGORY = "latent/advanced/operations"

    FUNCTION = "go"

    INPUT_TYPES = SonarLazyInputTypes(
        lambda: SonarInputTypes()
        .req_customnoise_custom_noise()
        .req_bool_scale_to_sigma(tooltip="Scales the noise to the current sigma.")
        .req_bool_cpu_noise(
            tooltip="Controls whether noise is generated on the CPU or GPU. GPU is usually faster but may change seeds for different models of GPU.",
        )
        .req_bool_normalize(
            default=True,
            tooltip="Controls whether the generated noise is normalized.",
        )
        .req_bool_lazy_noise_sampler(
            default=True,
            tooltip="When enabled, the latent operation will attempt to cache the noise sampler between calls and only recreate it when necessary. However, there isn't a 100% reliable way for a latent operation to know when sampling starts/ends so if we get it wrong this will lead to non-deterministic generations. I believe the heuristic I'm using to detect this should be reliable but you can disable it if you notice weird results.",
        ),
    )

    @classmethod
    def go(
        cls,
        *,
        custom_noise,
        scale_to_sigma: bool,
        cpu_noise: bool,
        normalize: bool,
        lazy_noise_sampler: bool,
    ) -> tuple[SonarLatentOperationNoise]:
        return (
            SonarLatentOperationNoise(
                custom_noise=custom_noise,
                scale_to_sigma=scale_to_sigma,
                cpu_noise=cpu_noise,
                normalize=normalize,
                lazy_noise_sampler=lazy_noise_sampler,
            ),
        )


class SonarLatentOperationSetSeedNode(metaclass=IntegratedNode):
    DESCRIPTION = "Latent operation that allows setting a seed. Can be useful for running latent operations that generate noise outside of a normal sampling context (i.e. operations on the initial latent before sampling)."
    RETURN_TYPES = ("LATENT_OPERATION",)
    CATEGORY = "latent/advanced/operations"

    FUNCTION = "go"

    INPUT_TYPES = SonarLazyInputTypes(
        lambda: SonarInputTypes()
        .req_field_operation("LATENT_OPERATION")
        .req_seed(
            tooltip="Seed to set. Note that this is called _every time_ before the operation.",
        )
        .req_bool_restore_rng_state(
            default=False,
            tooltip="When enabled, the current RNG state is saved just before calling the operation and restored afterwards. In other words, only the latent operation will see the seed you set. Note: This only handles the PyTorch and Python random module states.",
        ),
    )

    @classmethod
    def go(
        cls,
        *,
        operation,
        seed: int,
        restore_rng_state: bool,
    ) -> tuple[SonarLatentOperationSetSeed]:
        return (
            SonarLatentOperationSetSeed(
                op=operation,
                seed=seed,
                restore_rng_state=restore_rng_state,
            ),
        )


NODE_CLASS_MAPPINGS = {
    "SonarApplyLatentOperationCFG": SonarApplyLatentOperationCFG,
    "SonarLatentOperationQuantileFilter": SonarLatentOperationQuantileFilter,
    "SonarLatentOperationAdvanced": SonarLatentOperationAdvancedNode,
    "SonarLatentOperationNoise": SonarLatentOperationNoiseNode,
    "SonarLatentOperationSetSeed": SonarLatentOperationSetSeedNode,
}
