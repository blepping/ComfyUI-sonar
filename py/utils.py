from __future__ import annotations

import math
import random
from enum import Enum, auto
from functools import lru_cache, partial
from typing import TYPE_CHECKING, NamedTuple

import torch
from comfy.model_management import device_supports_non_blocking, get_torch_device
from comfy.utils import common_upscale

from .external import MODULES as EXT

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

F = torch.nn.functional

BLENDING_MODES = {
    "lerp": torch.lerp,
    "inject": lambda a, b, t: (b * t).add_(a),
    "subtract_b": lambda a, b, t: a - b * t,
    "weighted_average": lambda a, b, t: (b * t).add_(a) / (1.0 + abs(t)),
}
UPSCALE_METHODS = (
    "bilinear",
    "nearest-exact",
    "nearest",
    "area",
    "bicubic",
    "bislerp",
    "adaptive_avg_pool2d",
)


def blend_scalar(
    a: float,
    b: float,
    t: float,
    *,
    blend_function: Callable | None = None,
    clamp_function: Callable | None = None,
) -> float:
    if blend_function is None:
        return maybe_apply(
            a * (1.0 - t) + b * t,
            clamp_function is not None,
            clamp_function,
        )
    return maybe_apply(
        blend_function(
            *(torch.tensor((v,), device="cpu", dtype=torch.float64) for v in (a, b, t)),
        )
        .cpu()
        .item(),
        clamp_function is not None,
        clamp_function,
    )


def scale_samples(
    samples: torch.Tensor,
    width: int,
    height: int,
    *,
    mode: str = "bicubic",
) -> torch.Tensor:
    if mode == "adaptive_avg_pool2d":
        return torch.nn.functional.adaptive_avg_pool2d(samples, (height, width))
    return common_upscale(samples, width, height, mode, None)


def init_integrations(integrations) -> None:
    global scale_samples, BLENDING_MODES, UPSCALE_METHODS  # noqa: PLW0603

    bleh = integrations.bleh
    if bleh is None:
        return
    bleh_latentutils = bleh.py.latent_utils
    BLENDING_MODES = bleh_latentutils.BLENDING_MODES
    UPSCALE_METHODS = bleh_latentutils.UPSCALE_METHODS
    scale_samples = bleh_latentutils.scale_samples


EXT.register_init_handler(init_integrations)


def scale_noise(
    noise: torch.Tensor,
    factor: float = 1.0,
    *,
    normalized: bool = True,
    threshold_std_devs: float = 2.5,
    normalize_dims: tuple | None = None,
) -> torch.Tensor:
    numel = noise.numel()
    if not normalized or numel == 0:
        return noise.mul_(factor) if factor != 1 else noise
    if normalize_dims is not None:
        std = noise.std(dim=normalize_dims, keepdim=True)
        noise = (noise / std).nan_to_num_()
        return noise.sub_(noise.mean(dim=normalize_dims, keepdim=True)).mul_(factor)
    mean, std = noise.mean().item(), noise.std().item()
    threshold = threshold_std_devs / math.sqrt(numel)
    if abs(mean) > threshold:
        noise -= mean
    if abs(1.0 - std) > threshold:
        noise /= std
    return noise.mul_(factor) if factor != 1 else noise


CAN_NONBLOCK = {}


def tensor_to(
    tensor: torch.Tensor,
    dest: torch.Tensor | torch.Device | str,
) -> torch.Tensor:
    device = dest.device if isinstance(dest, torch.Tensor) else dest
    non_blocking = CAN_NONBLOCK.get(device)
    if non_blocking is None:
        non_blocking = device_supports_non_blocking(device)
        CAN_NONBLOCK[device] = non_blocking
    return tensor.to(dest, non_blocking=non_blocking)


def range_wrap(
    x: torch.Tensor,
    min_val: float | torch.Tensor,
    max_val: float | torch.Tensor,
) -> torch.Tensor:
    return min_val + (x - min_val).remainder_(max_val - min_val)


def softplus_soft_clamp(
    t: torch.Tensor,
    min_val: torch.Tensor | float = 0.0,
    max_val: torch.Tensor | float = 1.0,
    *,
    # We define stiffness as a multiplier (beta) for the softplus function.
    # Higher stiffness = sharper transition.
    stiffness: float = 1.0,
    safe: bool = True,
) -> torch.Tensor:
    if isinstance(min_val, (float, int)):
        min_val = t.new_tensor(min_val)
    if isinstance(max_val, (float, int)):
        max_val = t.new_tensor(max_val)

    if stiffness < 1e-04:
        return t.clamp(min_val, max_val)

    # Calculate how much we are exceeding the Max
    # softplus(beta * x) / beta
    upper_overshoot = F.softplus((t - max_val).mul_(stiffness)).div_(-stiffness)

    # Calculate how much we are falling short of the Min
    lower_undershoot = F.softplus((min_val - t).mul_(stiffness)).div_(stiffness)

    # Apply corrections:
    # Original - (Amount over max) + (Amount under min)
    t = upper_overshoot.add_(t).add_(lower_undershoot)
    if safe:
        t = t.clamp(min_val, max_val)
    return t


def _quantile_norm_scaledown(
    noise: torch.Tensor,
    nq: torch.Tensor,
    *,
    dim,
    **_kwargs: dict,
) -> torch.Tensor:
    noiseabs = noise.abs()
    mv = noiseabs.max(dim=dim, keepdim=True).values.clamp(min=1e-06)
    return (
        noise
        if mv.sum().item() == 0
        else torch.where(noiseabs > nq, noise * (nq / mv), noise)
    )


def _quantile_norm_wave(
    noise: torch.Tensor,
    nq: torch.Tensor,
    *,
    preserve_sign: bool = False,
    wave_function=torch.sin,
    pi_factor: float = 0.5,
    wrong_mode: bool = False,
    **_kwargs: dict,
) -> torch.Tensor:
    if wrong_mode:
        multiplier = 1.0 / ((math.pi * pi_factor) / nq)
    else:
        multiplier = 1.0 / (nq / (math.pi * pi_factor))
    pos_mask = noise >= 0
    neg_mask = ~pos_mask
    result = torch.zeros_like(noise)
    result[pos_mask] = wave_function(noise.mul(multiplier))[pos_mask]
    result[neg_mask] = wave_function(noise.mul(multiplier))[neg_mask]
    result *= nq
    return result.copysign(noise) if preserve_sign else result


def _quantile_norm_mode(
    noise: torch.Tensor,
    nq: torch.Tensor,
    *,
    dim: int | None,
    decimals=1,
    **_kwargs: dict,
) -> torch.Tensor:
    return torch.where(
        noise.abs() > nq,
        noise.round(decimals=decimals).mode(dim=dim, keepdim=True).values,
        noise,
    )


def _quantile_norm_replace(
    noise: torch.Tensor,
    nq: torch.Tensor,
    *,
    keep_sign: bool = False,
    avoid_sign: bool = False,
    count: int = 1,
    count_flipping: bool = False,
    **_kwargs: dict,
) -> torch.Tensor:
    mask = noise.abs() <= nq
    candidates = noise[mask].flatten()
    n_candidates = candidates.numel()
    idxs = torch.arange(noise.numel()) % n_candidates
    cresult = candidates[idxs]
    if count > 1:
        multiplier = 1.0 / count
        cresult = cresult * multiplier
        for i in range(1, count):
            cresult += (
                candidates[
                    torch.roll(
                        idxs,
                        i if not count_flipping or (i % 2) == 0 else -i,
                        dims=(-1,),
                    )
                ]
                * multiplier
            )
    candidates = cresult.reshape(noise.shape)
    if keep_sign or avoid_sign:
        candidates = candidates.copysign_(noise.neg() if avoid_sign else noise)
    return torch.where(mask, noise, candidates)


def _quantile_norm_redistribute(
    noise: torch.Tensor,
    nq: torch.Tensor,
    *,
    headroom_multiplier: float = 1.0,
    headroom_multiplier_neg: float | None = None,
    eps: float | None = None,
) -> torch.Tensor:
    if eps is None:
        eps = torch.finfo(noise.dtype).eps * 1.25
    nq_neg = -nq

    t_clamped = torch.clamp(noise, nq_neg, nq)

    t_signs = noise.signbit()
    excess = noise - t_clamped
    headroom = torch.where(
        t_signs,
        nq_neg - t_clamped,
        nq - t_clamped,
    ).masked_fill_(excess != 0, 0.0)

    total_headroom_pos = (
        headroom.clamp_min(0)
        .sum(dim=-1, keepdim=True)
        .mul_(headroom_multiplier)
        .clamp_min_(eps)
    )
    total_headroom_neg = (
        headroom.clamp_max(0)
        .sum(dim=-1, keepdim=True)
        .mul_(
            headroom_multiplier_neg
            if headroom_multiplier_neg is not None
            else headroom_multiplier,
        )
        .clamp_max_(-eps)
    )
    factor_pos = (
        excess.clamp_min(0.0).sum(dim=-1, keepdim=True).div_(total_headroom_pos)
    )
    factor_neg = (
        excess.clamp_max_(0.0).sum(dim=-1, keepdim=True).div_(total_headroom_neg)
    )
    t_clamped += torch.where(t_signs, headroom * factor_neg, headroom * factor_pos)

    return t_clamped


quantile_handlers = {
    "clamp": lambda noise, nq, **_kwargs: noise.clamp(-nq, nq),
    "scale_down": _quantile_norm_scaledown,
    "tanh": lambda noise, nq, **_kwargs: noise.tanh().mul_(nq.abs()),
    "tanh_outliers": lambda noise, nq, **_kwargs: torch.where(
        noise.abs() > nq,
        noise.tanh().mul_(nq.abs()),
        noise,
    ),
    "sigmoid_keepsign": lambda noise, nq, **_kwargs: (
        noise.sigmoid().mul_(nq.abs()).copysign(noise)
    ),
    "sigmoid": lambda noise, nq, **_kwargs: (
        noise.sigmoid().mul_(nq.abs() * 2).sub_(nq.abs())
    ),
    "sigmoid_outliers": lambda noise, nq, **_kwargs: torch.where(
        noise.abs() > nq,
        noise.sigmoid().mul_(nq.abs()).copysign(noise),
        noise,
    ),
    "sin": partial(_quantile_norm_wave, wave_function=torch.sin),
    "sin_wholepi": partial(
        _quantile_norm_wave,
        wave_function=torch.sin,
        pi_factor=1.0,
    ),
    "sin_keepsign": partial(
        _quantile_norm_wave,
        wave_function=torch.sin,
        preserve_sign=True,
    ),
    "sin_wrong": partial(_quantile_norm_wave, wave_function=torch.sin, wrong_mode=True),
    "sin_wrong_wholepi": partial(
        _quantile_norm_wave,
        wave_function=torch.sin,
        pi_factor=1.0,
        wrong_mode=True,
    ),
    "sin_wrong_keepsign": partial(
        _quantile_norm_wave,
        wave_function=torch.sin,
        preserve_sign=True,
        wrong_mode=True,
    ),
    "cos": partial(_quantile_norm_wave, wave_function=torch.cos),
    "cos_wholepi": partial(
        _quantile_norm_wave,
        wave_function=torch.cos,
        pi_factor=1.0,
    ),
    "cos_keepsign": partial(
        _quantile_norm_wave,
        wave_function=torch.cos,
        preserve_sign=True,
    ),
    "cos_wrong": partial(_quantile_norm_wave, wave_function=torch.cos, wrong_mode=True),
    "cos_wrong_wholepi": partial(
        _quantile_norm_wave,
        wave_function=torch.cos,
        pi_factor=1.0,
        wrong_mode=True,
    ),
    "cos_wrong_keepsign": partial(
        _quantile_norm_wave,
        wave_function=torch.cos,
        preserve_sign=True,
        wrong_mode=True,
    ),
    "atan": lambda noise, nq, **_kwargs: noise.atan().mul_(nq.abs() / (math.pi / 2)),
    "tenth": lambda noise, nq, **_kwargs: torch.where(
        noise.abs() > nq,
        noise * 0.1,
        noise,
    ),
    "half": lambda noise, nq, **_kwargs: torch.where(
        noise.abs() > nq,
        noise * 0.5,
        noise,
    ),
    "zero": lambda noise, nq, **_kwargs: torch.where(noise.abs() > nq, 0, noise),
    "reverse_zero": lambda noise, nq, **_kwargs: torch.where(
        noise.abs() >= nq,
        noise,
        0,
    ),
    "mean": lambda noise, nq, *, dim, **_kwargs: torch.where(
        noise.abs() > nq,
        noise.mean(dim=dim, keepdim=True),
        noise,
    ),
    "median": lambda noise, nq, *, dim, **_kwargs: torch.where(
        noise.abs() > nq,
        noise.median(dim=dim, keepdim=True).values,
        noise,
    ),
    "mode_1dec": partial(_quantile_norm_mode, decimals=1),
    "mode_2dec": partial(_quantile_norm_mode, decimals=2),
    "replace": _quantile_norm_replace,
    "replace_keepsign": partial(_quantile_norm_replace, keep_sign=True),
    "replace_avoidsign": partial(_quantile_norm_replace, avoid_sign=True),
    "replace_2pt": partial(_quantile_norm_replace, count=2),
    "replace_3pt": partial(_quantile_norm_replace, count=3),
    "replace_2pt_flip": partial(_quantile_norm_replace, count=2, count_flipping=True),
    "replace_3pt_flip": partial(_quantile_norm_replace, count=3, count_flipping=True),
    "replace_2pt_keepsign": partial(
        _quantile_norm_replace,
        count=2,
        keep_sign=True,
    ),
    "replace_3pt_keepsign": partial(
        _quantile_norm_replace,
        count=3,
        keep_sign=True,
    ),
    "replace_2pt_flip_keepsign": partial(
        _quantile_norm_replace,
        count=2,
        count_flipping=True,
        keep_sign=True,
    ),
    "replace_3pt_flip_keepsign": partial(
        _quantile_norm_replace,
        count=3,
        count_flipping=True,
        keep_sign=True,
    ),
    "replace_2pt_avoidsign": partial(
        _quantile_norm_replace,
        count=2,
        avoid_sign=True,
    ),
    "replace_3pt_avoidsign": partial(
        _quantile_norm_replace,
        count=3,
        avoid_sign=True,
    ),
    "replace_2pt_flip_avoidsign": partial(
        _quantile_norm_replace,
        count=2,
        count_flipping=True,
        avoid_sign=True,
    ),
    "replace_3pt_flip_avoidsign": partial(
        _quantile_norm_replace,
        count=3,
        count_flipping=True,
        avoid_sign=True,
    ),
    "wrap": lambda noise, nq, **_kwargs: range_wrap(noise, -nq, nq),
    "wrap_keepsign": lambda noise, nq, **_kwargs: torch.where(
        noise.abs() > nq,
        range_wrap(noise, -nq, nq).copysign_(noise),
        noise,
    ),
    "wrap_avoidsign": lambda noise, nq, **_kwargs: torch.where(
        noise.abs() > nq,
        range_wrap(noise, -nq, nq).copysign_(noise.neg()),
        noise,
    ),
    "softplus_clamp_s01": lambda noise, nq, **_kwargs: softplus_soft_clamp(
        noise,
        -nq,
        nq,
        stiffness=0.1,
    ),
    "softplus_clamp_s05": lambda noise, nq, **_kwargs: softplus_soft_clamp(
        noise,
        -nq,
        nq,
        stiffness=0.5,
    ),
    "softplus_clamp_s1": lambda noise, nq, **_kwargs: softplus_soft_clamp(
        noise,
        -nq,
        nq,
        stiffness=1.0,
    ),
    "softplus_clamp_s2": lambda noise, nq, **_kwargs: softplus_soft_clamp(
        noise,
        -nq,
        nq,
        stiffness=1.0,
    ),
    "softplus_clamp_s5": lambda noise, nq, **_kwargs: softplus_soft_clamp(
        noise,
        -nq,
        nq,
        stiffness=5.0,
    ),
    "redistribute": lambda noise, nq, **_kwargs: _quantile_norm_redistribute(noise, nq),
    "redistribute_hr075": lambda noise, nq, **_kwargs: _quantile_norm_redistribute(
        noise,
        nq,
        headroom_multiplier=0.75,
    ),
    "redistribute_hr150": lambda noise, nq, **_kwargs: _quantile_norm_redistribute(
        noise,
        nq,
        headroom_multiplier=1.5,
    ),
}


def flip_tensor_range(
    x: torch.Tensor,
    *,
    min_neg: torch.Tensor | None = None,
    max_pos: torch.Tensor | None = None,
    return_ranges: bool = False,
    dim: int = -1,
    eps: float | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if eps is None:
        eps = torch.finfo(x.dtype).eps * 1.25
    # 1. Use the provided maximum positive values, or calculate them dynamically
    if max_pos is None:
        max_pos = (
            torch.clamp_min(x, 0.0).max(dim=dim, keepdim=True).values.clamp_min_(eps)
        )

    # 2. Use the provided minimum negative values, or calculate them dynamically
    if min_neg is None:
        min_neg = (
            torch.clamp_max(x, 0.0).min(dim=dim, keepdim=True).values.clamp_max_(-eps)
        )

    # 3. Separate positive and negative elements
    is_pos = x >= 0

    # 4. Flip positive side: [0, max_pos] -> [eps, max_pos + eps]
    x_pos = x.clamp_min(eps)
    flipped_pos = (max_pos + eps) - x_pos

    # 5. Flip negative side: [min_neg, 0] -> [min_neg - eps, -eps]
    x_neg = x.clamp_max(-eps)
    flipped_neg = (min_neg - eps) - x_neg

    # 6. Recombine the domains
    result = torch.where(is_pos, flipped_pos, flipped_neg)
    return (result, max_pos, min_neg) if return_ranges else result


# Initial version based on StudentT distribution normalization from https://github.com/Clybius/ComfyUI-Extra-Samplers/
def quantile_normalize(
    noise: torch.Tensor,
    *,
    noise_reference: torch.Tensor | None = None,
    quantile: float | tuple | list = 0.75,
    dim: int | None = 1,
    flatten: bool = True,
    nq_fac: float = 1.0,
    pow_fac: float = 0.5,
    pow_fac_in: float = 0.0,
    strategy: str = "clamp",
    strategy_handler=None,
    # None, keep, avoid
    sign_mode: str | None = None,
    only_outliers: bool = False,
    abs_quantiles: bool = True,
    nq_lo: float | None = None,
    nq_hi: float | None = None,
    zero_mean_scale: float = 0.0,
    restore_mean_scale: float | None = None,
    zero_result_mean_scale: float | None = None,
    eps: float | None = None,
) -> torch.Tensor:
    if zero_result_mean_scale is None:
        zero_result_mean_scale = zero_mean_scale
    if restore_mean_scale is None:
        restore_mean_scale = zero_mean_scale
    if noise.numel() == 0:
        return noise
    if dim is not None and dim < 0:
        dim = noise.ndim + dim
    eff_dim = dim if dim is not None else 0
    if eps is None:
        eps = torch.finfo(noise.dtype).eps * 1.25
    while len(stratparts := strategy.rsplit("_", 1)) == 2 and stratparts[-1] in {
        "keepsign",
        "avoidsign",
        "outliers",
    }:
        strategy, stratadjust = stratparts
        if stratadjust == "outliers":
            only_outliers = True
        elif stratadjust in {"keepsign", "avoidsign"}:
            sign_mode = "keep" if stratadjust == "keepsign" else "avoid"
    if nq_lo is None:
        if isinstance(quantile, (tuple, list)):
            for q in quantile:
                noise = quantile_normalize(
                    noise=noise,
                    noise_reference=noise_reference,
                    quantile=q,
                    dim=dim,
                    flatten=flatten,
                    nq_fac=nq_fac,
                    pow_fac=pow_fac,
                    strategy=strategy,
                    strategy_handler=strategy_handler,
                    sign_mode=sign_mode,
                    only_outliers=only_outliers,
                    abs_quantiles=abs_quantiles,
                    eps=eps,
                )
            return noise
        if quantile is None or quantile >= 1 or quantile <= -1 or quantile == 0:
            return noise
        inverted = quantile < 0
        nq_pos = nq_neg = None
    absquantile = abs(quantile)
    orig_shape = noise.shape
    if noise.ndim > 1 and flatten:
        flatnoise = noise.flatten(start_dim=eff_dim)
        eff_dim = -1
    else:
        flatten = False
        flatnoise = noise
    if nq_lo is not None:
        inverted = False
        nq_neg = noise.new_tensor(max(eps, abs(nq_lo))).reshape((1,) * flatnoise.ndim)
        nq_pos = (
            nq_neg
            if nq_hi is None
            else noise.new_tensor(max(eps, abs(nq_hi))).reshape(nq_neg.shape)
        )

    if zero_mean_scale or restore_mean_scale:
        saved_mean = flatnoise.mean(dim=eff_dim, keepdim=True)
        if zero_mean_scale:
            flatnoise = flatnoise - (
                saved_mean * zero_mean_scale if zero_mean_scale != 1 else saved_mean
            )
        if restore_mean_scale == 0:
            saved_mean = None
        elif restore_mean_scale != 1:
            saved_mean *= restore_mean_scale
    else:
        saved_mean = None
    orig_noise_flat = flatnoise
    if inverted:
        flatnoise, max_pos, min_neg = flip_tensor_range(
            flatnoise,
            dim=eff_dim,
            return_ranges=True,
        )

    if noise_reference is None:
        noise_reference = flatnoise
    elif noise_reference.numel() != flatnoise.numel():
        raise ValueError(
            "noise_reference must have the same number of elements as noise",
        )
    else:
        noise_reference = noise_reference.to(flatnoise).reshape(flatnoise.shape)
        if zero_mean_scale:
            noise_reference = noise_reference - noise_reference.mean(
                dim=eff_dim,
                keepdim=True,
            ).mul_(zero_mean_scale)
        if inverted:
            noise_reference: torch.Tensor = flip_tensor_range(
                noise_reference,
                dim=eff_dim,
                return_ranges=False,
            )
    if pow_fac_in not in {0, 1}:
        noise_reference = (
            noise_reference.abs().pow_(pow_fac_in).copysign_(noise_reference)
        )
    handler = (
        quantile_handlers.get(strategy)
        if strategy_handler is None
        else strategy_handler
    )
    if handler is None:
        raise ValueError("Unknown strategy")
    handler = partial(handler, orig_noise=noise, dim=dim, flatten=flatten)
    need_outliers = only_outliers or sign_mode is not None
    outliers_mask = None
    if abs_quantiles:
        if nq_pos is None:
            nq = torch.quantile(
                noise_reference.abs(),
                absquantile,
                dim=eff_dim,
                keepdim=True,
            )
            nq = nq.mul_(nq_fac).add_(eps)
        else:
            nq = nq_pos
        if need_outliers:
            outliers_mask = flatnoise < -nq
            outliers_mask |= flatnoise > nq
        noise = handler(
            flatnoise,
            nq,
            orig_noise=noise,
            dim=dim,
            flatten=flatten,
        )
    else:
        noise_signs = noise_reference.signbit()
        if nq_pos is None or nq_neg is None:
            nq_pos = (
                torch.nanquantile(
                    torch.where(noise_signs, torch.nan, noise_reference),
                    absquantile,
                    dim=eff_dim,
                    keepdim=True,
                )
                .mul_(nq_fac)
                .add_(eps)
            )
            nq_neg = (
                torch.nanquantile(
                    torch.where(noise_signs, noise_reference.abs(), torch.nan),
                    absquantile,
                    dim=eff_dim,
                    keepdim=True,
                )
                .mul_(nq_fac)
                .add_(eps)
            )
        noise = torch.where(
            noise_signs,
            handler(flatnoise.neg(), nq_neg).neg_(),
            handler(flatnoise, nq_pos),
        )
        if need_outliers:
            outliers_mask = flatnoise < -nq_neg
            outliers_mask |= flatnoise > nq_pos

    if pow_fac not in {0.0, 1.0}:
        noise = noise.abs().pow_(pow_fac).copysign(noise)
    if inverted:
        noise = flip_tensor_range(
            noise,
            dim=eff_dim,
            min_neg=min_neg,
            max_pos=max_pos,
        )
    if outliers_mask is not None:
        noutliers = noise[outliers_mask].clone() if sign_mode else None
        if sign_mode in {"keep", "avoid"}:
            noutliers = noutliers.copysign_(
                (orig_noise_flat if sign_mode == "keep" else orig_noise_flat.neg())[
                    outliers_mask
                ],
            )
        elif sign_mode == "flip":
            noutliers = noutliers.neg_()
        elif sign_mode in {"pos", "neg"}:
            noutliers = noutliers.abs_()
            if sign_mode == "neg":
                noutliers = noutliers.neg_()
        if noutliers is not None:
            noise[outliers_mask] = noutliers
        if only_outliers:
            inv_outliers_mask = ~outliers_mask
            noise[inv_outliers_mask] = orig_noise_flat[inv_outliers_mask]
    if zero_result_mean_scale:
        noise_mean = noise.mean(dim=eff_dim, keepdim=True).mul_(-zero_result_mean_scale)
        saved_mean = noise_mean if saved_mean is None else saved_mean.add_(noise_mean)
    if saved_mean is not None:
        noise += saved_mean
    return noise if noise.shape == orig_shape else noise.reshape(orig_shape)


def quantile_normalize_old(
    noise: torch.Tensor,
    *,
    noise_reference: torch.Tensor | None = None,
    quantile: float | tuple | list = 0.75,
    dim: int | None = 1,
    flatten: bool = True,
    nq_fac: float = 1.0,
    pow_fac: float = 0.5,
    pow_fac_in: float = 0.0,
    strategy: str = "clamp",
    strategy_handler=None,
    # None, keep, avoid
    sign_mode: str | None = None,
    only_outliers: bool = False,
    abs_quantiles: bool = True,
    nq_lo: float | None = None,
    nq_hi: float | None = None,
    eps=1e-08,
) -> torch.Tensor:
    if noise.numel() == 0:
        return noise
    while len(stratparts := strategy.rsplit("_", 1)) == 2 and stratparts[-1] in {
        "keepsign",
        "avoidsign",
        "outliers",
    }:
        strategy, stratadjust = stratparts
        if stratadjust == "outliers":
            only_outliers = True
        else:
            sign_mode = "keep" if stratadjust == "keepsign" else "avoid"
    if nq_lo is None:
        if isinstance(quantile, (tuple, list)):
            for q in quantile:
                noise = quantile_normalize(
                    noise=noise,
                    noise_reference=noise_reference,
                    quantile=q,
                    dim=dim,
                    flatten=flatten,
                    nq_fac=nq_fac,
                    pow_fac=pow_fac,
                    strategy=strategy,
                    strategy_handler=strategy_handler,
                    sign_mode=sign_mode,
                    only_outliers=only_outliers,
                    abs_quantiles=abs_quantiles,
                    eps=eps,
                )
            return noise
        if quantile is None or quantile >= 1 or quantile <= -1 or quantile == 0:
            return noise
        centered = quantile < 0
        absquantile = abs(quantile)
        nq_pos = nq_neg = None
    orig_shape = noise.shape
    if noise.ndim > 1 and flatten:
        flatnoise = noise.flatten(start_dim=dim or 0)
    else:
        flatten = False
        flatnoise = noise
    if nq_lo is not None:
        centered = False
        nq_neg = noise.new_tensor(max(eps, abs(nq_lo))).reshape((1,) * flatnoise.ndim)
        nq_pos = (
            nq_neg
            if nq_hi is None
            else noise.new_tensor(max(eps, abs(nq_hi))).reshape(nq_neg.shape)
        )
    orig_noise_flat = flatnoise
    if noise_reference is None:
        noise_reference = flatnoise
    elif noise_reference.numel() != flatnoise.numel():
        raise ValueError(
            "noise_reference must have the same number of elements as noise",
        )
    else:
        noise_reference = noise_reference.to(flatnoise).reshape(flatnoise.shape)
    if pow_fac_in not in {0, 1}:
        noise_reference = (
            noise_reference.abs().pow_(pow_fac_in).copysign_(noise_reference)
        )
    handler = (
        quantile_handlers.get(strategy)
        if strategy_handler is None
        else strategy_handler
    )
    if handler is None:
        raise ValueError("Unknown strategy")
    handler = partial(handler, orig_noise=noise, dim=dim, flatten=flatten)
    need_outliers = only_outliers or sign_mode is not None
    outliers_mask = None
    if not centered:
        if abs_quantiles:
            if nq_pos is None:
                nq = torch.quantile(
                    noise_reference.abs(),
                    quantile,
                    dim=-1 if flatten else dim,
                    keepdim=True,
                )
                nq = nq.mul_(nq_fac).add_(eps)
            else:
                nq = nq_pos
            if need_outliers:
                outliers_mask = (flatnoise < -nq) | (flatnoise > nq)
            noise = handler(
                flatnoise,
                nq,
                orig_noise=noise,
                dim=dim,
                flatten=flatten,
            )
        else:
            noise_signs = noise_reference.signbit()
            if nq_pos is None or nq_neg is None:
                nq_pos = (
                    torch.nanquantile(
                        torch.where(noise_signs, torch.nan, noise_reference),
                        quantile,
                        dim=-1 if flatten else dim,
                        keepdim=True,
                    )
                    .mul_(nq_fac)
                    .add_(eps)
                )
                nq_neg = (
                    torch.nanquantile(
                        torch.where(noise_signs, noise_reference.abs(), torch.nan),
                        quantile,
                        dim=-1 if flatten else dim,
                        keepdim=True,
                    )
                    .mul_(nq_fac)
                    .add_(eps)
                )
            noise = torch.where(
                noise_signs,
                handler(flatnoise.neg(), nq_neg).neg_(),
                handler(flatnoise, nq_pos),
            )
            if need_outliers:
                outliers_mask = (flatnoise < -nq_neg) | (flatnoise > nq_pos)

    else:
        absnoise = noise_reference.abs()
        maxabs = absnoise.amax(dim=-1 if flatten else dim, keepdim=True)
        proxy = noise_reference.sign().mul_(maxabs - absnoise)
        nq_proxy = torch.quantile(
            proxy.abs(),
            absquantile,
            dim=-1 if flatten else dim,
            keepdim=True,
        )
        nq_proxy = nq_proxy.mul_(nq_fac).add_(eps)
        if need_outliers:
            outliers_mask = proxy > nq_proxy
        # print(f"\nNQ proxy: {nq_proxy}")
        out_proxy = handler(
            proxy,
            nq_proxy,
            orig_noise=noise,
            dim=dim,
            flatten=flatten,
        )
        noise = out_proxy.sign().mul_(maxabs - out_proxy.abs())
    if pow_fac not in {0.0, 1.0}:
        noise = noise.abs().pow_(pow_fac).copysign(noise)
    if outliers_mask is not None:
        if sign_mode in {"keep", "avoid"}:
            noise[outliers_mask].copysign_(
                (orig_noise_flat if sign_mode == "keep" else orig_noise_flat.neg())[
                    outliers_mask
                ],
            )
        if only_outliers:
            inv_outliers_mask = ~outliers_mask
            noise[inv_outliers_mask] = orig_noise_flat[inv_outliers_mask]

    return noise if noise.shape == orig_shape else noise.reshape(orig_shape)


class QuantileNormQuantileMode(Enum):
    QUANTILE = auto()
    # User supplied value to use as the quantile value.
    USER_HIGH = auto()
    # Like setting a negative quantile.
    USER_LOW = auto()


class QuantileNormSignMode(Enum):
    DEFAULT = auto()
    KEEP = auto()
    AVOID = auto()


class QuantileNormTargetMode(Enum):
    BOTH = auto()
    POSITIVE = auto()
    NEGATIVE = auto()


class QuantileNorm(NamedTuple):
    # mode: QuantileNormMode = QuantileNormMode.SYMMETRIC
    quantile_mode: QuantileNormQuantileMode = QuantileNormQuantileMode.QUANTILE
    sign_mode: QuantileNormSignMode = QuantileNormSignMode.DEFAULT
    target_mode: QuantileNormTargetMode = QuantileNormTargetMode.BOTH
    strategy: str = "clamp"
    # Overrides strategy.
    strategy_handler: Callable | None = None
    start_end_dim: tuple[int, int] | None = (1, -1)
    dims: tuple[int, ...] = ()
    quantile: float | torch.Tensor = 0.75
    # If none, we use absmax when quantile is negative, otherwise
    # abs value at this quantile.
    low_extreme_quantile: float | None = 0.99
    nq_scale: float = 1.0
    power: float = 0.0
    use_abs_quantile: bool = True
    # When setting a target other than BOTH, apply the mask to the quantile calculation as well.
    use_quantile_mask: bool = True
    use_float64: bool = True
    fix_invalid: bool = True

    @staticmethod
    def fix_dim(dim: int, ndim: int) -> int | None:
        if dim < 0:
            dim = ndim + dim
        return dim if 0 <= dim < ndim else None

    @classmethod
    def fix_dims(cls, dims: tuple[int, ...], ndim: int) -> tuple[int, ...]:
        dims = tuple(d for d in (cls.fix_dim(d_, ndim) for d_ in dims) if d is not None)
        return tuple({dims})

    def get_dims(self, ndim: int) -> tuple[int, ...]:
        dims = self.fix_dims(self.dims)
        if self.start_end_dim is None:
            return dims
        start_end_dim = self.fix_dims(*self.start_end_dim, ndim)
        if len(start_end_dim) != 2:
            return dims
        sd, ed = start_end_dim
        if sd > ed:
            sd, ed = ed, sd
        return tuple({range(sd, ed + 1), *dims})

    @classmethod
    @lru_cache(maxsize=128)
    def get_perms(
        cls,
        # Must be deduped and sanitized.
        dims: tuple[int, ...],
        ndim: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        other_dims = tuple(d for d in range(ndim) if d not in dims)
        perms = (*other_dims, *dims)
        inv_perms_t = torch.nn.utils.rnn.invert_permutation(
            torch.tensor(perms, device="cpu"),
        )
        if inv_perms_t is None:
            errstr = f"torch.nn.utils.rnn.invert_permutation returned None for input {perms}!"
            raise RuntimeError(errstr)
        inv_perms = tuple(inv_perms_t.tolist())
        return (perms, inv_perms)

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        ndim = t.ndim
        dims = self.get_dims(ndim)
        handler = (
            quantile_handlers.get(self.strategy)
            if self.strategy_handler is None
            else self.strategy_handler
        )
        if handler is None:
            raise ValueError("No strategy handler")
        if not dims or self.quantile == 0 or t.numel() < 2:
            return t
        orig_dtype = t.dtype
        orig_t = t
        eff_dtype = torch.float64 if self.use_float64 else torch.float32
        dlen = len(dims)
        olen = ndim - dlen
        perms, inv_perms = self.get_perms(dims, ndim)
        t = t.to(dtype=eff_dtype)
        # Move the dims we're working with to the end.
        t = t.permute(perms)
        permuted_shape = t.shape
        # And flatten them.
        t = t.flatten(start_dim=olen)
        use_low_extreme = (
            isinstance(self.quantile, float) and self.quantile < 0
        ) or self.quantile_mode == QuantileNormQuantileMode.USER_LOW
        if use_low_extreme:
            raise RuntimeError("NYI")
        if self.target_mode == QuantileNormTargetMode.NEGATIVE:
            mask = t.sign() < 0
        elif self.target_mode == QuantileNormTargetMode.POSITIVE:
            mask = t.sign() > 0
        else:
            mask = None
        masked_quantile = self.use_quantile_mask and mask is not None
        if self.quantile_mode == QuantileNormQuantileMode.QUANTILE:
            nq_input = t.abs() if self.use_abs_quantile else t
            if masked_quantile:
                nq_input = nq_input[mask]
            if not torch.any(nq_input):
                return orig_t
            nq = torch.quantile(nq_input, dim=-1, keepdim=True)
        else:
            raise RuntimeError("NYI")
        if self.nq_scale != 1.0:
            nq *= self.nq_scale
        # ...
        if self.power not in {0.0, 1.0}:
            t = t.abs().pow_(self.power).copysign_(t)
        if self.fix_invalid:
            t = t.nan_to_num_()
        t = t.to(dtype=orig_dtype)
        # Back to the original shape.
        return t.reshape(permuted_shape).permute(inv_perms).contiguous()


def quantile_normalize_adv(
    noise: torch.Tensor,
    *,
    quantile: float | tuple | list = 0.75,
    dim: int | None = 1,
    flatten: bool = True,
    nq_fac: float = 1.0,
    pow_fac: float = 0.5,
    strategy: str = "clamp",
    strategy_handler=None,
    eps=1e-08,
) -> torch.Tensor:
    if noise.numel() == 0:
        return noise
    if isinstance(quantile, (tuple, list)):
        for q in quantile:
            noise = quantile_normalize(
                noise=noise,
                quantile=q,
                dim=dim,
                flatten=flatten,
                nq_fac=nq_fac,
                pow_fac=pow_fac,
                strategy=strategy,
                strategy_handler=strategy_handler,
            )
        return noise
    if quantile is None or quantile >= 1 or quantile <= -1:
        return noise
    centered = quantile < 0
    absquantile = abs(quantile)
    orig_shape = noise.shape
    if noise.ndim > 1 and flatten:
        flatnoise = noise.flatten(start_dim=dim)
    else:
        flatten = False
        flatnoise = noise
    handler = (
        quantile_handlers.get(strategy)
        if strategy_handler is None
        else strategy_handler
    )
    if handler is None:
        raise ValueError("Unknown strategy")
    if not centered:
        nq = torch.quantile(
            flatnoise.abs(),
            quantile,
            dim=-1 if flatten else dim,
            keepdim=True,
        )
        nq = nq.mul_(nq_fac).add_(eps)
        # print(f"\nNQ: {nq}")
        noise = handler(
            flatnoise,
            nq,
            orig_noise=noise,
            dim=dim,
            flatten=flatten,
        )
    else:
        absnoise = flatnoise.abs()
        maxabs = absnoise.amax(dim=-1 if flatten else dim, keepdim=True)
        proxy = flatnoise.sign().mul_(maxabs - absnoise)
        nq_proxy = torch.quantile(
            proxy.abs(),
            absquantile,
            dim=-1 if flatten else dim,
            keepdim=True,
        )
        nq_proxy = nq_proxy.mul_(nq_fac).add_(eps)
        # print(f"\nNQ proxy: {nq_proxy}")
        out_proxy = handler(
            proxy,
            nq_proxy,
            orig_noise=noise,
            dim=dim,
            flatten=flatten,
        )
        noise = out_proxy.sign().mul_(maxabs - out_proxy.abs())
    if pow_fac not in {0.0, 1.0}:
        noise = noise.abs().pow_(pow_fac).copysign(noise)
    return noise if noise.shape == orig_shape else noise.reshape(orig_shape)


def normalize_to_scale(
    latent: torch.Tensor,
    target_min: float,
    target_max: float,
    *,
    dim=(-3, -2, -1),
    eps: float = 1e-07,
) -> torch.Tensor:
    min_val, max_val = (
        latent.amin(dim=dim, keepdim=True),
        latent.amax(dim=dim, keepdim=True),
    )
    normalized = latent - min_val
    normalized /= (max_val - min_val).add_(eps)
    return (
        normalized.mul_(target_max - target_min)
        .add_(target_min)
        .clamp_(target_min, target_max)
    )


def normalize_to_scale_adv(
    t: torch.Tensor,
    *,
    min_pos: float,
    max_pos: float,
    min_neg: float,
    max_neg: float,
    dim=(-3, -2, -1),
) -> torch.Tensor:
    skip_pos = max_pos <= 0 or min_pos >= max_pos
    skip_neg = min_neg >= 0 or min_neg >= max_neg
    neg_idxs, pos_idxs = t < 0.0, t > 0.0
    result = torch.zeros_like(t)
    if skip_neg:
        result[neg_idxs] = t[neg_idxs]
    elif torch.any(neg_idxs):
        neg_values = t[neg_idxs]
        if max_neg >= 0:
            max_neg = neg_values.max().detach().cpu().item()
        result[neg_idxs] = normalize_to_scale(
            neg_values,
            target_min=min_neg,
            target_max=max_neg,
            dim=dim,
        )
    if skip_pos:
        result[pos_idxs] = t[pos_idxs]
    elif torch.any(pos_idxs):
        pos_values = t[pos_idxs]
        if min_pos < 0:
            min_pos = pos_values.min().detach().cpu().item()
        result[pos_idxs] = normalize_to_scale(
            pos_values,
            target_min=min_pos,
            target_max=max_pos,
            dim=dim,
        )
    return result


def adjust_slice(s: slice, size: int, offset: int) -> slice:
    if offset == 0:
        return s
    # Input slice must have positive start/stop and be in bounds for the object that will be sliced here.
    start = s.start if s.start is not None else 0
    stop = s.stop if s.stop is not None else size
    if offset < 0:
        adj = min(start, abs(offset))
        return slice(start - adj, stop - adj)
    adj = min(size - stop, offset)
    return slice(start + adj, stop + adj)


def crop_samples(
    tensor: torch.Tensor,
    width: int,
    height: int,
    *,
    mode="center",
    offset_width: int = 0,
    offset_height: int = 0,
):
    if tensor.ndim < 3:
        raise ValueError("Can only handle >= 3 dimensional tensors")
    th, tw = tensor.shape[-2:]
    if (tw, th) == (width, height):
        return tensor
    if tw < width or th < height:
        raise ValueError("Can't crop sample smaller than requested width or height")
    if mode == "center":
        hmode = wmode = "center"
    else:
        hmode, wmode, *splitextra = mode.split("_")
        if splitextra:
            raise ValueError("Bad composite mode")
    if hmode == "top":
        hslice = slice(0, height)
    elif hmode == "center":
        hoffs = (th - height) // 2
        hslice = slice(hoffs, hoffs + height)
    elif hmode == "bottom":
        hslice = slice(th - height, th)
    else:
        raise ValueError("Bad height mode in composite mode")
    if wmode == "left":
        wslice = slice(0, width)
    elif wmode == "center":
        woffs = (tw - width) // 2
        wslice = slice(woffs, woffs + width)
    elif wmode == "right":
        wslice = slice(tw - width, tw)
    else:
        raise ValueError("Bad width mode in composite mode")
    wslice = adjust_slice(wslice, tw, offset_width)
    hslice = adjust_slice(hslice, th, offset_height)
    return tensor[..., hslice, wslice]


def fallback(val, default=None):
    return val if val is not None else default


# Pattern break algorithm adapted from https://github.com/Extraltodeus/noise_latent_perlinpinpin
def pattern_break(
    noise: torch.Tensor,
    *,
    percentage: float = 0.5,
    detail_level=0.0,
    restore_scale=True,
    blend_function=torch.lerp,
    high_precision: bool = True,
):
    orig_dtype = noise.dtype
    if high_precision:
        noise = noise.to(
            dtype=torch.complex128 if noise.is_complex() else torch.float64,
        )
    if restore_scale:
        orig_min, orig_max = noise.min().item(), noise.max().item()
    noise_normed = normalize_to_scale(
        noise,
        -1.0,
        1.0,
        dim=tuple(range(1, noise.ndim)),
    )
    result = (
        noise_normed.abs_()
        .mul_(1000000)
        .remainder_(11)
        .div_(11 / 2)
        .sub_(1)
        .erfinv_()
        .mul_((1 + detail_level / 10) * (2**0.5) * 0.2)
        .clamp_(-1, 1)
    )

    if restore_scale:
        result = normalize_to_scale(result, orig_min, orig_max, dim=())
    return blend_function(noise, result, percentage).to(dtype=orig_dtype)


def elementwise_shuffle_by_dim(
    t: torch.Tensor,
    *,
    dim: int = -1,
    prob: float = 1.0,
    no_identity: bool = False,
    generator=None,
) -> torch.Tensor:
    orig_shape = t.shape
    device = t.device

    num_positions = math.prod(orig_shape[:dim] + orig_shape[dim + 1 :])
    num_elements = orig_shape[dim]

    tensor_2d = t.permute(
        *tuple(d for d in range(t.dim()) if d != dim),
        dim,
    ).reshape(-1, num_elements)

    rand_perms = (
        torch.arange(num_elements, device=device).expand(num_positions, -1).clone()
    )

    if prob < 1.0:
        mask = torch.rand(num_positions, device=device, generator=generator) < prob
    else:
        mask = torch.ones(num_positions, device=device, dtype=torch.bool)

    if no_identity:
        offsets = torch.randint(
            1,
            num_elements,
            (num_positions,),
            device=device,
            generator=generator,
        )
        rand_perms[mask] = (
            torch.arange(num_elements, device=device) + offsets[mask][:, None]
        ) % num_elements
    else:
        rand_perms[mask] = torch.rand(
            num_positions,
            num_elements,
            device=device,
            generator=generator,
        )[mask].argsort(dim=1)

    shuffled_2d = torch.gather(tensor_2d, 1, rand_perms)

    shuffled = shuffled_2d.reshape(
        *orig_shape[:dim],
        *orig_shape[dim + 1 :],
        orig_shape[dim],
    )
    return shuffled.permute(
        *tuple(d for d in range(t.dim() - 1) if d < dim),
        t.dim() - 1,
        *tuple(d for d in range(t.dim() - 1) if d >= dim),
    ).contiguous()


def trunc_decimals(x: torch.Tensor, decimals: int = 3) -> torch.Tensor:
    x_i = x.trunc()
    x_f = x - x_i
    scale = 10.0**decimals
    return x_i.add_(x_f.mul_(scale).trunc_().mul_(1.0 / scale))


def maybe_apply(val, cond, fun):
    return fun(val) if cond else val


def maybe_apply_kwargs(d: dict | None, cond, fun, *, default=None):
    return default if d is None or not cond else fun(**d)


def tensor_item(val: torch.Tensor | float, *, collapse_function=torch.max) -> float:
    if isinstance(val, torch.Tensor):
        return float(collapse_function(val).detach().cpu().item())
    return float(val)


# Does not handle out of order or duplicated sigmas.
def step_from_sigmas(
    sigma: float | torch.Tensor,
    sigmas: torch.Tensor,
    *,
    decimals: int | None = 4,
    output_decimals: int = 2,
) -> float | None:
    sigma = tensor_item(sigma)
    sigmas = sigmas.detach().cpu()
    if sigmas.ndim == 2:
        sigmas = sigmas.max(dim=0).values
    elif sigmas.ndim != 1:
        errstr = f"Unexpected number of dimensions in sigmas, should be 1 or 2 but got shape {sigmas.shape}"
        raise ValueError(errstr)
    sigmas = sigmas[:-1]
    if not len(sigmas) or torch.any(sigmas <= 0):
        return None
    if decimals is not None:
        sigmas = sigmas.round(decimals=decimals)
        sigma = round(sigma, decimals)
    sigma_min, sigma_max = sigmas.aminmax()
    if not sigma_min <= sigma <= sigma_max:
        return None
    max_idx = len(sigmas) - 1
    idx = int(tensor_item((sigmas - sigma).abs().argmin()))
    idx_sigma = tensor_item(sigmas[idx])
    if decimals is not None:
        idx_sigma = round(idx_sigma, decimals)
    if sigma == idx_sigma:
        return float(idx)
    # Between sigmas, but guaranteed to be in range here.
    idx_low, idx_high = (idx, idx - 1) if sigma > idx_sigma else (idx + 1, idx)
    if idx_low < 0 or idx_high < 0 or idx_low > max_idx or idx_high > max_idx:
        return None
    sigma_low, sigma_high = tensor_item(sigmas[idx_low]), tensor_item(sigmas[idx_high])
    step_diff = sigma_high - sigma_low
    if step_diff == 0:
        return float(idx)
    pct = 1.0 - ((sigma - sigma_low) / step_diff)
    return round(idx_high + pct, output_decimals)


def clamp_float(val: float, minval=0.0, maxval=1.0) -> float:
    return max(minval, min(val, maxval))


def filter_dict(d: dict, keep: set | Sequence, *, recursive: bool = False) -> dict:
    return {
        k: v if not (recursive and isinstance(v, dict)) else filter_dict(v, keep)
        for k, v in d.items()
        if k in keep
    }


class RNGStates:
    DEFAULT_GPU_TYPE = get_torch_device().type

    def __init__(
        self,
        device_types: set | str | Sequence | None = None,
        *,
        add_defaults: bool = True,
    ):
        if device_types is None:
            device_types = set()
        elif isinstance(device_types, str):
            device_types = {device_types}
        elif not isinstance(device_types, set):
            device_types = set(device_types)
        if add_defaults:
            device_types = device_types | {"python", "cpu", self.DEFAULT_GPU_TYPE}  # noqa: PLR6104
        self.rng_states = self.get_states(device_types)

    def update(self):
        self.rng_states = self.get_states(set(self.rng_states))

    @staticmethod
    def get_states(device_types: set) -> dict:
        return {
            k: torch.get_rng_state()
            if k == "cpu"
            else (
                random.getstate()
                if k == "python"
                else getattr(torch, k).get_rng_state()
            )
            for k in device_types
            if k in {"python", "cpu"} or hasattr(torch, k)
        }

    def set_states(self, *, update: bool = True, override_states: dict | None = None):
        states = self.rng_states if override_states is None else override_states
        new_states = {}
        for k, v in states.items():
            if isinstance(v, torch.Tensor):
                v = v.clone()  # noqa: PLW2901
            if k == "cpu":
                new_states[k] = v
                torch.set_rng_state(v)
                continue
            if k == "python":
                new_states[k] = v
                random.setstate(v)
                continue
            tm = getattr(torch, k, None)
            if tm is not None:
                new_states[k] = v
                tm.set_rng_state(v)
        if update:
            self.rng_states = new_states


def robust_normalize(
    t: torch.Tensor,
    *,
    start_dim: int = 1,
    eps: float = 1e-8,
) -> torch.Tensor:
    orig_shape = t.shape
    t_flat = t.flatten(start_dim=start_dim)

    # Calculate along the flattened spatial/channel dimensions
    median = t_flat.median(dim=-1, keepdim=True).values
    mad = (t_flat - median).abs_().median(dim=-1, keepdim=True).values

    robust_std = mad.mul_(1.4826).clamp_min_(eps)
    return (t_flat - median).div_(robust_std).reshape(orig_shape)


def force_gaussian_distribution(
    t: torch.Tensor,
    *,
    start_dim: int = 1,
    end_dim: int = -1,
    # Invert the argsorts, option for crazy people. Not recommended.
    invert1: bool = False,
    invert2: bool = False,
    eps: float = 1e-08,
) -> torch.Tensor:
    if start_dim < 0:
        start_dim = t.ndim + start_dim
    orig_shape = t.shape
    t_flat = t.flatten(start_dim=start_dim, end_dim=end_dim).movedim(start_dim, -1)

    # Get the rank of each element (0 to N-1)
    # Double argsort safely returns the rank of the original elements
    ranks = (
        t_flat.argsort(dim=-1, descending=invert1)
        .argsort(dim=-1, descending=invert2)
        .to(t)
    )

    # Map ranks to a uniform distribution (0.0 to 1.0 exclusive)
    # then to a Gaussian curve.
    factor = max(eps, t_flat.shape[-1] / 2)
    gaussian = ranks.div_(factor).add_(0.5 / factor - 1).erfinv_().mul_(2**0.5)

    return gaussian.movedim(-1, start_dim).reshape(orig_shape)


# Forces source to the distribution of reference.
def match_distribution(
    source: torch.Tensor,
    *,
    reference: torch.Tensor,
    start_dim: int = 1,
    end_dim: int = -1,
    # Invert the sorts, option for crazy people. Not recommended.
    invert1: bool = False,
    invert2: bool = False,
    invert3: bool = False,
) -> torch.Tensor:
    if source is reference:
        return source.clone()
    if start_dim < 0:
        start_dim = source.ndim + start_dim
    orig_shape = source.shape
    s_flat = source.flatten(
        start_dim=start_dim,
        end_dim=end_dim,
    ).movedim(start_dim, -1)
    r_flat = reference.flatten(
        start_dim=start_dim,
        end_dim=end_dim,
    ).movedim(start_dim, -1)

    r_sorted = r_flat.sort(dim=-1, descending=invert1).values
    s_ranks = s_flat.argsort(
        dim=-1,
        descending=invert2,
    ).argsort(dim=-1, descending=invert3)

    # 4. Give the source elements the values from the reference.
    return (
        r_sorted.gather(dim=-1, index=s_ranks)
        .movedim(-1, start_dim)
        .reshape(orig_shape)
    )


# Scales the source tensor to match the median and variance of the reference.
def robust_scale_match(
    source: torch.Tensor,
    *,
    reference: torch.Tensor | None = None,
    # Default MAD if the reference is not passed. Targets the Gaussian distribution.
    mad: float = 0.6745,
    start_dim: int = 1,
    end_dim: int = -1,
    eps: float = 1e-8,
) -> torch.Tensor:
    if start_dim < 0:
        start_dim = source.ndim + start_dim
    orig_shape = source.shape
    source = source.flatten(start_dim=start_dim, end_dim=end_dim).movedim(start_dim, -1)
    # Find the median and spread (MAD) of the source
    src_sub_median = source - source.median(dim=-1, keepdim=True).values
    s_mad = (
        src_sub_median.abs()
        .median(
            dim=-1,
            keepdim=True,
        )
        .values.clamp_min_(eps)
    )

    # If no reference, target a Standard Gaussian scale.
    # (A standard Gaussian has a median of 0 and a MAD of ~0.6745)
    if reference is None:
        mad = min(-eps, mad) if mad < 0 else max(eps, mad)
        return (
            src_sub_median.mul_(s_mad.reciprocal_().mul_(mad))
            .movedim(-1, start_dim)
            .reshape(orig_shape)
        )
    reference = reference.flatten(
        start_dim=start_dim,
        end_dim=end_dim,
    ).movedim(start_dim, -1)

    # Find the reference median and spread
    r_median = reference.median(dim=-1, keepdim=True).values
    r_mad = (reference - r_median).abs_().median(dim=-1, keepdim=True).values

    # Stretch the source to match the reference
    return (
        src_sub_median.mul_(r_mad.div_(s_mad))
        .add_(r_median)
        .movedim(-1, start_dim)
        .reshape(orig_shape)
    )


def safe_pow(
    t: torch.Tensor,
    power: torch.Tensor | float,
    *,
    use_abs: bool = True,
    restore_sign: bool = True,
    in_place: bool = False,
) -> torch.Tensor:
    if not use_abs:
        return t.pow_(power) if in_place else t**power
    t_abs = t.abs().pow_(power)
    return t_abs if not restore_sign else t_abs.copysign_(t)
