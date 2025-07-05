from __future__ import annotations

import math
from functools import partial

import torch
from comfy.model_management import device_supports_non_blocking
from comfy.utils import common_upscale

from .external import MODULES as EXT

BLENDING_MODES = {
    "lerp": torch.lerp,
    "inject": lambda a, b, t: (b * t).add_(a),
    "subtract_b": lambda a, b, t: a - b * t,
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
        noise = noise / std  # noqa: PLR6104
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


def _quantile_norm_scaledown(
    noise: torch.Tensor,
    nq: torch.Tensor,
    *,
    dim,
    **_kwargs: dict,
) -> torch.Tensor:
    noiseabs = noise.abs()
    mv = noiseabs.max(dim=dim, keepdim=True).clamp(min=1e-06)
    return noise if mv == 0 else torch.where(noiseabs > nq, noise * (nq / mv), noise)


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


quantile_handlers = {
    "clamp": lambda noise, nq, **_kwargs: noise.clamp(-nq, nq),
    "scale_down": _quantile_norm_scaledown,
    "tanh": lambda noise, nq, **_kwargs: noise.tanh().mul_(nq.abs()),
    "tanh_outliers": lambda noise, nq, **_kwargs: torch.where(
        noise.abs() > nq,
        noise.tanh().mul_(nq.abs()),
        noise,
    ),
    "sigmoid": lambda noise, nq, **_kwargs: noise.sigmoid()
    .mul_(nq.abs())
    .copysign(noise),
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
}


# Initial version based on Studentt distribution normalizatino from https://github.com/Clybius/ComfyUI-Extra-Samplers/
def quantile_normalize(
    noise: torch.Tensor,
    *,
    quantile: float = 0.75,
    dim: int | None = 1,
    flatten: bool = True,
    nq_fac: float = 1.0,
    pow_fac: float = 0.5,
    strategy: str = "clamp",
    strategy_handler=None,
    use_abs: bool = True,
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
                use_abs=use_abs,
            )
        return noise
    if quantile is None or quantile <= 0 or quantile >= 1:
        return noise
    if not use_abs:
        pos_mask = noise >= 0
        neg_mask = ~pos_mask
        result = torch.zeros_like(noise)
        result[pos_mask] = quantile_normalize(
            noise=noise[pos_mask],
            quantile=quantile,
            dim=dim,
            flatten=flatten,
            nq_fac=nq_fac,
            pow_fac=pow_fac,
            strategy=strategy,
            strategy_handler=strategy_handler,
            use_abs=True,
        )
        result[neg_mask] = quantile_normalize(
            noise=noise[neg_mask],
            quantile=quantile,
            dim=dim,
            flatten=flatten,
            nq_fac=nq_fac,
            pow_fac=pow_fac,
            strategy=strategy,
            strategy_handler=strategy_handler,
            use_abs=True,
        )
        return result
    orig_shape = noise.shape
    if noise.ndim > 1 and flatten:
        flatnoise = noise.flatten(start_dim=dim)
    else:
        flatten = False
        flatnoise = noise
    nq = torch.quantile(
        flatnoise.abs(),
        quantile,
        dim=-1 if flatten else dim,
        keepdim=True,
    )
    nq = nq.mul_(nq_fac)
    handler = (
        quantile_handlers.get(strategy)
        if strategy_handler is None
        else strategy_handler
    )
    if handler is None:
        raise ValueError("Unknown strategy")
    noise = handler(
        flatnoise,
        nq,
        orig_noise=noise,
        dim=dim,
        flatten=flatten,
    )
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
):
    orig_dtype = noise.dtype
    if restore_scale:
        orig_min, orig_max = noise.min().item(), noise.max().item()
    noise_normed = normalize_to_scale(noise.to(dtype=torch.float32), -1.0, 1.0, dim=())
    result = torch.remainder(torch.abs(noise_normed) * 1000000, 11) / 11
    result = (
        ((1 + detail_level / 10) * torch.erfinv(2 * result - 1) * (2**0.5))
        .mul_(0.2)
        .clamp_(-1, 1)
    )
    if restore_scale:
        result = normalize_to_scale(result, orig_min, orig_max, dim=())
    return blend_function(noise, result, percentage).to(dtype=orig_dtype)


def trunc_decimals(x: torch.Tensor, decimals: int = 3) -> torch.Tensor:
    x_i = x.trunc()
    x_f = x - x_i
    scale = 10.0**decimals
    return x_i.add_(x_f.mul_(scale).trunc_().mul_(1.0 / scale))


def maybe_apply(val, cond, fun):
    return fun(val) if cond else val
