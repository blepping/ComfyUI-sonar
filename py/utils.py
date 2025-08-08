from __future__ import annotations

import math
import random
from functools import partial
from typing import TYPE_CHECKING, Callable

import torch
from comfy.model_management import device_supports_non_blocking, get_torch_device
from comfy.utils import common_upscale

from .external import MODULES as EXT

if TYPE_CHECKING:
    from collections.abc import Sequence

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
    if count < 2:
        candidates = cresult
    else:
        multiplier = 1.0 / count
        cresult = cresult * multiplier  # noqa: PLR6104
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


quantile_handlers = {
    "clamp": lambda noise, nq, **_kwargs: noise.clamp(-nq, nq),
    "scale_down": _quantile_norm_scaledown,
    "tanh": lambda noise, nq, **_kwargs: noise.tanh().mul_(nq.abs()),
    "tanh_outliers": lambda noise, nq, **_kwargs: torch.where(
        noise.abs() > nq,
        noise.tanh().mul_(nq.abs()),
        noise,
    ),
    "sigmoid_keepsign": lambda noise, nq, **_kwargs: noise.sigmoid()
    .mul_(nq.abs())
    .copysign(noise),
    "sigmoid": lambda noise, nq, **_kwargs: noise.sigmoid()
    .mul_(nq.abs() * 2)
    .sub_(nq.abs()),
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
}


# Initial version based on Studentt distribution normalizatino from https://github.com/Clybius/ComfyUI-Extra-Samplers/
def quantile_normalize(
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
