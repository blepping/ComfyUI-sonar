from __future__ import annotations

import math
from enum import Enum, auto
from typing import TYPE_CHECKING, Callable, NamedTuple

import torch
from tqdm import tqdm

from . import utils
from .wavelet_functions import (
    Wavelet,
    expand_yh_scales,
    wavelet_blend,
    wavelet_scaling,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def pretty_non_default(obj: NamedTuple, *, defaults: object | None = None) -> str:
    result = ", ".join(
        f"{fn}={fv.pretty_non_default()}"
        if hasattr(fv, "pretty_non_default")
        else f"{fn}={fv!r}"
        for fn, fv in ((_fn, getattr(obj, _fn)) for _fn in obj._fields)
        if defaults is None or fv != getattr(defaults, fn)
    )
    return f"{obj.__class__.__name__}({result})"


class WCFGSchedule(Enum):
    LINEAR = auto()
    LOGARITHMIC = auto()
    LOG = LOGARITHMIC
    EXPONENTIAL = auto()
    EXP = EXPONENTIAL
    HALF_COSINE = auto()
    SINE = auto()
    SIN = SINE

    def interp(self, val: float) -> float:
        val = utils.clamp_float(val)
        if self == WCFGSchedule.LINEAR:
            return val
        if self == WCFGSchedule.LOGARITHMIC:
            result = 0.0 if val == 0 else math.log(val) + 1.0
        elif self == WCFGSchedule.EXPONENTIAL:
            result = math.exp(val) - 1.0
        elif self == WCFGSchedule.HALF_COSINE:
            result = 1.0 - ((1.0 + math.cos(val * math.pi)) / 2)
        elif self == WCFGSchedule.SINE:
            result = math.sin(val * math.pi)
        else:
            raise ValueError("Bad interpolation schedule!?")
        return utils.clamp_float(result)


class WCFGSchedMode(Enum):
    SAMPLING = auto()
    ENABLED_SAMPLING = auto()
    SIGMAS = auto()
    ENABLED_SIGMAS = auto()
    STEP = auto()
    ENABLED_STEPS = auto()

    # Aliases
    MODEL_SAMPLING = SAMPLING
    ENABLED_MODEL_SAMPLING = ENABLED_SAMPLING
    SIGMA_RANGE = SIGMAS
    ENABLED_SIGMA_RANGE = ENABLED_SIGMAS


class WCFGTarget(Enum):
    DENOISED = auto()
    NOISE = auto()
    NOISE_NORM = auto()


class WCFGPercentages(NamedTuple):
    sigma: float
    sigma_min: float
    sigma_max: float
    sigma_first: float | None
    sigma_last: float | None
    steps: int | None
    step: float | None
    step_first: int | None
    step_last: int | None
    pct_sampling: float
    pct_enabled_sampling: float
    pct_sigmas: float | None
    pct_enabled_sigmas: float | None
    pct_steps: float | None
    pct_enabled_steps: float | None

    def invert(self) -> WCFGPercentages:
        return self._replace(
            pct_sampling=1.0 - self.pct_sampling,
            pct_enabled_sampling=1.0 - self.pct_enabled_sampling,
            pct_sigmas=None if self.pct_sigmas is None else 1.0 - self.pct_sigmas,
            pct_enabled_sigmas=None
            if self.pct_enabled_sigmas is None
            else 1.0 - self.pct_enabled_sigmas,
            pct_steps=None if self.pct_steps is None else 1.0 - self.pct_steps,
            pct_enabled_steps=None
            if self.pct_enabled_steps is None
            else 1.0 - self.pct_enabled_steps,
        )

    def pct_from_schedmode(self, mode: WCFGSchedMode) -> float | None:
        if mode == WCFGSchedMode.MODEL_SAMPLING:
            return self.pct_sampling
        if mode == WCFGSchedMode.SIGMA_RANGE:
            return self.pct_sigmas
        if mode == WCFGSchedMode.ENABLED_MODEL_SAMPLING:
            return self.pct_enabled_sampling
        if mode == WCFGSchedMode.ENABLED_SIGMA_RANGE:
            return self.pct_enabled_sigmas
        if mode == WCFGSchedMode.STEP:
            if self.pct_steps is None:
                raise RuntimeError("Step percentage not available")
            return self.pct_steps
        raise ValueError("Unknown mode")

    @classmethod
    def build(
        cls,
        *,
        ms: object,
        start_sigma: float,
        end_sigma: float,
        sigma: float,
        sigmas: torch.Tensor | None,
        **_kwargs: dict,
    ) -> WCFGPercentages:
        if start_sigma < end_sigma:
            raise ValueError("start/end sigmas out of order")
        sigma_max = ms.sigma_max.detach().item()
        sigma_min = ms.sigma_min.detach().item()
        start_sigma = min(sigma_max, start_sigma)
        end_sigma = min(max(sigma_min, end_sigma), sigma_max)
        sigma = min(max(sigma, sigma_min), sigma_max)
        rstart = torch.tensor(start_sigma)
        rend = torch.tensor(end_sigma)
        pct_start = 1.0 - (ms.timestep(rstart) / 999).clamp(0, 1).detach().item()
        pct_end = 1.0 - (ms.timestep(rend) / 999).clamp(0, 1).detach().item()
        pct_curr = (
            1.0 - (ms.timestep(torch.tensor(sigma)) / 999).clamp(0, 1).detach().item()
        )
        pct_range_curr = (pct_curr - pct_start) / (pct_end - pct_start)

        if sigmas is not None:
            if sigmas.ndim == 2:
                sigmas = sigmas.max(dim=0).values
            elif sigmas.ndim != 1:
                raise ValueError("Unexpected number of dimensions for sample_sigmas")
            sigmas = sigmas.detach().cpu()
            sigma_first = sigmas[0].item()
            sigma_last = sigmas[-2].item()
            if sigma_first <= sigma_last:
                raise ValueError(
                    "Cannot handle non-descending sigmas (possibly Restart or unsampling)",
                )
            pct_sigmas = (sigma_first - sigma) / (sigma_first - sigma_last)
            start_sigma = min(start_sigma, sigma_first)
            end_sigma = max(end_sigma, sigma_last)
            sigma = min(max(sigma, sigma_last), sigma_first)
            if start_sigma == end_sigma:
                pct_enabled_sigmas = 1.0
            else:
                pct_enabled_sigmas = (start_sigma - sigma) / (start_sigma - end_sigma)
            steps = len(sigmas) - 1
            if steps > 1:
                step = utils.step_from_sigmas(sigma, sigmas)
                pct_steps = step / (steps - 1) if step is not None else None
                enabled_steps = torch.arange(len(sigmas), dtype=torch.int32)[
                    (sigmas <= start_sigma) & (sigmas >= end_sigma)
                ]
                if len(enabled_steps) > 1:
                    step_first = enabled_steps[0].item()
                    step_last = enabled_steps[-1].item()
                    pct_enabled_steps = (step - step_first) / (step_last - step_first)
            else:
                step = 0.0
                pct_steps = 1.0
                step_first = step_last = None
                pct_enabled_steps = None
        else:
            pct_enabled_sigmas = pct_sigmas = None
            step = steps = None
            pct_enabled_steps = pct_steps = None
            sigma_first = sigma_last = None
        return WCFGPercentages(
            pct_sampling=pct_curr,
            pct_enabled_sampling=pct_range_curr,
            pct_sigmas=pct_sigmas,
            pct_enabled_sigmas=pct_enabled_sigmas,
            pct_steps=pct_steps,
            pct_enabled_steps=pct_enabled_steps,
            sigma=sigma,
            sigma_first=sigma_first,
            sigma_last=sigma_last,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            steps=steps,
            step=step,
            step_first=step_first,
            step_last=step_last,
        )


class WCFGScales(NamedTuple):
    yl_scale: float = 1.0
    yh_scales: float | Sequence = 1.0

    def get_scales(
        self,
        *_args: list,
        verbose: bool = False,
        **_kwargs: dict,
    ) -> WCFGScales:
        if verbose:
            tqdm.write(f"WCFG:     {self.pretty_scales()}")
        return self

    def apply_scales(
        self,
        yl: torch.Tensor,
        yh: Sequence,
    ) -> tuple[torch.Tensor, Sequence]:
        return wavelet_scaling(yl, yh, yl_scale=self.yl_scale, yh_scales=self.yh_scales)

    def get_and_apply_scales(
        self,
        pcts: WCFGPercentages,
        yl: torch.Tensor,
        yh: Sequence,
        *,
        verbose: bool = False,
    ) -> tuple[torch.Tensor, Sequence]:
        return self.get_scales(pcts, yh, verbose=verbose).apply_scales(yl, yh)

    def pretty_yh_scales(self, *, target=None) -> str:
        if target is None:
            target = self.yh_scales
        if isinstance(target, float):
            return f"{target:.4f}"
        result = ", ".join(
            self.pretty_yh_scales(target=val)
            if isinstance(val, (list, tuple))
            else (val if isinstance(val, str) else f"{val:.4f}")
            for val in target
        )
        return f"({result})"

    def pretty_scales(self):
        return f"low={self.yl_scale:.4f}, high={self.pretty_yh_scales()}"


class WCFGScheduledScale(NamedTuple):
    schedule: WCFGSchedule = WCFGSchedule.LINEAR
    schedule_mode: WCFGSchedMode = WCFGSchedMode.ENABLED_MODEL_SAMPLING
    schedule_offset: float = 0.0
    schedule_offset_after: float = 0.0
    schedule_multiplier: float = 1.0
    schedule_multiplier_after: float = 1.0
    reverse_schedule: bool = False
    reverse_schedule_after: bool = False
    schedule_min: float = 0.0
    schedule_max: float = 1.0

    @classmethod
    def build(cls, **kwargs: dict) -> WCFGScheduledScale:
        schedule = kwargs.pop("schedule", DEFAULT_SCHEDULEDSCALE.schedule)
        if isinstance(schedule, str):
            schedule = getattr(WCFGSchedule, schedule.upper())
        schedule_mode = kwargs.pop(
            "schedule_mode",
            DEFAULT_SCHEDULEDSCALE.schedule_mode,
        )
        if isinstance(schedule_mode, str):
            schedule_mode = getattr(WCFGSchedMode, schedule_mode.upper())
        return WCFGScheduledScale(
            schedule=schedule,
            schedule_mode=schedule_mode,
            **utils.filter_dict(kwargs, cls._fields),
        )

    def get_b_scale(self, pcts: WCFGPercentages) -> float:
        if self.reverse_schedule:
            pcts = pcts.invert()
        pct = pcts.pct_from_schedmode(self.schedule_mode)
        if pct is None:
            raise RuntimeError("Couldn't get percentage")
        pct = utils.clamp_float(
            (
                self.schedule.interp(
                    utils.clamp_float(
                        (pct + self.schedule_offset) * self.schedule_multiplier,
                    ),
                )
                + self.schedule_offset_after
            )
            * self.schedule_multiplier_after,
            minval=utils.clamp_float(self.schedule_min),
            maxval=utils.clamp_float(self.schedule_max),
        )
        if self.reverse_schedule_after:
            pct = utils.clamp_float(1.0 - pct)
        return pct

    def pretty_non_default(self) -> str:
        return pretty_non_default(self, defaults=DEFAULT_SCHEDULEDSCALE)


DEFAULT_SCHEDULEDSCALE = WCFGScheduledScale()


class WCFGScalesRange(NamedTuple):
    scales_start: WCFGScales = WCFGScales()
    scales_end: WCFGScales | None = None
    scheduler: WCFGScheduledScale | None = None
    blend_mode: str = "lerp"

    @classmethod
    def build(cls, **kwargs: dict) -> WCFGScales | WCFGScalesRange:
        scales_start = kwargs.pop("scales_start", None)
        if scales_start is None:
            scales_start = {
                "yl_scale": kwargs.pop("yl_scale", 1.0),
                "yh_scales": kwargs.pop("yh_scales", 1.0),
            }
        scales_end = utils.filter_dict(kwargs.pop("scales_end", {}), WCFGScales._fields)
        if not scales_end or scales_end == scales_start:
            return WCFGScales(
                yl_scale=scales_start.get("yl_scale", 1.0),
                yh_scales=scales_start.get("yh_scales", 1.0),
            )
        blend_mode = kwargs.pop("blend_mode", "lerp")
        return WCFGScalesRange(
            scales_start=WCFGScales(**scales_start),
            scales_end=WCFGScales(**scales_end),
            scheduler=utils.maybe_apply_kwargs(
                kwargs,
                bool(scales_end),
                WCFGScheduledScale.build,
            ),
            blend_mode=blend_mode,
        )

    def get_scales(
        self,
        pcts: WCFGPercentages,
        yh: Sequence,
        *,
        verbose: bool = False,
    ) -> WCFGScales:
        if self.scales_end is None or self.scheduler is None:
            return self.scales_start.get_scales()
        pct = self.scheduler.get_b_scale(pcts)
        if verbose:
            tqdm.write(f"WCFG:   pct={pct:.4f}, percentages: {pcts}")
        start, end = self.scales_start, self.scales_end
        simple_blend = self.blend_mode == "lerp"
        if pct <= 0 and simple_blend:
            simple_result = start
        elif pct >= 1 and simple_blend:
            simple_result = end
        else:
            simple_result = None
        if simple_result is not None:
            if verbose:
                tqdm.write(
                    f"WCFG:     {simple_result.pretty_scales()}",
                )
            return simple_result
        start_yh_scales = expand_yh_scales(yh, yh_scales=start.yh_scales)
        end_yh_scales = expand_yh_scales(yh, yh_scales=end.yh_scales)
        blend_function = (
            None if self.blend_mode == "lerp" else utils.BLENDING_MODES[self.blend_mode]
        )
        yl_scale = utils.blend_scalar(
            start.yl_scale,
            end.yl_scale,
            pct,
            blend_function=blend_function,
        )
        yh_scales = tuple(
            tuple(
                utils.blend_scalar(os, oe, pct, blend_function=blend_function)
                for os, oe in zip(bs, be)
            )
            for bs, be in zip(start_yh_scales, end_yh_scales)
        )
        result = WCFGScales(yl_scale=yl_scale, yh_scales=yh_scales)
        if verbose:
            tqdm.write(
                f"WCFG:     {result.pretty_scales()}",
            )
        return result

    def apply_scales(
        self,
        yl: torch.Tensor,
        yh: Sequence,
    ) -> tuple[torch.Tensor, Sequence]:
        return self.scales_start.apply_scales(yl, yh)

    def get_and_apply_scales(
        self,
        pcts: WCFGPercentages,
        yl: torch.Tensor,
        yh: Sequence,
        *,
        verbose: bool = False,
    ) -> tuple[torch.Tensor, Sequence]:
        return self.get_scales(pcts, yh, verbose=verbose).apply_scales(yl, yh)

    def pretty_non_default(self) -> str:
        return pretty_non_default(self, defaults=DEFAULT_SCALESRANGE)


DEFAULT_SCALESRANGE = WCFGScalesRange()


class WCFGScheduledFloat(NamedTuple):
    value_start: float
    value_end: float | None = None
    scheduler: WCFGScheduledScale | None = None

    @classmethod
    def build(
        cls,
        val: float | dict,
        *,
        default_start: float | None = None,
        default_end: float | None = None,
        **_kwargs: dict,
    ) -> WCFGScheduledFloat:
        if isinstance(val, float):
            return WCFGScheduledFloat(value_start=val)
        if not isinstance(val, dict):
            raise TypeError("Bad type for scheduled float value")
        val = val.copy()
        value_start = val.pop("value_start", default_start)
        value_end = val.pop("value_end", default_end)
        if not isinstance(value_start, (float, int)):
            raise TypeError("Bad type for scheduled float start_value")
        if value_end is None:
            return WCFGScheduledFloat(value_start=val)
        if not isinstance(value_end, (float, int)):
            raise TypeError("Bad type for scheduled float end_value")
        return WCFGScheduledFloat(
            value_start=float(value_start),
            value_end=float(value_end),
            scheduler=WCFGScheduledScale.build(**val),
        )

    def get_value(self, pcts: WCFGPercentages) -> float:
        if self.value_end is None or self.scheduler is None:
            return self.value_start
        pct = self.scheduler.get_b_scale(pcts)
        return (1.0 - pct) * self.value_start + pct * self.value_end


class WCFGWaveletSettings(NamedTuple):
    wave: str = "db4"
    level: int = 5
    padding_mode: str = "symmetric"
    use_1d_dwt: bool = False
    use_dtcwt: bool = False
    biort: str = "near_sym_a"
    qshift: str = "qshift_a"
    inv_wave: str | None = None
    inv_padding_mode: str | None = None
    inv_biort: str | None = None
    inv_qshift: str | None = None

    @classmethod
    def build(cls, **kwargs: dict) -> WCFGWaveletSettings:
        return WCFGWaveletSettings(**utils.filter_dict(kwargs, cls._fields))

    def make_wavelet(self, **kwargs: dict) -> Wavelet:
        return Wavelet(
            wave=self.wave,
            level=self.level,
            mode=self.padding_mode,
            use_1d_dwt=self.use_1d_dwt,
            use_dtcwt=self.use_dtcwt,
            biort=self.biort,
            qshift=self.qshift,
            inv_wave=self.inv_wave,
            inv_mode=self.inv_padding_mode,
            inv_biort=self.inv_biort,
            inv_qshift=self.inv_qshift,
            **kwargs,
        )

    def pretty_non_default(self) -> str:
        return pretty_non_default(self, defaults=DEFAULT_WAVELETSETTINGS)


DEFAULT_WAVELETSETTINGS = WCFGWaveletSettings()


class WCFGRule(NamedTuple):
    start_sigma: float = math.inf
    end_sigma: float = 0.0
    verbose: bool = False
    blend_mode: str = "lerp"
    blend_strength: WCFGScheduledFloat = WCFGScheduledFloat(1.0)
    fallback_existing: bool = True
    target_mode: WCFGTarget = WCFGTarget.DENOISED
    diff: WCFGScalesRange | WCFGScales | None = None
    cond: WCFGScalesRange | WCFGScales | None = None
    uncond: WCFGScalesRange | WCFGScales | None = None
    final: WCFGScalesRange | WCFGScales | None = None
    wavelet: WCFGWaveletSettings = DEFAULT_WAVELETSETTINGS
    high_precision_mode: bool = True
    difference_blend_mode: str = "inject"
    difference_blend_strength: WCFGScheduledFloat = WCFGScheduledFloat(1.0)

    @classmethod
    def build(cls, **kwargs: dict) -> WCFGRule:
        target_mode = kwargs.pop("target_mode", DEFAULT_RULE.target_mode)
        if isinstance(target_mode, str):
            target_mode = getattr(WCFGTarget, target_mode.upper())
        difference = kwargs.pop("diff", None)
        if difference is None:
            difference = kwargs.pop("difference", None)
        if difference is not None:
            difference = WCFGScalesRange.build(**difference)
        cond = kwargs.pop("cond", None)
        if cond is not None:
            cond = WCFGScalesRange.build(**cond)
        uncond = kwargs.pop("uncond", None)
        if uncond is not None:
            uncond = WCFGScalesRange.build(**uncond)
        final = kwargs.pop("final", None)
        if final is not None:
            final = WCFGScalesRange.build(**final)
        blend_strength = kwargs.pop("blend_strength", 1.0)
        if not isinstance(blend_strength, (float, int, dict)):
            raise TypeError("Bad type for blend_strength, must be float or dict")
        difference_blend_strength = kwargs.pop("difference_blend_strength", 1.0)
        if not isinstance(difference_blend_strength, (float, int, dict)):
            raise TypeError(
                "Bad type for difference_blend_strength, must be float or dict",
            )
        return WCFGRule(
            target_mode=target_mode,
            diff=difference,
            cond=cond,
            uncond=uncond,
            final=final,
            blend_strength=WCFGScheduledFloat(blend_strength),
            difference_blend_strength=WCFGScheduledFloat(difference_blend_strength),
            wavelet=WCFGWaveletSettings.build(**kwargs),
            **utils.filter_dict(kwargs, cls._fields),
        )

    def make_wavelet(self, **kwargs: dict) -> Wavelet:
        return self.wavelet.make_wavelet(**kwargs)

    def get_and_apply_scales(
        self,
        name: str,
        pcts: WCFGPercentages,
        yl: torch.Tensor,
        yh: Sequence,
        *,
        verbose: bool = False,
    ) -> tuple[torch.Tensor, Sequence]:
        scales = getattr(self, name).get_scales(pcts, yh)
        if verbose and (scales.yl_scale != 1.0 or scales.yh_scales != 1.0):
            tqdm.write(
                f"WCFG:     scales({name:>6}): {scales.pretty_scales()}",
            )
        return scales.apply_scales(yl, yh)

    def pretty_non_default(self) -> str:
        return pretty_non_default(self, defaults=DEFAULT_RULE)


DEFAULT_RULE = WCFGRule()


class WCFGRules(NamedTuple):
    rules: Sequence = ()

    def __len__(self) -> int:
        return len(self.rules)

    def __getitem__(self, idx: int) -> WCFGRule:
        return self.rules[idx]

    def __bool__(self) -> bool:
        return bool(self.rules)

    def get_rule(self, sigma: float) -> WCFGRule | None:
        for rule in self.rules:
            if (
                rule.end_sigma
                <= sigma
                <= (math.inf if rule.start_sigma < 0 else rule.start_sigma)
            ):
                return rule
        return None

    @classmethod
    def build(cls, **params: dict) -> WCFGRules:
        params = params.copy()
        rules = params.pop("rules", ())
        rule_1 = WCFGRule.build(**params)
        other_rules = (WCFGRule.build(**rparams) for rparams in rules)
        return WCFGRules(rules=(rule_1, *other_rules))


class WCFGContext(NamedTuple):
    cond: torch.Tensor
    uncond: torch.Tensor
    x: torch.Tensor
    sigma: torch.Tensor
    wavelet: Wavelet
    dtype: torch.dtype
    op_kwargs: dict


class WaveletCFG:
    def __init__(
        self,
        *,
        existing_cfg: Callable | None,
        rules: WCFGRules,
        operation_cond: Callable | None = None,
        operation_uncond: Callable | None = None,
        operation_fallback_cfg: Callable | None = None,
        operation_wavelet_cfg: Callable | None = None,
        operation_result: Callable | None = None,
    ):
        self.wavelet_cache = {}
        self.rules = rules
        self.fallback_cfg_function = (
            existing_cfg
            if existing_cfg is not None and (not rules or rules[0].fallback_existing)
            else self.basic_cfg_function
        )
        self.operation_cond = operation_cond
        self.operation_uncond = operation_uncond
        self.operation_fallback_cfg = operation_fallback_cfg
        self.operation_wavelet_cfg = operation_wavelet_cfg
        self.operation_result = operation_result

    @staticmethod
    def basic_cfg_function(args: dict) -> torch.Tensor:
        x, scale = args["input"], args["cond_scale"]
        uncond, cond = args["uncond_denoised"], args["cond_denoised"]
        return x - (cond - uncond).mul_(scale).add_(uncond)

    @staticmethod
    def maybe_op(
        t: torch.Tensor,
        mop: Callable | None,
        **kwargs: dict,
    ) -> torch.Tensor:
        return (
            t
            if mop is None
            else mop(
                latent=t,
                **(kwargs if getattr(mop, "EXTENDED_LATENT_OPERATION", None) else {}),
            )
        )

    def get_context(self, *, rule: WCFGRule, args: dict) -> WCFGContext:
        sigma_orig = sigma = args["sigma"]
        rule_id = id(rule)
        x = args["input"]
        if x.ndim == 3 and not rule.wavelet.use_1d_dwt:
            raise RuntimeError("Enable use_1d_dwt mode for 3D latents.")
        if x.ndim < 3:
            raise RuntimeError(
                "Wavelet CFG can't handle latents with 2 or less dimensions.",
            )
        if sigma.ndim != x.ndim:
            sigma = sigma.reshape(x.shape[0], *((1,) * (x.ndim - sigma.ndim)))
        if rule.target_mode in {WCFGTarget.NOISE, WCFGTarget.NOISE_NORM}:
            cond, uncond = args["cond"], args["uncond"]
            if rule.target_mode == WCFGTarget.NOISE_NORM:
                cond = cond / sigma  # noqa: PLR6104
                uncond = uncond / sigma  # noqa: PLR6104
        elif rule.target_mode == WCFGTarget.DENOISED:
            cond, uncond = args["cond_denoised"], args["uncond_denoised"]
        else:
            raise ValueError("Bad target mode")
        op_kwargs = {
            "sigma": sigma_orig,
            "cond": cond,
            "uncond": uncond,
            "cond_scale": args["cond_scale"],
            "raw_args": args,
        }
        cond = self.maybe_op(cond, self.operation_cond, **op_kwargs)
        uncond = self.maybe_op(uncond, self.operation_uncond, **op_kwargs)
        eff_dtype = torch.float64 if rule.high_precision_mode else x.dtype
        wavelet = self.wavelet_cache.get(rule_id)
        if wavelet is None:
            wavelet = rule.make_wavelet()
            self.wavelet_cache[rule_id] = wavelet
        wavelet = wavelet.to(device=x.device, dtype=eff_dtype)
        if rule.wavelet.use_1d_dwt:
            cond = cond.flatten(start_dim=2)
            uncond = uncond.flatten(start_dim=2)
        elif x.ndim > 4:
            cond = cond.flatten(start_dim=1, end_dim=cond.ndim - 3)
            uncond = uncond.flatten(start_dim=1, end_dim=uncond.ndim - 3)
        return WCFGContext(
            cond=cond,
            uncond=uncond,
            x=x,
            sigma=sigma,
            wavelet=wavelet,
            dtype=eff_dtype,
            op_kwargs=op_kwargs,
        )

    def process_output(
        self,
        *,
        result: torch.Tensor,
        rule: WCFGRule,
        ctx: WCFGContext,
    ) -> torch.Tensor:
        x_shape = ctx.x.shape
        if rule.wavelet.use_1d_dwt:
            result = result[..., : ctx.cond.shape[2]].reshape(x_shape)
        elif ctx.x.ndim > 4:
            result = result[..., : x_shape[-2], : x_shape[-1]].reshape(x_shape)
        else:
            result = result[tuple(slice(None, sz) for sz in x_shape)]
        if rule.target_mode == WCFGTarget.DENOISED:
            result = ctx.x - result
        elif rule.target_mode == WCFGTarget.NOISE_NORM:
            result *= ctx.sigma
        return self.maybe_op(result, self.operation_wavelet_cfg, **ctx.op_kwargs)

    @classmethod
    def wavelet_cfg(
        cls,
        *,
        rule: WCFGRule,
        ctx: WCFGContext,
        pcts: WCFGPercentages,
    ) -> torch.Tensor:
        verbose = rule.verbose
        diff_blend_function = utils.BLENDING_MODES[rule.difference_blend_mode]
        condw = ctx.wavelet.forward(ctx.cond.to(dtype=ctx.dtype))
        uncondw = ctx.wavelet.forward(ctx.uncond.to(ctx.dtype))
        if rule.cond is not None:
            condw = rule.get_and_apply_scales("cond", pcts, *condw, verbose=verbose)
        if rule.uncond is not None:
            uncondw = rule.get_and_apply_scales(
                "uncond",
                pcts,
                *uncondw,
                verbose=verbose,
            )
        diffw = wavelet_blend(
            condw,
            uncondw,
            yl_factor=1.0,
            blend_function=lambda a, b, _t: a - b,
        )
        if rule.diff is not None:
            diffw = rule.get_and_apply_scales("diff", pcts, *diffw, verbose=verbose)
        resultw = wavelet_blend(
            uncondw,
            diffw,
            yl_factor=rule.difference_blend_strength.get_value(pcts),
            blend_function=diff_blend_function,
        )
        if rule.final is not None:
            resultw = rule.get_and_apply_scales(
                "final",
                pcts,
                *resultw,
                verbose=verbose,
            )
        return ctx.wavelet.inverse(*resultw).to(dtype=ctx.x.dtype)

    def __call__(self, args: dict) -> torch.Tensor:
        sigma = args["sigma"]
        sigma_f = sigma.max().item()
        rule = self.rules.get_rule(sigma_f)
        if rule is None:
            return self.fallback_cfg_function(args)
        if rule.verbose:
            tqdm.write(
                f"\nWCFG: Rule matched, sigma={sigma_f:.4f}, rule={rule.pretty_non_default()}",
            )
        blend_function = utils.BLENDING_MODES[rule.blend_mode]
        model = args["model"]
        pcts = WCFGPercentages.build(
            ms=model.model_sampling,
            start_sigma=rule.start_sigma,
            end_sigma=rule.end_sigma,
            sigma=sigma_f,
            sigmas=args.get("model_options", {})
            .get("transformer_options", {})
            .get("sample_sigmas"),
        )
        wcfg_blend = rule.blend_strength.get_value(pcts)
        if rule.blend_mode == "lerp" and wcfg_blend == 0:
            return self.maybe_op(
                self.fallback_cfg_function(args),
                self.operation_fallback_cfg,
                sigma=sigma,
                cond=args["cond_denoised"],
                uncond=args["uncond_denoised"],
                raw_args=args,
            )
        ctx = self.get_context(rule=rule, args=args)
        result = self.wavelet_cfg(rule=rule, ctx=ctx, pcts=pcts)
        if rule.blend_mode != "lerp" or wcfg_blend != 1.0:
            normal_result = self.maybe_op(
                self.fallback_cfg_function(args),
                self.operation_fallback_cfg,
                **ctx.op_kwargs,
            )
            if rule.target_mode == WCFGTarget.DENOISED:
                normal_result = ctx.x - normal_result
            elif rule.target_mode == WCFGTarget.NOISE_NORM:
                normal_result /= ctx.sigma
            result = blend_function(normal_result, result, wcfg_blend)
        result = self.process_output(result=result, ctx=ctx, rule=rule)
        return self.maybe_op(
            result,
            self.operation_result,
            **ctx.op_kwargs,
        ).contiguous()
