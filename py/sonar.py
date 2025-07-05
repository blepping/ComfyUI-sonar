# Sonar sampler part adapted from https://github.com/alexblattner/modified-euler-samplers-for-sonar-diffusers and https://github.com/Kahsolt/stable-diffusion-webui-sonar

from __future__ import annotations

import importlib
from enum import Enum, auto
from functools import lru_cache
from sys import stderr
from typing import Any, Callable, NamedTuple

import torch
from comfy.k_diffusion.sampling import get_ancestral_step, to_d
from comfy.samplers import KSampler, k_diffusion_sampling
from torch import Tensor
from tqdm.auto import trange

from . import noise, utils


class HistoryType(Enum):
    ZERO = auto()
    RAND = auto()
    SAMPLE = auto()
    SAMPLE_NORM = auto()


class GuidanceType(Enum):
    LINEAR = auto()
    EULER = auto()


class GuidanceConfig(NamedTuple):
    guidance_type: GuidanceType = GuidanceType.LINEAR
    factor: float = 0.01
    start_step: int = 1
    end_step: int = 9999
    latent: Tensor | None = None


class MomentumMode(Enum):
    CLASSIC = auto()
    NEW = auto()
    DENOISED = auto()


class SonarConfig(NamedTuple):
    momentum: float = 0.95
    momentum_hist: float = 0.75
    direction: float = 1.0
    momentum_start_step: int = 0
    momentum_end_step: int = 9999
    always_update_history: bool = True
    momentum_mode: MomentumMode = MomentumMode.NEW
    init: HistoryType = HistoryType.ZERO
    noise_type: noise.NoiseType | None = None
    custom_noise: noise.CustomNoise | None = None
    rand_init_noise_type: noise.NoiseType | None = None
    rand_init_noise_multiplier: float | int = 1.0
    guidance: GuidanceConfig | None = None
    blend_mode: str = "lerp"
    momentum_blend_mode: str | None = None
    history_blend_mode: str | None = None
    guidance_blend_mode: str | None = None

    def get_with_default(self, k: str, default: Any) -> Any:  # noqa: ANN401
        val = getattr(self, k)
        return val if val is not None else default


class SonarBase:
    DEFAULT_NOISE_TYPE = noise.NoiseType.GAUSSIAN

    def __init__(self, cfg: SonarConfig) -> None:
        self.history_d = None
        self.cfg = cfg
        self.noise_sampler = None
        blend_mode = cfg.blend_mode
        momentum_blend_mode = cfg.get_with_default("momentum_blend_mode", blend_mode)
        history_blend_mode = cfg.get_with_default("history_blend_mode", blend_mode)
        guidance_blend_mode = cfg.get_with_default("guidance_blend_mode", blend_mode)
        bf = self.blend = utils.BLENDING_MODES[blend_mode]
        self.momentum_blend = (
            bf
            if momentum_blend_mode == blend_mode
            else utils.BLENDING_MODES[momentum_blend_mode]
        )
        self.history_blend = (
            bf
            if history_blend_mode == blend_mode
            else utils.BLENDING_MODES[history_blend_mode]
        )
        self.guidance_blend = (
            bf
            if guidance_blend_mode == blend_mode
            else utils.BLENDING_MODES[guidance_blend_mode]
        )

    _cfg_fixups = (
        ("momentum_mode", MomentumMode),
        ("init", HistoryType),
        ("noise_type", noise.NoiseType),
    )

    @classmethod
    def get_config(
        cls,
        cfg: SonarConfig | None = None,
        ext: dict | None = None,
    ) -> SonarConfig:
        cfgdict = ext.copy() if ext is not None else {}
        empty = object()
        for k, enum_class in cls._cfg_fixups:
            val = cfgdict.get(k, empty)
            if val is empty:
                continue
            if isinstance(val, str):
                val = getattr(enum_class, val.strip().upper(), empty)
                if val is empty:
                    validstr = ", ".join(enum_class.__members__.keys())
                    errstr = f"Bad value for {k} of type enum {enum_class.__name__}, must be one of the following: {validstr}"
                    raise ValueError(errstr)
                cfgdict[k] = val
                continue
            if not isinstance(val, enum_class):
                errstr = f"Bad parameter type for {k}: Must be valid string or instance of {enum_class.__name__}"
                raise TypeError(errstr)

        if cfg is None:
            return SonarConfig(**cfgdict)
        cfgdict = cfg._asdict() | cfgdict
        return SonarConfig(**cfgdict)

    def set_noise_sampler(
        self,
        x: Tensor,
        sigmas: Tensor,
        noise_sampler: Callable | None,
        seed: int | None = None,
    ) -> Callable:
        sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
        if noise_sampler is not None and self.cfg.noise_type not in {
            None,
            self.DEFAULT_NOISE_TYPE,
        }:
            print(
                "Sonar: Warning: Noise sampler supplied, overriding noise type from settings",
                file=stderr,
            )
        if self.cfg.custom_noise:
            noise_sampler = self.cfg.custom_noise.make_noise_sampler(
                x,
                sigma_min,
                sigma_max,
                seed=seed,
            )
        elif noise_sampler is None:
            noise_sampler = noise.get_noise_sampler(
                self.cfg.noise_type or self.DEFAULT_NOISE_TYPE,
                x,
                sigma_min,
                sigma_max,
                seed=seed,
                cpu=True,
                normalized=True,
            )
        self.noise_sampler = noise_sampler
        return noise_sampler

    def init_hist_d(
        self,
        x: Tensor,
        denoised: Tensor,
        sigma: Tensor,
        *,
        step: int,
    ) -> None:
        if self.history_d is not None or not self.check_step(step, is_history=True):
            return
        cfg = self.cfg
        init = cfg.init
        # memorize delta momentum
        if init == HistoryType.ZERO:
            self.history_d = None
        elif init == HistoryType.SAMPLE:
            self.history_d = (
                x if cfg.momentum_mode != MomentumMode.DENOISED else denoised
            )
        elif init == HistoryType.SAMPLE_NORM:
            self.history_d = (
                x if cfg.momentum_mode != MomentumMode.DENOISED else denoised
            ) / sigma
        elif init == HistoryType.RAND:
            ns = noise.get_noise_sampler(
                cfg.rand_init_noise_type,
                x,
                None,
                None,
                seed=self.extra_args.get("seed"),
                cpu=True,
                normalized=True,
            )
            self.history_d = ns(None, None)
            if cfg.rand_init_noise_multiplier != 1:
                self.history_d *= cfg.rand_init_noise_multiplier
        else:
            raise ValueError("Sonar sampler: bad history type")

    @property
    @lru_cache(maxsize=1)  # noqa: B019
    def history_ratios(self):
        direction = self.cfg.direction
        momentum_hist = self.cfg.momentum_hist
        return (
            momentum_hist,
            1.0 + abs(direction) * (1 - momentum_hist)
            if direction < 0
            else 2.0 - direction,
            direction,
        )

    def check_step(self, step: int, *, is_history: bool = False):
        cfg = self.cfg
        if is_history and cfg.always_update_history:
            return True
        return cfg.momentum_start_step <= step <= cfg.momentum_end_step

    def update_hist(self, momentum_d: torch.Tensor, step: int) -> None:
        hd, cfg = self.history_d, self.cfg
        if cfg.momentum_hist == 1 or not self.check_step(step, is_history=True):
            return
        hd_ratio, hd_scale, md_scale = self.history_ratios
        self.history_d = (
            momentum_d
            if hd is None
            else self.history_blend(momentum_d * md_scale, hd * hd_scale, hd_ratio)
        )

    def momentum_mix(
        self,
        history: Tensor | None,
        item: Tensor,
        sigma: Tensor,
        *,
        is_denoised: bool = False,
        momentum=None,
    ) -> Tensor:
        momentum = self.cfg.momentum if momentum is None else momentum
        mode = self.cfg.momentum_mode
        if (
            momentum == 1  # noqa: PLR0916
            or history is None
            or (mode == MomentumMode.DENOISED and not is_denoised)
            or (mode != MomentumMode.DENOISED and is_denoised)
        ):
            return item
        return self.momentum_blend(
            history * sigma if is_denoised else history,
            item,
            momentum,
        )

    def get_momentum_denoised(
        self,
        x: Tensor,
        denoised: Tensor,
        sigma: Tensor,
        *,
        step: int,
        momentum: float | None = None,
        update_history=True,
    ) -> Tensor:
        hd = self.history_d
        momentum_denoised = self.momentum_mix(
            hd,
            denoised,
            sigma,
            is_denoised=True,
            momentum=momentum,
        )
        if update_history:
            self.init_hist_d(x, denoised, sigma, step=step)
            self.update_hist(denoised / sigma, step=step)
        return momentum_denoised if self.check_step(step) else denoised

    def get_momentum_d(
        self,
        x: Tensor,
        denoised: Tensor,
        sigma: Tensor,
        *,
        step: int,
        momentum: float | None = None,
        d: Tensor | None = None,
        update_history=True,
    ) -> Tensor:
        hd = self.history_d
        cfg = self.cfg
        momentum = cfg.momentum if momentum is None else momentum
        mode = cfg.momentum_mode
        d = to_d(x, sigma, denoised) if d is None else d
        if momentum == 1 or mode == MomentumMode.DENOISED:
            return d
        momentum_d = self.momentum_mix(hd, d, sigma)
        if update_history:
            self.init_hist_d(x, denoised, sigma, step=step)
            self.update_hist(d if mode == MomentumMode.NEW else momentum_d, step=step)
        return momentum_d if self.check_step(step) else d

    def momentum_step(
        self,
        step: int,
        x: Tensor,
        denoised: Tensor,
        sigma: Tensor,
        sigma_down: Tensor,
    ) -> Tensor:
        dt = sigma_down - sigma
        denoised = self.get_momentum_denoised(x, denoised, sigma, step=step)
        momentum_d = self.get_momentum_d(x, denoised, sigma, step=step)
        return (momentum_d * dt).add_(x)


class SonarGuidanceMixin:
    def __init__(
        self,
        cfg: GuidanceConfig | None = None,
    ) -> None:
        self.guidance = cfg
        self.ref_latent = (
            self.prepare_ref_latent(cfg.latent)
            if cfg and cfg.latent is not None
            else None
        )

    @staticmethod
    def prepare_ref_latent(latent: Tensor | None) -> Tensor:
        if latent is None:
            return None
        avg_s = latent.mean(dim=(-2, -1), keepdim=True)
        std_s = latent.std(dim=(-2, -1), keepdim=True)
        return (latent - avg_s).div_(std_s).to(latent.dtype)

    def guidance_step(self, step_index: int, x: Tensor, denoised: Tensor) -> Tensor:
        if (
            self.guidance is None
            or self.guidance.factor == 0.0
            or not self.guidance.start_step <= step_index <= self.guidance.end_step
        ):
            return x
        if self.ref_latent.device != x.device:
            self.ref_latent = self.ref_latent.to(device=x.device)
        if self.guidance.guidance_type == GuidanceType.LINEAR:
            return self.guidance_linear(
                x,
                self.ref_latent,
                self.guidance.factor,
                blend=self.guidance_blend,
            )
        if self.guidance.guidance_type == GuidanceType.EULER:
            sigma, sigma_next = self.sigmas[step_index], self.sigmas[step_index + 1]
            return self.guidance_euler(
                sigma,
                sigma_next,
                x,
                denoised,
                self.ref_latent,
                self.guidance.factor,
            )
        raise ValueError("Sonar: Guidance: Unknown guidance type")

    @classmethod
    def guidance_shift(cls, t: Tensor, ref_latent: Tensor, *, dim=None):
        if dim is None:
            dim = tuple(range(-(t.ndim - 1), 0))
        avg_t = t.mean(dim=dim, keepdim=True)
        std_t = t.std(dim=dim, keepdim=True)
        return (ref_latent * std_t).add_(avg_t)

    @classmethod
    def guidance_euler(
        cls,
        sigma: Tensor,
        sigma_next: Tensor,
        x: Tensor,
        denoised: Tensor,
        ref_latent: Tensor,
        factor: float = 0.2,
        *,
        do_shift: bool = True,
    ) -> Tensor:
        if torch.equal(sigma, sigma_next):
            return cls.guidance_linear(x, ref_latent, factor=factor, do_shift=do_shift)
        ref_img_shift = (
            cls.guidance_shift(denoised, ref_latent) if do_shift else ref_latent
        )
        d = to_d(x, sigma, ref_img_shift)
        dt = (sigma_next - sigma) * factor
        return (d * dt).add_(x)

    @classmethod
    def guidance_linear(
        cls,
        x: Tensor,
        ref_latent: Tensor,
        factor: float = 0.2,
        *,
        blend=torch.lerp,
        do_shift: bool = True,
    ) -> Tensor:
        ref_img_shift = cls.guidance_shift(x, ref_latent) if do_shift else ref_latent
        return blend(x, ref_img_shift, factor)


class SonarWithGuidance(SonarBase, SonarGuidanceMixin):
    def __init__(self, *args: list[Any], **kwargs: dict[str, Any]):
        super().__init__(*args, **kwargs)
        SonarGuidanceMixin.__init__(self, self.cfg.guidance)


class SonarSampler(SonarWithGuidance):
    def __init__(
        self,
        model,
        sigmas: Tensor,
        s_in: Tensor,
        extra_args: dict[str, Any],
        *args: list[Any],
        **kwargs: dict[str, Any],
    ):
        super().__init__(*args, **kwargs)
        self.model = model
        self.sigmas = sigmas
        self.s_in = s_in
        self.extra_args = extra_args

    def call_model(
        self,
        x: Tensor,
        sigma: Tensor,
        *args: list[Any],
        s_in=None,
        extra_args=None,
    ) -> Tensor:
        if s_in is None:
            s_in = self.s_in
        extra_args = (
            self.extra_args if extra_args is None else self.extra_args | extra_args
        )
        return self.model(x, sigma * s_in, *args, **extra_args)


class SonarEuler(SonarSampler):
    def __init__(
        self,
        *args: list[Any],
        **kwargs: dict[str, Any],
    ):
        super().__init__(*args, **kwargs)

    def step(self, step_index: int, sample: torch.FloatTensor):
        sigma, sigma_next = self.sigmas[step_index], self.sigmas[step_index + 1]

        denoised = self.call_model(sample, sigma)
        result_sample = self.momentum_step(
            step_index,
            sample,
            denoised,
            sigma,
            sigma_next,
        )

        if sigma_next > 0:
            result_sample = self.guidance_step(step_index, result_sample, denoised)

        return (
            result_sample,
            sigma,
            sigma,
            denoised,
        )

    @classmethod
    def sampler(
        cls,
        model,
        x: Tensor,
        sigmas: Tensor,
        extra_args: dict | None = None,
        callback=None,
        disable: bool | None = None,  # noqa: FBT001
        noise_sampler: Callable | None = None,
        sonar_config: SonarConfig | None = None,
        sonar_params: dict | None = None,
    ) -> Tensor:
        sonar_config = cls.get_config(sonar_config, sonar_params)
        s_in = x.new_ones((x.shape[0],))
        sonar = cls(
            model,
            sigmas,
            s_in,
            {} if extra_args is None else extra_args,
            sonar_config,
        )
        sonar.set_noise_sampler(
            x,
            sigmas,
            noise_sampler,
            seed=extra_args.get("seed"),
        )

        for i in trange(len(sigmas) - 1, disable=disable):
            x, sigma, sigma_hat, denoised = sonar.step(
                i,
                x,
            )
            if callback is not None:
                callback(
                    {
                        "x": x,
                        "i": i,
                        "sigma": sigma,
                        "sigma_hat": sigma_hat,
                        "denoised": denoised,
                    },
                )
        return x


class SonarEulerAncestral(SonarSampler):
    def __init__(
        self,
        eta: float = 1.0,
        s_noise: float = 1.0,
        *args: list[Any],
        **kwargs: dict[str, Any],
    ):
        super().__init__(*args, **kwargs)
        self.eta = eta
        self.s_noise = s_noise

    def step(
        self,
        step_index: int,
        sample: torch.FloatTensor,
    ):
        sigma, sigma_next = self.sigmas[step_index], self.sigmas[step_index + 1]
        sigma_down, sigma_up = get_ancestral_step(
            sigma,
            sigma_next,
            eta=self.eta,
        )

        denoised = self.call_model(sample, sigma)
        result_sample = self.momentum_step(
            step_index,
            sample,
            denoised,
            sigma,
            sigma_down,
        )
        if sigma_next > 0:
            result_sample = self.guidance_step(step_index, result_sample, denoised)
            result_sample = (  # noqa: PLR6104
                result_sample
                + self.noise_sampler(sigma, sigma_next) * (self.s_noise * sigma_up)
            )

        return (
            result_sample,
            sigma,
            sigma,
            denoised,
        )

    @classmethod
    def sampler(
        cls,
        model,
        x,
        sigmas,
        extra_args=None,
        callback=None,
        disable=None,
        sonar_config: SonarConfig | None = None,
        sonar_params: dict | None = None,
        eta=1.0,
        s_noise=1.0,
        noise_sampler: Callable | None = None,
    ):
        sonar_config = cls.get_config(sonar_config, sonar_params)
        s_in = x.new_ones((x.shape[0],))
        sonar = cls(
            eta,
            s_noise,
            model,
            sigmas,
            s_in,
            {} if extra_args is None else extra_args,
            sonar_config,
        )
        sonar.set_noise_sampler(
            x,
            sigmas,
            noise_sampler,
            seed=extra_args.get("seed"),
        )

        for i in trange(len(sigmas) - 1, disable=disable):
            x, _sigma, sigma_hat, denoised = sonar.step(
                i,
                x,
            )
            if callback is not None:
                callback(
                    {
                        "x": x,
                        "i": i,
                        "sigma": sigmas[i],
                        "sigma_hat": sigma_hat,
                        "denoised": denoised,
                    },
                )
        return x


class SonarDPMPPSDE(SonarSampler):
    DEFAULT_NOISE_TYPE = noise.NoiseType.BROWNIAN

    def __init__(
        self,
        eta: float = 1.0,
        s_noise: float = 1.0,
        *args: list[Any],
        **kwargs: dict[str, Any],
    ):
        super().__init__(*args, **kwargs)
        self.eta = eta
        self.s_noise = s_noise

    @staticmethod
    def sigma_fn(t: Tensor) -> float:
        return t.neg().exp()

    @staticmethod
    def t_fn(sigma: Tensor) -> float:
        return sigma.log().neg()

    # DPM++ solver algorithm copied from ComfyUI source.
    def momentum_step(  # noqa: PLR0914
        self,
        step_index: int,
        x: Tensor,
        denoised: Tensor,
        sigma: Tensor,
        sigma_next: Tensor,
        sigma_down: Tensor,
    ) -> Tensor:
        if sigma_next == 0:
            return super().momentum_step(step_index, x, denoised, sigma, sigma_down)

        cfg = self.cfg
        # Halve the momentum proportion if there's history since we will use it twice.
        adjusted_momentum = (
            cfg.momentum + (1 - cfg.momentum) / 2
            if self.history_d is not None
            else cfg.momentum
        )

        r = 1 / 2
        # DPM-Solver++
        t, t_next = self.t_fn(sigma), self.t_fn(sigma_next)
        h = t_next - t
        s = t + h * r
        fac = 1 / (2 * r)

        # Step 1
        s_t, s_s = self.sigma_fn(t), self.sigma_fn(s)
        sd, su = get_ancestral_step(
            s_t,
            s_s,
            self.eta,
        )
        s_ = self.t_fn(sd)
        momentum_denoised = self.get_momentum_denoised(
            x,
            denoised,
            sigma,
            step=step_index,
        )
        diff_2 = (t - s_).expm1() * momentum_denoised
        momentum_d = self.get_momentum_d(
            x,
            momentum_denoised,
            sigma,
            step=step_index,
            momentum=adjusted_momentum,
            d=diff_2,
        )
        x_2 = ((self.sigma_fn(s_) / s_t) * x).sub_(momentum_d)
        x_2 += self.noise_sampler(s_t, s_s).mul_(
            self.s_noise * su,
        )
        sigma_2 = s_s
        denoised_2 = self.call_model(x_2, sigma_2)
        momentum_denoised_2 = self.get_momentum_denoised(
            x,
            denoised_2,
            sigma_2,
            step=step_index,
        )

        # Step 2
        s_t_next = self.sigma_fn(t_next)
        sd, su = get_ancestral_step(
            s_t,
            s_t_next,
            self.eta,
        )
        t_down = self.t_fn(sd)
        denoised_d = (1 - fac) * momentum_denoised + fac * momentum_denoised_2
        diff_1 = (t - t_down).expm1() * denoised_d
        momentum_d = self.get_momentum_d(
            x,
            momentum_denoised_2,
            sigma_2,
            step=step_index,
            momentum=adjusted_momentum,
            d=diff_1,
        )
        x = ((self.sigma_fn(t_down) / s_t) * x).sub_(momentum_d)
        x = self.guidance_step(step_index, x, denoised_d)
        x += self.noise_sampler(s_t, s_t_next).mul_(
            self.s_noise * su,
        )
        return x

    def step(
        self,
        step_index: int,
        sample: torch.FloatTensor,
    ) -> Tensor:
        def sigma_fn(t):
            return t.neg().exp()

        def t_fn(sigma):
            return sigma.log().neg()

        sigma, sigma_next = self.sigmas[step_index], self.sigmas[step_index + 1]
        sigma_down, _sigma_up = get_ancestral_step(
            sigma,
            sigma_next,
            eta=self.eta,
        )

        denoised = self.call_model(sample, sigma)
        result_sample = self.momentum_step(
            step_index,
            sample,
            denoised,
            sigma,
            sigma_next,
            sigma_down,
        )

        return (
            result_sample,
            sigma,
            sigma,
            denoised,
        )

    @classmethod
    def sampler(
        cls,
        model,
        x: Tensor,
        sigmas: Tensor,
        extra_args: dict | None = None,
        callback=None,
        disable: bool | None = None,  # noqa: FBT001
        sonar_config: SonarConfig | None = None,
        sonar_params: dict | None = None,
        eta=1.0,
        s_noise=1.0,
        noise_sampler=None,
    ) -> Tensor:
        sonar_config = cls.get_config(sonar_config, sonar_params)
        s_in = x.new_ones((x.shape[0],))
        sonar = cls(
            eta,
            s_noise,
            model,
            sigmas,
            s_in,
            {} if extra_args is None else extra_args,
            sonar_config,
        )
        sonar.set_noise_sampler(
            x,
            sigmas,
            noise_sampler,
            seed=extra_args.get("seed"),
        )

        for i in trange(len(sigmas) - 1, disable=disable):
            x, _sigma, sigma_hat, denoised = sonar.step(
                i,
                x,
            )
            if callback is not None:
                callback(
                    {
                        "x": x,
                        "i": i,
                        "sigma": sigmas[i],
                        "sigma_hat": sigma_hat,
                        "denoised": denoised,
                    },
                )
        return x


def add_samplers() -> None:
    extra_samplers = {
        "sonar_euler": SonarEuler.sampler,
        "sonar_euler_ancestral": SonarEulerAncestral.sampler,
        "sonar_dpmpp_sde": SonarDPMPPSDE.sampler,
    }
    added = 0
    for (
        name,
        sampler,
    ) in extra_samplers.items():
        if name in KSampler.SAMPLERS:
            continue
        try:
            KSampler.SAMPLERS.append(name)
            setattr(
                k_diffusion_sampling,
                f"sample_{name}",
                sampler,
            )
            added += 1
        except ValueError as exc:
            print(f"Sonar: Failed to add {name} to built in samplers list: {exc}")
    if added > 0:
        importlib.reload(k_diffusion_sampling)
