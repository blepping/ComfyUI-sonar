# Sonar sampler part adapted from https://github.com/alexblattner/modified-euler-samplers-for-sonar-diffusers and https://github.com/Kahsolt/stable-diffusion-webui-sonar

from __future__ import annotations

from enum import Enum, auto
from sys import stderr
from typing import Any, Callable, NamedTuple

import torch
from comfy.k_diffusion import sampling
from torch import Tensor
from tqdm.auto import trange

from . import noise


class HistoryType(Enum):
    ZERO = auto()
    RAND = auto()
    SAMPLE = auto()


class GuidanceType(Enum):
    LINEAR = auto()
    EULER = auto()


class GuidanceConfig(NamedTuple):
    guidance_type: GuidanceType = GuidanceType.LINEAR
    factor: float = 0.01
    start_step: int = 1
    end_step: int = 9999
    latent: Tensor | None = None


class SonarConfig(NamedTuple):
    momentum: float = 0.95
    momentum_hist: float = 0.75
    direction: float = 1.0
    init: HistoryType = HistoryType.ZERO
    noise_type: noise.NoiseType | None = None
    custom_noise: noise.CustomNoise | None = None
    rand_init_noise_type: noise.NoiseType | None = None
    guidance: GuidanceConfig | None = None


class SonarBase:
    DEFAULT_NOISE_TYPE = noise.NoiseType.GAUSSIAN

    def __init__(self, cfg: SonarConfig) -> None:
        self.history_d = None
        self.cfg = cfg
        self.noise_sampler = None

    def set_noise_sampler(
        self,
        x: Tensor,
        sigmas,
        noise_sampler: Callable | None,
        seed: int | None = None,
    ):
        sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
        if noise_sampler is not None and self.cfg.noise_type not in (
            None,
            self.DEFAULT_NOISE_TYPE,
        ):
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

    def init_hist_d(self, x: Tensor) -> None:
        if self.history_d is not None:
            return
        # memorize delta momentum
        if self.cfg.init == HistoryType.ZERO:
            self.history_d = 0
        elif self.cfg.init == HistoryType.SAMPLE:
            self.history_d = x
        elif self.cfg.init == HistoryType.RAND:
            ns = noise.get_noise_sampler(
                self.cfg.rand_init_noise_type,
                x,
                None,
                None,
                seed=self.extra_args.get("seed"),
                cpu=True,
                normalized=True,
            )
            self.history_d = ns(None, None)
        else:
            raise ValueError("Sonar sampler: bad history type")

    def update_hist(self, momentum_d):
        q = 1.0 - self.cfg.momentum_hist
        hd = self.history_d
        if isinstance(hd, int) and hd == 0:
            self.history_d = momentum_d
        else:
            self.history_d = (1.0 - q) * hd + q * momentum_d

    def momentum_step(self, x: Tensor, d: Tensor, dt: Tensor):
        if self.cfg.momentum == 1.0:
            return x + d * dt
        hd = self.history_d
        # correct current `d` with momentum
        p = (1.0 - self.cfg.momentum) * self.cfg.direction
        momentum_d = (1.0 - p) * d + p * hd

        # Euler method with momentum
        x = x + momentum_d * dt

        self.update_hist(momentum_d)

        return x


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
        avg_s = latent.mean(dim=[2, 3], keepdim=True)
        std_s = latent.std(dim=[2, 3], keepdim=True)
        return ((latent - avg_s) / std_s).to(latent.dtype)

    def guidance_step(self, step_index: int, x: Tensor, denoised: Tensor):
        if (self.guidance is None or self.guidance.factor == 0.0) or not (
            self.guidance.start_step <= (step_index + 1) <= self.guidance.end_step
        ):
            return x
        if self.ref_latent.device != x.device:
            self.ref_latent = self.ref_latent.to(device=x.device)
        if self.guidance.guidance_type == GuidanceType.LINEAR:
            return self.guidance_linear(x, self.ref_latent, self.guidance.factor)
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

    @staticmethod
    def guidance_euler(
        sigma: Tensor,
        sigma_next: Tensor,
        x: Tensor,
        denoised: Tensor,
        ref_latent: Tensor,
        factor: float = 0.2,
    ) -> Tensor:
        avg_t = denoised.mean(dim=[1, 2, 3], keepdim=True)
        std_t = denoised.std(dim=[1, 2, 3], keepdim=True)
        ref_img_shift = ref_latent * std_t + avg_t

        d = sampling.to_d(x, sigma, ref_img_shift)
        dt = (sigma_next - sigma) * factor
        return x + d * dt

    @staticmethod
    def guidance_linear(x: Tensor, ref_latent: Tensor, factor: float = 0.2) -> Tensor:
        avg_t = x.mean(dim=[1, 2, 3], keepdim=True)
        std_t = x.std(dim=[1, 2, 3], keepdim=True)
        ref_img_shift = ref_latent * std_t + avg_t
        return (1.0 - factor) * x + factor * ref_img_shift


class SonarWithGuidance(SonarBase, SonarGuidanceMixin):
    def __init__(self, *args: list[Any], **kwargs: dict[str, Any]):
        super().__init__(*args, **kwargs)
        SonarGuidanceMixin.__init__(self, self.cfg.guidance)


class SonarSampler(SonarWithGuidance):
    def __init__(
        self,
        model,
        sigmas,
        s_in,
        extra_args,
        *args: list[Any],
        **kwargs: dict[str, Any],
    ):
        super().__init__(*args, **kwargs)
        self.model = model
        self.sigmas = sigmas
        self.s_in = s_in
        self.extra_args = extra_args


class SonarEuler(SonarSampler):
    def __init__(
        self,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        *args: list[Any],
        **kwargs: dict[str, Any],
    ):
        super().__init__(*args, **kwargs)
        self.s_churn = s_churn
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise

    def step(
        self,
        step_index: int,
        sample: torch.FloatTensor,
    ):
        self.init_hist_d(sample)

        sigma, sigma_to = self.sigmas[step_index], self.sigmas[step_index + 1]

        gamma = (
            min(self.s_churn / (len(self.sigmas) - 1), 2**0.5 - 1)
            if self.s_tmin <= sigma <= self.s_tmax
            else 0.0
        )

        sigma_hat = sigma * (gamma + 1)

        if gamma > 0:
            noise = (
                self.noise_sampler(sigma, sigma_to)
                if self.noise_sampler
                else torch.randn_like(sample)
            )
            eps = noise * self.s_noise
            sample = sample + eps * (sigma_hat**2 - sigma**2) ** 0.5

        denoised = self.model(sample, sigma_hat * self.s_in, **self.extra_args)
        derivative = sampling.to_d(sample, sigma, denoised)
        dt = self.sigmas[step_index + 1] - sigma_hat

        result_sample = self.momentum_step(sample, derivative, dt)

        if self.sigmas[step_index + 1] > 0:
            result_sample = self.guidance_step(step_index, result_sample, denoised)

        return (
            result_sample,
            sigma,
            sigma_hat,
            denoised,
        )

    @classmethod
    @torch.no_grad()
    def sampler(
        cls,
        model,
        x,
        sigmas,
        extra_args=None,
        callback=None,
        disable=None,
        noise_sampler: Callable | None = None,
        sonar_config=None,
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float("inf"),
        s_noise=1.0,
    ):
        if sonar_config is None:
            sonar_config = SonarConfig()
        s_in = x.new_ones([x.shape[0]])
        sonar = cls(
            s_churn,
            s_tmin,
            s_tmax,
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
            x, sigma, sigma_hat, denoised = sonar.step(
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
        self.init_hist_d(sample)

        sigma_from, sigma_to = self.sigmas[step_index], self.sigmas[step_index + 1]
        sigma_down, sigma_up = sampling.get_ancestral_step(
            sigma_from,
            sigma_to,
            eta=self.eta,
        )

        denoised = self.model(sample, sigma_from * self.s_in, **self.extra_args)
        derivative = sampling.to_d(sample, sigma_from, denoised)
        dt = sigma_down - sigma_from

        result_sample = self.momentum_step(sample, derivative, dt)
        if sigma_to > 0:
            result_sample = self.guidance_step(step_index, result_sample, denoised)
            result_sample = (
                result_sample
                + self.noise_sampler(sigma_from, sigma_to) * self.s_noise * sigma_up
            )

        return (
            result_sample,
            sigma_from,
            sigma_from,
            denoised,
        )

    @classmethod
    @torch.no_grad()
    def sampler(
        cls,
        model,
        x,
        sigmas,
        extra_args=None,
        callback=None,
        disable=None,
        sonar_config=None,
        eta=1.0,
        s_noise=1.0,
        noise_sampler: Callable | None = None,
    ):
        if sonar_config is None:
            sonar_config = SonarConfig()
        s_in = x.new_ones([x.shape[0]])
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
            x, sigma, sigma_hat, denoised = sonar.step(
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
    def sigma_fn(t) -> float:
        return t.neg().exp()

    @staticmethod
    def t_fn(sigma) -> float:
        return sigma.log.neg()

    # DPM++ solver algorithm copied from ComfyUI source.
    def momentum_step(
        self,
        step_index,
        x: Tensor,
        denoised: Tensor,
        sigma_from,
        sigma_to,
        sigma_down,
    ):
        if sigma_to == 0:
            derivative = sampling.to_d(x, sigma_from, denoised)
            dt = sigma_down - sigma_from
            return super().momentum_step(x, derivative, dt)

        def sigma_fn(t):
            return t.neg().exp()

        def t_fn(sigma):
            return sigma.log().neg()

        hd = self.history_d
        p = (1.0 - self.cfg.momentum) * self.cfg.direction

        r = 1 / 2
        # DPM-Solver++
        t, t_next = t_fn(sigma_from), t_fn(sigma_to)
        h = t_next - t
        s = t + h * r
        fac = 1 / (2 * r)

        # Step 1
        sd, su = sampling.get_ancestral_step(sigma_fn(t), sigma_fn(s), self.eta)
        s_ = t_fn(sd)
        diff_2 = (t - s_).expm1() * denoised
        momentum_d = (1.0 - p) * diff_2 + p * hd
        self.update_hist(momentum_d)
        hd = self.history_d
        x_2 = (sigma_fn(s_) / sigma_fn(t)) * x - momentum_d
        x_2 = x_2 + self.noise_sampler(sigma_fn(t), sigma_fn(s)) * self.s_noise * su
        denoised_2 = self.model(x_2, sigma_fn(s) * self.s_in, **self.extra_args)

        # Step 2
        sd, su = sampling.get_ancestral_step(
            sigma_fn(t),
            sigma_fn(t_next),
            self.eta,
        )
        t_next_ = t_fn(sd)
        denoised_d = (1 - fac) * denoised + fac * denoised_2
        diff_1 = (t - t_next_).expm1() * denoised_d
        momentum_d = (1.0 - p) * diff_1 + p * hd
        self.update_hist(momentum_d)
        x = (sigma_fn(t_next_) / sigma_fn(t)) * x - momentum_d
        x = self.guidance_step(step_index, x, denoised_d)
        return x + self.noise_sampler(sigma_fn(t), sigma_fn(t_next)) * self.s_noise * su

    def step(
        self,
        step_index: int,
        sample: torch.FloatTensor,
    ):
        def sigma_fn(t):
            return t.neg().exp()

        def t_fn(sigma):
            return sigma.log().neg()

        self.init_hist_d(sample)

        sigma_from, sigma_to = self.sigmas[step_index], self.sigmas[step_index + 1]
        sigma_down, sigma_up = sampling.get_ancestral_step(
            sigma_from,
            sigma_to,
            eta=self.eta,
        )

        denoised = self.model(sample, sigma_from * self.s_in, **self.extra_args)
        result_sample = self.momentum_step(
            step_index,
            sample,
            denoised,
            sigma_from,
            sigma_to,
            sigma_down,
        )

        return (
            result_sample,
            sigma_from,
            sigma_from,
            denoised,
        )

    @classmethod
    @torch.no_grad()
    def sampler(
        cls,
        model,
        x,
        sigmas,
        extra_args=None,
        callback=None,
        disable=None,
        sonar_config=None,
        eta=1.0,
        s_noise=1.0,
        noise_sampler=None,
    ):
        if sonar_config is None:
            sonar_config = SonarConfig()
        s_in = x.new_ones([x.shape[0]])
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
            x, sigma, sigma_hat, denoised = sonar.step(
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


def add_samplers():
    import importlib

    from comfy.samplers import KSampler, k_diffusion_sampling

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
