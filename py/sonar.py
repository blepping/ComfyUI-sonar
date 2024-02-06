# Sonar sampler part adapted from https://github.com/alexblattner/modified-euler-samplers-for-sonar-diffusers and https://github.com/Kahsolt/stable-diffusion-webui-sonar

from __future__ import annotations

from enum import Enum, auto
from typing import Any, NamedTuple

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


class SonarBase:
    def __init__(
        self,
        history_type: HistoryType | None = None,
        momentum: float = 0.95,
        momentum_hist: float = 0.75,
        direction: float = 1.0,
    ) -> None:
        self.history_d = None
        self.history_type = HistoryType.ZERO if history_type is None else history_type
        self.momentum = momentum
        self.momentum_hist = momentum_hist
        self.direction = direction

    def init_hist_d(self, x: Tensor) -> None:
        if self.history_d is not None:
            return
        # memorize delta momentum
        if self.history_type == HistoryType.ZERO:
            self.history_d = 0
        elif self.history_type == HistoryType.SAMPLE:
            self.history_d = x
        elif self.history_type == HistoryType.RAND:
            self.history_d = torch.randn_like(x)
        else:
            raise ValueError("Sonar sampler: bad history type")

    def momentum_step(self, x: Tensor, d: Tensor, dt: Tensor):
        if self.momentum == 1.0:
            return x + d * dt
        hd = self.history_d
        # correct current `d` with momentum
        p = (1.0 - self.momentum) * self.direction
        momentum_d = (1.0 - p) * d + p * hd

        # Euler method with momentum
        x = x + momentum_d * dt

        # update momentum history
        q = 1.0 - self.momentum_hist
        if isinstance(hd, int) and hd == 0:
            hd = momentum_d
        else:
            hd = (1.0 - q) * hd + q * momentum_d
        self.history_d = hd
        return x


class SonarSampler(SonarBase):
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

        sigma = self.sigmas[step_index]

        gamma = (
            min(self.s_churn / (len(self.sigmas) - 1), 2**0.5 - 1)
            if self.s_tmin <= sigma <= self.s_tmax
            else 0.0
        )

        sigma_hat = sigma * (gamma + 1)

        if gamma > 0:
            noise = torch.randn_like(sample.shape)

            eps = noise * self.s_noise
            sample = sample + eps * (sigma_hat**2 - sigma**2) ** 0.5

        denoised = self.model(sample, sigma_hat * self.s_in, **self.extra_args)
        derivative = sampling.to_d(sample, sigma, denoised)
        dt = self.sigmas[step_index + 1] - sigma_hat

        return (
            self.momentum_step(sample, derivative, dt),
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
        momentum=0.95,
        momentum_hist=0.75,
        momentum_init=HistoryType.ZERO,
        direction=1.0,
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float("inf"),
        s_noise=1.0,
    ):
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
            momentum=momentum,
            momentum_hist=momentum_hist,
            history_type=momentum_init,
            direction=direction,
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
        noise_sampler,
        eta: float = 1.0,
        s_noise: float = 1.0,
        *args: list[Any],
        **kwargs: dict[str, Any],
    ):
        super().__init__(*args, **kwargs)
        self.noise_sampler = noise_sampler
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
        momentum=0.95,
        momentum_hist=0.75,
        momentum_init=HistoryType.ZERO,
        noise_type="gaussian",
        direction=1.0,
        eta=1.0,
        s_noise=1.0,
        noise_sampler=None,
    ):
        if noise_type != "gaussian" and noise_sampler is not None:
            # Possibly we should just use the supplied already-created noise sampler here.
            raise ValueError(
                "Unexpected noise_sampler presence with non-default noise type requested",
            )
        sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
        noise_sampler = noise.get_noise_sampler(
            noise_type,
            x,
            sigma_min,
            sigma_max,
            seed=None,
            use_cpu=True,
        )
        s_in = x.new_ones([x.shape[0]])
        sonar = cls(
            noise_sampler,
            eta,
            s_noise,
            model,
            sigmas,
            s_in,
            {} if extra_args is None else extra_args,
            momentum=momentum,
            momentum_hist=momentum_hist,
            history_type=momentum_init,
            direction=direction,
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


class SonarGuidanceMixin:
    def __init__(
        self,
        cfg: GuidanceConfig | None = None,
    ) -> None:
        self.cfg = cfg
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
        if (
            self.ref_latent is None
            or self.cfg.guidance_type is None
            or self.cfg.factor == 0.0
        ) or not (self.cfg.start_step <= (step_index + 1) <= self.cfg.end_step):
            return x
        if self.ref_latent.device != x.device:
            self.ref_latent = self.ref_latent.to(device=x.device)
        if self.cfg.guidance_type == GuidanceType.LINEAR:
            return self.guidance_linear(x)
        if self.cfg.guidance_type == GuidanceType.EULER:
            return self.guidance_euler(step_index, x, denoised)
        raise ValueError("Sonar: Guidance: Unknown guidance type")

    def guidance_euler(
        self,
        step_index: int,
        x: Tensor,
        denoised: Tensor,
    ):
        avg_t = denoised.mean(dim=[1, 2, 3], keepdim=True)
        std_t = denoised.std(dim=[1, 2, 3], keepdim=True)
        ref_img_shift = self.ref_latent * std_t + avg_t
        sigma, sigma_next = self.sigmas[step_index], self.sigmas[step_index + 1]

        d = sampling.to_d(x, sigma, ref_img_shift)
        dt = (sigma_next - sigma) * self.cfg.factor
        return x + d * dt

    def guidance_linear(
        self,
        x: Tensor,
    ):
        avg_t = x.mean(dim=[1, 2, 3], keepdim=True)
        std_t = x.std(dim=[1, 2, 3], keepdim=True)
        ref_img_shift = self.ref_latent * std_t + avg_t
        return (1.0 - self.cfg.factor) * x + self.cfg.factor * ref_img_shift


class SonarNaive(SonarSampler, SonarGuidanceMixin):
    def __init__(
        self,
        noise_sampler,
        s_noise: float = 1.0,
        guidance: GuidanceConfig | None = None,
        *args: list[Any],
        **kwargs: dict[str, Any],
    ):
        super().__init__(*args, **kwargs)
        SonarGuidanceMixin.__init__(self, guidance)
        self.noise_sampler = noise_sampler
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
            eta=1.0,
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
        momentum=0.95,
        momentum_hist=0.75,
        momentum_init=HistoryType.ZERO,
        noise_type="gaussian",
        direction=1.0,
        s_noise=1.0,
        noise_sampler=None,
        guidance: GuidanceConfig | None = None,
    ):
        if noise_type != "gaussian" and noise_sampler is not None:
            # Possibly we should just use the supplied already-created noise sampler here.
            raise ValueError(
                "Unexpected noise_sampler presence with non-default noise type requested",
            )
        sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
        noise_sampler = noise.get_noise_sampler(
            noise_type,
            x,
            sigma_min,
            sigma_max,
            seed=None,
            use_cpu=True,
        )
        s_in = x.new_ones([x.shape[0]])
        sonar = cls(
            noise_sampler,
            s_noise,
            guidance,
            model,
            sigmas,
            s_in,
            {} if extra_args is None else extra_args,
            momentum=momentum,
            momentum_hist=momentum_hist,
            history_type=momentum_init,
            direction=direction,
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
