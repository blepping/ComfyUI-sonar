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


class SonarConfig(NamedTuple):
    momentum: float = 0.95
    momentum_hist: float = 0.75
    direction: float = 1.0
    init: HistoryType = HistoryType.ZERO
    noise_type: noise.NoiseType | None = None
    guidance: GuidanceConfig | None = None


class SonarBase:
    def __init__(
        self,
        cfg: SonarConfig,
    ) -> None:
        self.history_d = None
        self.cfg = cfg

    def init_hist_d(self, x: Tensor) -> None:
        if self.history_d is not None:
            return
        # memorize delta momentum
        if self.cfg.init == HistoryType.ZERO:
            self.history_d = 0
        elif self.cfg.init == HistoryType.SAMPLE:
            self.history_d = x
        elif self.cfg.init == HistoryType.RAND:
            self.history_d = torch.randn_like(x)
        else:
            raise ValueError("Sonar sampler: bad history type")

    def momentum_step(self, x: Tensor, d: Tensor, dt: Tensor):
        if self.cfg.momentum == 1.0:
            return x + d * dt
        hd = self.history_d
        # correct current `d` with momentum
        p = (1.0 - self.cfg.momentum) * self.cfg.direction
        momentum_d = (1.0 - p) * d + p * hd

        # Euler method with momentum
        x = x + momentum_d * dt

        # update momentum history
        q = 1.0 - self.cfg.momentum_hist
        if isinstance(hd, int) and hd == 0:
            hd = momentum_d
        else:
            hd = (1.0 - q) * hd + q * momentum_d
        self.history_d = hd
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
            return self.guidance_linear(x)
        if self.guidance.guidance_type == GuidanceType.EULER:
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
        dt = (sigma_next - sigma) * self.guidance.factor
        return x + d * dt

    def guidance_linear(
        self,
        x: Tensor,
    ):
        avg_t = x.mean(dim=[1, 2, 3], keepdim=True)
        std_t = x.std(dim=[1, 2, 3], keepdim=True)
        ref_img_shift = self.ref_latent * std_t + avg_t
        return (1.0 - self.guidance.factor) * x + self.guidance.factor * ref_img_shift


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
        sonar_config=None,
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float("inf"),
        s_noise=1.0,
    ):
        if sonar_config is None:
            raise ValueError("Missing Sonar config")
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
        noise_sampler=None,
    ):
        if sonar_config is None:
            raise ValueError("Missing Sonar config")
        if (
            noise_sampler is not None
            and sonar_config.noise_type != noise.NoiseType.GAUSSIAN
        ):
            # Possibly we should just use the supplied already-created noise sampler here.
            raise ValueError(
                "Unexpected noise_sampler presence with non-default noise type requested",
            )
        sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
        noise_sampler = noise.get_noise_sampler(
            sonar_config.noise_type,
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
            sonar_config,
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
