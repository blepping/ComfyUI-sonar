# Adapted from https://github.com/alexblattner/modified-euler-samplers-for-sonar-diffusers and https://github.com/Kahsolt/stable-diffusion-webui-sonar

from __future__ import annotations

import torch
from comfy import samplers
from comfy.k_diffusion import sampling
from torch import Tensor
from tqdm.auto import trange


class SonarEuler:
    def __init__(
        self,
        history_type="ZERO",
        momentum=0.95,
        momentum_hist=0.75,
        direction=1.0,
    ):
        if history_type not in ("ZERO", "RAND", "SAMPLE"):
            raise ValueError("Bad history_type: must be one of zero, rand, samples")
        self.history_d = None
        self.history_type = history_type
        self.momentum = momentum
        self.momentum_hist = momentum_hist
        self.direction = direction

    def init_hist_d(self, x: Tensor) -> None:
        if self.history_d is not None:
            return
        # memorize delta momentum
        if self.history_type == "ZERO":
            self.history_d = 0
        elif self.history_type == "SAMPLE":
            self.history_d = x
        elif self.history_type == "RAND":
            self.history_d = torch.randn_like(x)

    def momentum_step(self, x: Tensor, d: Tensor, dt: Tensor):
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

    def step(
        self,
        step_index,
        model,
        sample: torch.FloatTensor,
        s_in,
        extra_args,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
    ):
        self.init_hist_d(sample)

        sigma = self.sigmas[step_index]

        gamma = (
            min(s_churn / (len(self.sigmas) - 1), 2**0.5 - 1)
            if s_tmin <= sigma <= s_tmax
            else 0.0
        )

        sigma_hat = sigma * (gamma + 1)

        if gamma > 0:
            noise = torch.randn_like(sample.shape)

            eps = noise * s_noise
            sample = sample + eps * (sigma_hat**2 - sigma**2) ** 0.5

        denoised = model(sample, sigma_hat * s_in, **extra_args)
        derivative = sampling.to_d(sample, sigma, denoised)
        dt = self.sigmas[step_index + 1] - sigma_hat

        return (
            self.momentum_step(sample, derivative, dt),
            sigma,
            sigma_hat,
            denoised,
        )

    def step_ancestral(
        self,
        step_index,
        model,
        sample: torch.FloatTensor,
        noise_sampler,
        s_in,
        extra_args,
        eta=1.0,
        s_noise: float = 1.0,
    ):
        self.init_hist_d(sample)

        sigma_from = self.sigmas[step_index]
        sigma_to = self.sigmas[step_index + 1]
        sigma_down, sigma_up = sampling.get_ancestral_step(
            sigma_from,
            sigma_to,
            eta=eta,
        )

        denoised = model(sample, sigma_from * s_in, **extra_args)
        derivative = sampling.to_d(sample, sigma_from, denoised)
        dt = sigma_down - sigma_from

        result_sample = self.momentum_step(sample, derivative, dt)
        if sigma_to > 0:
            result_sample = (
                result_sample + noise_sampler(sigma_from, sigma_to) * s_noise * sigma_up
            )

        return (
            result_sample,
            sigma_from,
            sigma_from,
            denoised,
        )


@torch.no_grad()
def sample_sonar_euler(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    momentum=0.95,
    momentum_hist=0.75,
    momentum_init="ZERO",
    direction=1.0,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
):
    s = SonarEuler(
        momentum=momentum,
        momentum_hist=momentum_hist,
        history_type=momentum_init,
        direction=direction,
    )
    s.sigmas = sigmas
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    for i in trange(len(sigmas) - 1, disable=disable):
        x, sigma, sigma_hat, denoised = s.step(
            i,
            model,
            x,
            s_in,
            extra_args,
            s_churn,
            s_tmin,
            s_tmax,
            s_noise,
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


@torch.no_grad()
def sample_sonar_euler_ancestral(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    momentum=0.95,
    momentum_hist=0.75,
    momentum_init="ZERO",
    direction=1.0,
    eta=1.0,
    s_noise=1.0,
    noise_sampler=None,
):
    s = SonarEuler(
        momentum=momentum,
        momentum_hist=momentum_hist,
        history_type=momentum_init,
        direction=direction,
    )
    s.sigmas = sigmas
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = (
        sampling.default_noise_sampler(x) if noise_sampler is None else noise_sampler
    )
    s_in = x.new_ones([x.shape[0]])

    for i in trange(len(sigmas) - 1, disable=disable):
        x, sigma, sigma_hat, denoised = s.step_ancestral(
            i,
            model,
            x,
            noise_sampler,
            s_in,
            extra_args,
            eta,
            s_noise,
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


class SamplerSonarEuler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "momentum": (
                    "FLOAT",
                    {
                        "default": 0.95,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "round": False,
                    },
                ),
                "momentum_hist": (
                    "FLOAT",
                    {
                        "default": 0.75,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "round": False,
                    },
                ),
                "momentum_init": (("ZERO", "SAMPLE", "RAND"),),
                "direction": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -2.0,
                        "max": 2.0,
                        "step": 0.01,
                        "round": False,
                    },
                ),
                "s_noise": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.01,
                        "round": False,
                    },
                ),
            },
        }

    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, momentum, momentum_hist, momentum_init, direction, s_noise):
        return (
            samplers.KSAMPLER(
                sample_sonar_euler,
                {
                    "momentum_init": momentum_init,
                    "momentum": momentum,
                    "momentum_hist": momentum_hist,
                    "direction": direction,
                    "s_noise": s_noise,
                },
            ),
        )


class SamplerSonarEulerAncestral(SamplerSonarEuler):
    @classmethod
    def INPUT_TYPES(cls):
        result = SamplerSonarEuler.INPUT_TYPES()
        result["required"]["eta"] = (
            "FLOAT",
            {
                "default": 1.0,
                "min": 0.0,
                "max": 100.0,
                "step": 0.01,
                "round": False,
            },
        )
        return result

    def get_sampler(
        self,
        momentum,
        momentum_hist,
        momentum_init,
        direction,
        eta,
        s_noise,
    ):
        return (
            samplers.KSAMPLER(
                sample_sonar_euler_ancestral,
                {
                    "momentum_init": momentum_init,
                    "momentum": momentum,
                    "momentum_hist": momentum_hist,
                    "direction": direction,
                    "eta": eta,
                    "s_noise": s_noise,
                },
            ),
        )
