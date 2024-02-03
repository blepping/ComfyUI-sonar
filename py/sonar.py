# Sonar sampler part adapted from https://github.com/alexblattner/modified-euler-samplers-for-sonar-diffusers and https://github.com/Kahsolt/stable-diffusion-webui-sonar

from __future__ import annotations

from enum import Enum, auto

import torch
from comfy import samplers
from comfy.k_diffusion import sampling
from torch import Tensor
from tqdm.auto import trange

from . import noise


class HistoryType(Enum):
    ZERO = auto()
    RAND = auto()
    SAMPLE = auto()


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
        *args,
        **kwargs,
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
        *args,
        **kwargs,
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
        *args,
        **kwargs,
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


class SamplerNodeSonarEuler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "momentum": (
                    "FLOAT",
                    {
                        "default": 0.95,
                        "min": -0.5,
                        "max": 2.5,
                        "step": 0.01,
                        "round": False,
                    },
                ),
                "momentum_hist": (
                    "FLOAT",
                    {
                        "default": 0.75,
                        "min": -1.5,
                        "max": 1.5,
                        "step": 0.01,
                        "round": False,
                    },
                ),
                "momentum_init": (("ZERO", "SAMPLE", "RAND"),),
                "direction": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -30.0,
                        "max": 15.0,
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
                SonarEuler.sampler,
                {
                    "momentum_init": HistoryType[momentum_init],
                    "momentum": momentum,
                    "momentum_hist": momentum_hist,
                    "direction": direction,
                    "s_noise": s_noise,
                },
            ),
        )


class SamplerNodeSonarEulerAncestral(SamplerNodeSonarEuler):
    @classmethod
    def INPUT_TYPES(cls):
        result = super().INPUT_TYPES()
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
        result["required"]["noise_type"] = (
            (
                "gaussian",
                "uniform",
                "brownian",
                "perlin",
                "studentt",
                "studentt_test",
                "highres_pyramid",
                "pink",
                # "green_ish",
            ),
        )
        return result

    def get_sampler(
        self,
        momentum,
        momentum_hist,
        momentum_init,
        direction,
        noise_type,
        eta,
        s_noise,
    ):
        return (
            samplers.KSAMPLER(
                SonarEulerAncestral.sampler,
                {
                    "momentum_init": HistoryType[momentum_init],
                    "momentum": momentum,
                    "momentum_hist": momentum_hist,
                    "direction": direction,
                    "noise_type": noise_type,
                    "eta": eta,
                    "s_noise": s_noise,
                },
            ),
        )


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
