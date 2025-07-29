from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

import torch

from . import utils

if TYPE_CHECKING:
    from types import Sequence


class SonarLatentOperation:
    EXTENDED_LATENT_OPERATION = True

    def __init__(
        self,
        *,
        start_sigma: float = math.inf,
        end_sigma: float = 0.0,
        op=None,
    ):
        self.start_sigma = start_sigma if start_sigma >= 0 else math.inf
        self.end_sigma = end_sigma
        self.op = op

    def enabled(self, sigma: torch.Tensor | float | None = None) -> bool:
        if isinstance(sigma, torch.Tensor):
            sigma = sigma.detach().max().cpu().item()
        return sigma is None or self.end_sigma <= sigma <= self.start_sigma

    def call_op(
        self,
        t: torch.Tensor,
        *args: list,
        op=None,
        **kwargs: dict,
    ) -> torch.Tensor:
        if op is None:
            op = self.op
        if op is None:
            return t
        if not getattr(op, "EXTENDED_LATENT_OPERATION", False):
            return op(latent=t)
        return op(*args, latent=t, **kwargs)

    def __call__(
        self,
        latent: torch.Tensor,
        *,
        sigma: torch.Tensor | float | None = None,
        **kwargs: dict,
    ) -> torch.Tensor:
        if not self.enabled(sigma=sigma):
            return latent
        return self.call_op(latent, sigma=sigma, **kwargs)


class SonarLatentOperationAdvanced(SonarLatentOperation):
    def __init__(
        self,
        *,
        blend_mode: str,
        blend_strength: float,
        input_multiplier: float,
        output_multiplier: float,
        difference_multiplier: float,
        ops: Sequence,
        op_alt=None,
        **kwargs: dict,
    ) -> None:
        super().__init__(**kwargs)
        self.blend_function = utils.BLENDING_MODES[blend_mode]
        self.blend_strength = blend_strength
        self.input_multiplier = input_multiplier
        self.output_multiplier = output_multiplier
        self.difference_multiplier = difference_multiplier
        self.op_alt = op_alt
        self.ops = ops

    def __call__(
        self,
        latent: torch.Tensor,
        *,
        sigma: torch.Tensor | float | None = None,
        **kwargs: dict,
    ) -> torch.Tensor:
        t = latent
        enabled = self.enabled(sigma)
        if not enabled:
            return (
                t
                if self.op_alt is None
                else self.call_op(t, sigma=sigma, op=self.op_alt, **kwargs)
            )
        output = t * self.input_multiplier if self.input_multiplier != 1.0 else t
        for op in self.ops:
            output = self.call_op(output, sigma=sigma, op=op, **kwargs)
        diff = (
            output * self.output_multiplier if self.output_multiplier == 1.0 else output
        ) - t
        if self.difference_multiplier != 1.0:
            diff *= self.difference_multiplier
        return self.blend_function(t, diff, self.blend_strength)


class SonarLatentOperationNoise(SonarLatentOperation):
    def __init__(
        self,
        *args: list,
        custom_noise,
        scale_to_sigma: bool = False,
        cpu_noise: bool = False,
        normalize: bool = True,
        lazy_noise_sampler: bool = False,
        **kwargs: dict,
    ):
        super().__init__(*args, **kwargs)
        self.custom_noise = custom_noise
        self.normalize = normalize
        self.scale_to_sigma = scale_to_sigma
        self.cpu_noise = cpu_noise
        self.lazy_noise_sampler = lazy_noise_sampler
        self.noise_sampler = None
        self.cache_id = None

    def __call__(
        self,
        latent: torch.Tensor,
        *,
        sigma: torch.Tensor | float | None = None,
        **kwargs: dict,
    ) -> torch.Tensor:
        t = latent
        enabled = self.enabled(sigma)
        if not enabled:
            return t
        if isinstance(sigma, float):
            sigma = t.new_full((1,), sigma)
        make_ns = not self.lazy_noise_sampler or self.noise_sampler is None
        sigma_min = sigma_max = sigma_next = None
        sample_sigmas = (
            kwargs.get("raw_args", {})
            .get("model_options", {})
            .get("transformer_options", {})
            .get("sample_sigmas")
        )
        if sample_sigmas is not None and sigma is not None:
            guessed_step = (sample_sigmas - sigma).abs().argmin().detach().item()
            guessed_sigma = sample_sigmas[guessed_step].max().detach().item()
            if guessed_sigma == sigma and guessed_step + 1 < len(sample_sigmas):
                sigma_next = sample_sigmas[guessed_step + 1]
        if self.lazy_noise_sampler and not make_ns:
            cache_id = (
                id(sample_sigmas) if isinstance(sample_sigmas, torch.Tensor) else None
            )
            make_ns = cache_id is None or cache_id != self.cache_id
            self.cache_id = cache_id
            if make_ns and sample_sigmas is not None:
                sigmas_min = sample_sigmas[sample_sigmas > 0]
                sigma_min = (
                    sigmas_min.min().detach().item() if torch.any(sigmas_min) else 0.0
                )
                del sigmas_min
                sigma_max = sample_sigmas.max().detach().item()
        ns = (
            self.custom_noise.make_noise_sampler(
                t,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                normalized=self.normalize,
                seed=torch.randint(1, 1 << 31, (), device="cpu").item(),
                cpu=self.cpu_noise,
            )
            if make_ns
            else self.noise_sampler
        )
        if make_ns and self.lazy_noise_sampler:
            self.noise_sampler = ns
        noise = ns(sigma, sigma if sigma_next is None else sigma_next)
        if self.scale_to_sigma and sigma is not None:
            noise *= sigma
        noise += t
        return noise


class SonarLatentOperationSetSeed(SonarLatentOperation):
    def __init__(self, *args: list, seed: int, restore_rng_state: bool, **kwargs: dict):
        super().__init__(*args, **kwargs)
        self.seed = seed
        self.restore_rng_state = restore_rng_state

    def __call__(self, *args: list, **kwargs: dict) -> torch.Tensor:
        if self.restore_rng_state:
            pyrandst = random.getstate()
            torchrandst = torch.random.get_rng_state()
        else:
            pyrandst = torchrandst = None
        try:
            torch.manual_seed(self.seed)
            random.seed(self.seed)
            result = super().__call__(*args, **kwargs)
        finally:
            if self.restore_rng_state:
                torch.random.set_rng_state(torchrandst)
                random.setstate(pyrandst)
        return result
