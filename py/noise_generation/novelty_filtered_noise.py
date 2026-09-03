# ruff: noqa: ANN002, ANN003
from __future__ import annotations

import math
from functools import partial
from typing import TYPE_CHECKING

import torch

from .base import NoiseGenerator

if TYPE_CHECKING:
    from collections.abc import Callable

F = torch.nn.functional


def sum_rms_blend(
    a: torch.Tensor,
    b: torch.Tensor,
    t: torch.Tensor | float = 1.0,
    *,
    orig_shape: torch.Size | tuple[int, ...],
    dims_a: tuple[int, ...] = (1,),
    dims_b: tuple[int, ...] = (-1, -2),
) -> torch.Tensor:
    rms_a = a / math.prod(orig_shape[d] for d in dims_a) ** 0.5
    rms_b = b / math.prod(orig_shape[d] for d in dims_b) ** 0.5
    variance_a = rms_a.pow_(2.0)
    variance_b = rms_b.pow_(2.0)
    result = variance_a
    result += variance_b * t
    result /= 1.0 + t
    result **= 0.5
    return result


def metrics_blend(
    a: torch.Tensor,
    b: torch.Tensor,
    t: torch.Tensor | float = 1.0,
    *,
    orig_shape: torch.Size | tuple[int, ...],
    dims_a: tuple[int, ...] = (-1, -2),
    dims_b: tuple[int, ...] = (1,),
    use_rms: bool = True,
    rms_power: float = 2.0,
) -> torch.Tensor:
    count_a = math.prod(orig_shape[d] for d in dims_a)
    count_b = math.prod(orig_shape[d] for d in dims_b)
    denom_a = count_a ** (1 / rms_power) if use_rms else count_a
    denom_b = count_b ** (1 / rms_power) if use_rms else count_b
    curr_a = a / denom_a
    curr_b = b / denom_b
    if use_rms:
        curr_a **= rms_power
        curr_b = curr_b.pow_(rms_power) * t
    result = curr_b.add_(curr_a)
    result /= 1.0 + t
    return result.pow_(1.0 / rms_power) if use_rms else result


class NoveltyFilteredNoiseGenerator(NoiseGenerator):
    name = "novelty"

    initial_noise_state: torch.Tensor | None = None
    noise_state: torch.Tensor | None = None
    blend_function: Callable | None = None

    @classmethod
    def ng_params(cls):
        return super().ng_params() | {
            "skip_initial": 1,
            "iters_per_call": 1,
            "dim_groups": ((1,), (-1, -2)),
            "blend_ratio": 1.0,
            "blend_function": None,
            "update_blend_ratio": 1.0,
            "update_blend_function": None,
            "noise_sampler": None,
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.blend_function is None:
            raise ValueError("Missing blend function!")

    def generate(self, *args) -> torch.Tensor:
        ng = (
            partial(self.noise_sampler, *args) if self.noise_sampler else self.rand_like
        )
        noise_state = self.noise_state
        had_state = self.noise_state is not None
        it_counter = 0 if had_state else 0 - self.skip_initial
        its_call = max(1, self.iters_per_call)
        bf = self.blend_function
        blend_ratio = self.blend_ratio
        update_blend_ratio = self.update_blend_ratio
        ubf = self.update_blend_function
        if ubf is None:
            # Linear weighted average
            def ubf(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
                return (b * t).add_(a).div_(1.0 + abs(t))

        call_initial_noise = None
        curr_noise = None
        call_noise_state = None
        while it_counter < its_call:
            if noise_state is None:
                noise_state = ng()
                self.initial_noise_state = noise_state.clone()
                self.noise_state = noise_state.clone()
                continue
            curr_noise = ng()
            if call_initial_noise is None:
                call_initial_noise = curr_noise.clone()
            seen = {id(curr_noise)}
            for ortho_target in (
                self.initial_noise_state,
                call_initial_noise if it_counter > 0 else None,
                noise_state,
            ):
                tid = id(ortho_target)
                if ortho_target is None or tid in seen:
                    continue
                seen.add(tid)
                curr_noise = bf(ortho_target, curr_noise, blend_ratio)
                curr_noise -= ortho_target
            it_counter += 1
            if it_counter < 1:
                self.noise_state = curr_noise.clone()
                noise_state = curr_noise
                continue
            if call_noise_state is None:
                call_noise_state = curr_noise
            else:
                call_noise_state = ubf(call_noise_state, curr_noise, update_blend_ratio)
        if call_noise_state is None:
            raise RuntimeError("Unexpected unpopulated call_noise_state!")
        # self.noise_state = call_noise_state.clone()
        self.noise_state = ubf(noise_state, call_noise_state, update_blend_ratio)
        return call_noise_state

    # def generate(self, *args) -> torch.Tensor:
    #     ng = (
    #         partial(self.noise_sampler, *args) if self.noise_sampler else self.rand_like
    #     )
    #     noise_state = self.noise_state
    #     had_state = self.noise_state is not None
    #     it_counter = 0 if had_state else 0 - self.skip_initial
    #     its_call = self.iters_per_call
    #     bf = self.blend_function
    #     blend_ratio = self.blend_ratio
    #     update_blend_ratio = self.update_blend_ratio
    #     ubf = self.update_blend_function
    #     if ubf is None or True:
    #         # Linear weighted average
    #         def ubf(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    #             return (b * t).add_(a).div_(1.0 + abs(t))

    #     call_initial_noise = None
    #     while it_counter < its_call:
    #         if noise_state is None:
    #             noise_state = ng()
    #             self.initial_noise_state = noise_state.clone()
    #             continue
    #         curr_noise = ng()
    #         it_counter += 1
    #         if it_counter < 1:
    #             noise_state = curr_noise
    #             continue
    #         if call_initial_noise is None and self.iters_per_call > 1:
    #             call_initial_noise = curr_noise.clone()
    #         for ortho_target in (
    #             self.initial_noise_state,
    #             call_initial_noise if it_counter > 0 else None,
    #             noise_state,
    #         ):
    #             if ortho_target is None:
    #                 continue
    #             curr_noise = bf(ortho_target, curr_noise, blend_ratio)
    #             curr_noise -= ortho_target
    #         # curr_noise = bf(noise_state, ng(), blend_ratio).sub_(noise_state)
    #         noise_state = ubf(noise_state, curr_noise, update_blend_ratio)
    #     self.noise_state = noise_state.clone()
    #     return noise_state
