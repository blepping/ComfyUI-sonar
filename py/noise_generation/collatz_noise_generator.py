# ruff: noqa: ANN002
from __future__ import annotations

import math
from typing import TYPE_CHECKING, ClassVar

import torch
from comfy.model_management import throw_exception_if_processing_interrupted

from .. import utils
from ..utils import fallback, normalize_to_scale, tensor_to
from .base import NoiseGenerator

if TYPE_CHECKING:
    from collections.abc import Sequence

F = torch.nn.functional


class CollatzNoiseGenerator(NoiseGenerator):
    name = "collatz"

    @classmethod
    def ng_params(cls):
        return super().ng_params() | {
            "adjust_scale": False,
            "iteration_sign_flipping": True,
            "chain_length": (1, 1, 2, 2, 3, 3),
            "iterations": 10,
            "rmin": -8000.0,
            "rmax": 8000.0,
            "flatten": False,
            "dims": (-1, -1, -2, -2),
            # values, ratios, mults, adds
            # seed_x_ratios, seed_x_mults, seed_x_adds
            # noise_x_ratios, noise_x_mults, noise_x_adds
            "output_mode": "values",
            "quantile": 0.5,
            "quantile_strategy": "clamp",
            "noise_dtype": torch.float32,
            "integer_math": True,
            "even_multiplier": 0.5,
            "even_addition": 0.0,
            "odd_multiplier": 3.0,
            "odd_addition": 1.0,
            "add_preserves_sign": True,
            "chain_offset": 5,
            "break_loops": True,
            "seed_mode": "default",
            "seed_noise_sampler": None,
            "mix_noise_sampler": None,
        }

    @staticmethod
    def _get_iter_slices(n_dims, dim, idx, stride) -> tuple:
        result = [slice(None)] * n_dims
        result[dim] = slice(idx, None, stride)
        return tuple(result)

    def _generate_iteration(
        self,
        *args,
        dim: int,
        chain_length: int,
        flatten: False,
        shape=None,
    ):
        dtype, device = self.dtype, self.device
        out_shape = shape = fallback(shape, self.shape)
        if dim >= len(shape):
            raise ValueError("Requested dimension out of range")
        rmin, rmax = self.rmin, self.rmax
        emul, eadd = self.even_multiplier, self.even_addition
        omul, oadd = self.odd_multiplier, self.odd_addition
        keepsign = self.add_preserves_sign
        intmode = self.integer_math
        rmaxsubmin = rmax - rmin
        if flatten:
            shape = torch.Size((*shape[:dim], math.prod(shape[dim:])))
        size = shape[dim]
        chain_length = min(size, chain_length)
        n_chunks = math.ceil(size / chain_length)
        chain_length += self.chain_offset
        result_shape = list(shape)
        chunk_shape = result_shape.copy()
        result_shape[dim] = chain_length * n_chunks
        chunk_shape[dim] = n_chunks
        result = torch.zeros(result_shape, dtype=self.noise_dtype, device=device)
        adds, muls = result.clone(), result.clone()
        if self.seed_noise_sampler is not None:
            orig_noise = self.seed_noise_sampler(*args)[
                tuple(slice(None, sz) for sz in chunk_shape)
            ].to(result)
            if flatten:
                orig_noise = orig_noise.flatten(start_dim=dim)
            orig_noise = normalize_to_scale(
                orig_noise[tuple(slice(None, sz) for sz in chunk_shape)],
                1e-06,
                1.0,
                dim=tuple(range(1, len(chunk_shape))),
            )
        else:
            orig_noise = self.rand_like(
                fun=torch.rand,
                shape=chunk_shape,
                dtype=result.dtype,
            )
        noise = orig_noise * (rmaxsubmin + 1) + rmin
        # Derp.
        noise = torch.where(noise == 0, noise.max() / noise.numel(), noise)
        if self.seed_mode != "default":
            noise = torch.where(
                (noise % 2.0) < 1
                if self.seed_mode == "force_odd"
                else (noise % 2.0) >= 1,
                noise + 1,
                noise,
            )
        if noise.device != self.device:
            noise = tensor_to(noise, self.device)
        slice_0 = self._get_iter_slices(result.ndim, dim, 0, chain_length)
        for chainidx in range(chain_length):
            if chainidx == 0:
                muls[slice_0] = 1.0
                result[slice_0] = noise
                continue
            slice_curr = self._get_iter_slices(result.ndim, dim, chainidx, chain_length)
            slice_prev = self._get_iter_slices(
                result.ndim,
                dim,
                chainidx - 1,
                chain_length,
            )
            prev = result[slice_prev]
            prev_trunc = utils.trunc_decimals(prev, 2)
            need_reset = (
                ((prev_trunc >= 1.0) & (prev_trunc < 1.001))
                | (prev_trunc.abs() < 0.001)
                if self.break_loops
                else False
            )
            prev_evens = prev % 2 < 1.0
            prev_adds, prev_muls = adds[slice_prev], muls[slice_prev]
            muls_next = (
                torch.where(
                    prev_evens,
                    prev_muls if emul == 1 else prev_muls * emul,
                    prev_muls if omul == 1 else prev_muls * omul,
                )
                if emul != 1 or omul != 1
                else prev_muls
            )
            muls[slice_curr] = (
                torch.where(need_reset, 1.0, muls_next)
                if need_reset is not False
                else muls_next
            )
            curr_muls = muls[slice_curr]
            prev_adds_scaled = prev_adds * curr_muls
            prev_sign = prev.sign() if keepsign else 1.0
            adds_next = (
                torch.where(
                    prev_evens,
                    prev_adds_scaled
                    if eadd == 0
                    else prev_adds_scaled + eadd * prev_sign,
                    prev_adds_scaled
                    if oadd == 0
                    else prev_adds_scaled + oadd * prev_sign,
                )
                if eadd != 0 or oadd != 0
                else prev_adds_scaled
            )
            adds[slice_curr] = (
                torch.where(need_reset, 0.0, adds_next)
                if need_reset is not False
                else adds_next
            )
            curr_adds = adds[slice_curr]
            result_next = utils.maybe_apply(
                (noise * curr_muls).add_(curr_adds),
                intmode,
                torch.trunc,
            )
            result[slice_curr] = (
                torch.where(need_reset, noise, result_next)
                if need_reset is not False
                else result_next
            )
        output_slice = tuple(
            slice(None, sz) for sz in (shape if flatten else out_shape)
        )
        return self._iteration_output(
            *args,
            result_chains=result,
            orig_noise=orig_noise,
            noise=noise,
            raw_adds=adds,
            muls=muls,
            chain_length=chain_length,
            dim=dim,
            output_shape=out_shape,
            output_slice=output_slice,
            dtype=dtype,
        )

    def _trim_chain_offset(
        self,
        t: torch.Tensor,
        dim: int,
        chain_length: int,
    ) -> torch.Tensor:
        co = self.chain_offset
        if co < 1:
            return t
        chunks = t.split(chain_length, dim)
        slices = tuple(
            slice(None) if i != dim else slice(co, None) for i in range(t.ndim)
        )
        return torch.cat(
            tuple(chunk[slices] for chunk in chunks),
            dim=dim,
        )

    def _iteration_output(
        self,
        *args,
        result_chains: torch.Tensor,
        orig_noise: torch.Tensor,
        noise: torch.Tensor,
        raw_adds: torch.Tensor,
        muls: torch.Tensor,
        chain_length: int,
        dim: int,
        output_shape: Sequence,
        output_slice: Sequence,
        dtype: str | torch.dtype,
    ) -> torch.Tensor:
        omode = self.output_mode
        quantile = self.quantile
        noise_exp = noise.repeat_interleave(chain_length, dim)
        nadds = raw_adds.div_(noise_exp)
        ratios = result_chains / noise_exp
        if omode in {"values", "ratios", "seed_x_ratios", "noise_x_ratios"}:
            out1 = ratios
        elif omode in {"mults", "seed_x_mults", "noise_x_mults"}:
            out1 = muls
        elif omode in {"adds", "seed_x_adds", "noise_x_adds"}:
            out1 = nadds
        else:
            raise ValueError("Bad output mode")
        out1 = self._trim_chain_offset(out1, dim=dim, chain_length=chain_length)
        if quantile not in {0, 1}:
            out1 = utils.quantile_normalize(
                out1,
                quantile=quantile,
                dim=0,
                strategy=self.quantile_strategy,
            )
        out1 = out1[output_slice].reshape(output_shape).to(dtype=dtype)
        if omode in {"ratios", "mults", "adds"}:
            return out1
        if omode in {"values", "seed_x_ratios", "seed_x_mults", "seed_x_adds"}:
            out2 = orig_noise.repeat_interleave(chain_length - self.chain_offset, dim)
        elif omode in {"noise_x_ratios", "noise_x_mults", "noise_x_adds"}:
            out2 = (
                self.rand_like(dtype=out1.dtype)
                if self.mix_noise_sampler is None
                else self.mix_noise_sampler(*args)
            )
        out2 = out2[output_slice].reshape(output_shape).to(dtype=dtype)
        return out2 * out1

    def generate(self, *args):
        out_dims = len(self.shape)
        dims = tuple(dim if dim >= 0 else out_dims + dim for dim in self.dims)
        n_dims, n_chainlens = len(dims), len(self.chain_length)
        if not all(0 <= d < out_dims for d in dims):
            raise ValueError("Dimension out of range")
        dtype, device = self.dtype, self.device
        result = torch.zeros(self.shape, dtype=dtype, device=device)
        it_scale = 1.0 / self.iterations
        for iteration in range(self.iterations):
            if iteration > 0 and (iteration % 25) == 0:
                # It's soooo slow!
                throw_exception_if_processing_interrupted()
            temp = self._generate_iteration(
                *args,
                dim=dims[iteration % n_dims],
                chain_length=self.chain_length[iteration % n_chainlens],
                flatten=self.flatten,
            ).mul_(
                it_scale
                * (-1 if self.iteration_sign_flipping and (iteration & 1) == 1 else 1),
            )
            result += temp
        if self.adjust_scale:
            result = normalize_to_scale(
                result,
                -1.0,
                1.0,
                dim=tuple(range(1 if result.ndim < 4 else 2, result.ndim)),
            )
        return result
