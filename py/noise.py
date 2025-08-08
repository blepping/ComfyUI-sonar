from __future__ import annotations

import abc
import math
import random
from functools import partial
from typing import Callable

import comfy
import torch
import yaml
from comfy.k_diffusion import sampling
from torch import Tensor

from . import external, utils
from .noise_generation import *
from .sonar import SonarGuidanceMixin
from .utils import (
    RNGStates,
    crop_samples,
    fallback,
    pattern_break,
    quantile_normalize,
    scale_noise,
)

# ruff: noqa: ANN002, ANN003, FBT001


class CustomNoiseItemBase(abc.ABC):
    def __init__(self, factor, *, yaml_parameters=None, **kwargs):
        if yaml_parameters:
            extra_params = yaml.safe_load(yaml_parameters)
            if extra_params is None:
                pass
            elif not isinstance(extra_params, dict):
                raise ValueError(
                    "CustomNoiseItem: yaml_parameters must either be null or an object",
                )
            else:
                kwargs["ns_kwargs"] = extra_params
        self.factor = factor
        self.keys = set(kwargs.keys())
        for k, v in kwargs.items():
            do_clone = k in {
                "custom_noise",
                "custom_noise_opt",
                "noise",
                "noise_opt",
                "sonar_custom_noise",
                "sonar_custom_noise_opt",
            } and hasattr(v, "clone")
            setattr(self, k, v.clone() if do_clone else v)

    def clone_key(self, k):
        return getattr(self, k)

    def clone(self):
        return self.__class__(self.factor, **{k: self.clone_key(k) for k in self.keys})

    def set_factor(self, factor):
        self.factor = factor
        return self

    def get_normalize(self, k, default=None):
        val = getattr(self, k, None)
        return default if val is None else val

    @abc.abstractmethod
    def make_noise_sampler(
        self,
        x: Tensor,
        sigma_min=None,
        sigma_max=None,
        seed=None,
        cpu=True,
        normalized=True,
        **kwargs,
    ):
        raise NotImplementedError


class CustomNoiseItem(CustomNoiseItemBase):
    def __init__(self, factor, **kwargs):
        super().__init__(factor, **kwargs)
        if getattr(self, "noise_type", None) is None:
            raise ValueError("Noise type required!")

    @torch.no_grad()
    def make_noise_sampler(
        self,
        x: Tensor,
        sigma_min=None,
        sigma_max=None,
        seed=None,
        cpu=True,
        normalized=True,
        **kwargs,
    ):
        ns_kwargs = getattr(self, "ns_kwargs", {}).copy()
        override_sigma, override_sigma_next, override_sigma_min, override_sigma_max = (
            ns_kwargs.pop(k, None)
            for k in (
                "override_sigma",
                "override_sigma_next",
                "override_sigma_min",
                "override_sigma_max",
            )
        )
        ns = get_noise_sampler(
            self.noise_type,
            x,
            fallback(override_sigma_min, sigma_min),
            fallback(override_sigma_max, sigma_max),
            seed=ns_kwargs.pop("seed", seed),
            cpu=ns_kwargs.pop("cpu", cpu),
            factor=self.factor,
            normalized=ns_kwargs.pop(
                "normalized",
                self.get_normalize("normalize", normalized),
            ),
            **ns_kwargs,
            **kwargs,
        )
        if override_sigma is None and override_sigma_next is None:
            return ns

        def noise_sampler(sigma, sigma_next):
            return ns(
                fallback(override_sigma, sigma),
                fallback(override_sigma_next, sigma_next),
            )

        return noise_sampler


class CustomNoiseChain:
    def __init__(self, items=None):
        self.items = items if items is not None else []

    def clone(self):
        return CustomNoiseChain(
            [i.clone() for i in self.items],
        )

    def add(self, item):
        if item is None:
            raise ValueError("Attempt to add nil item")
        self.items.append(item)

    @property
    def factor(self):
        return sum(abs(i.factor) for i in self.items)

    def rescaled(self, scale=1.0):
        divisor = self.factor / scale
        divisor = divisor if divisor != 0 else 1.0
        result = self.clone()
        if divisor != 1:
            for i in result.items:
                i.set_factor(i.factor / divisor)
        return result

    @torch.no_grad()
    def make_noise_sampler(
        self,
        x: Tensor,
        sigma_min=None,
        sigma_max=None,
        seed=None,
        cpu=True,
        normalized=True,
    ) -> Callable:
        noise_samplers = tuple(
            i.make_noise_sampler(
                x,
                sigma_min,
                sigma_max,
                seed=seed,
                cpu=cpu,
                normalized=False,
            )
            for i in self.items
        )
        if not noise_samplers or not all(noise_samplers):
            raise ValueError("Failed to get noise sampler")
        factor = self.factor

        def noise_sampler(sigma, sigma_next):
            result = None
            for ns in noise_samplers:
                noise = ns(sigma, sigma_next)
                result = noise if result is None else result.add_(noise)
            return scale_noise(result, factor, normalized=normalized)

        return noise_sampler


class NoiseSampler:
    def __init__(
        self,
        x: Tensor,
        sigma_min: float | None = None,
        sigma_max: float | None = None,
        seed: int | None = None,
        cpu: bool = False,
        transform: Callable = lambda t: t,
        normalized=False,
        factor: float = 1.0,
        *,
        make_noise_sampler: Callable,
        **kwargs,
    ):
        self.factor = factor
        self.normalized = normalized
        self.transform = transform
        self.device = x.device
        self.dtype = x.dtype
        try:
            self.noise_sampler = make_noise_sampler(
                x,
                sigma_min=transform(torch.as_tensor(sigma_min))
                if sigma_min is not None
                else None,
                sigma_max=transform(torch.as_tensor(sigma_max))
                if sigma_max is not None
                else None,
                seed=seed,
                cpu=cpu,
                normalized=False,
                **kwargs,
            )
        except TypeError as _exc:
            print("GOT EXC", _exc)
            self.noise_sampler = make_noise_sampler(x)

    @classmethod
    def simple(cls, f):
        return lambda *args, **kwargs: cls(
            *args,
            **kwargs,
            make_noise_sampler=lambda x, *_args, **_kwargs: lambda _s, _sn: f(x),
        )

    @classmethod
    def wrap(cls, f):
        return lambda *args, **kwargs: cls(*args, **kwargs, make_noise_sampler=f)

    def __call__(self, *args, **kwargs):
        args = (
            self.transform(torch.as_tensor(s)) if s is not None else s for s in args
        )
        noise = self.noise_sampler(*args, **kwargs)
        noise = scale_noise(noise, self.factor, normalized=self.normalized)
        if hasattr(noise, "to"):
            noise = noise.to(dtype=self.dtype, device=self.device)
        return noise


class AdvancedNoiseBase(CustomNoiseItemBase):
    ns_factory_arg_keys = ()

    # This has to be done as a property for some reason.
    @property
    def ns_factory(self):
        raise NotImplementedError

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.ns_factory is None:
            raise NotImplementedError("ns_factory not implemented")
        noise_sampler_kwargs = {}
        for k in self.ns_factory_arg_keys:
            v = getattr(self, k, None)
            if v is not None:
                noise_sampler_kwargs[k] = v
        self.sampler_factory = NoiseSampler.wrap(
            partial(self.ns_factory, **noise_sampler_kwargs),
        )

    @torch.no_grad()
    def make_noise_sampler(self, *args, **kwargs):
        return self.sampler_factory(*args, factor=self.factor, **kwargs)


class AdvancedPyramidNoise(AdvancedNoiseBase):
    ns_factory_arg_keys = ("discount", "iterations", "upscale_mode")

    pyramid_variants_map = {  # noqa: RUF012
        "pyramid": PyramidNoiseGenerator,
        "pyramid_old": PyramidOldNoiseGenerator,
        "highres_pyramid": HighresPyramidNoiseGenerator,
    }

    @property
    def ns_factory(self):
        return self.pyramid_variants_map[self.variant]


class Advanced1fNoise(AdvancedNoiseBase):
    ns_factory_arg_keys = ("alpha", "hfac", "wfac", "k", "use_sqrt", "base_power")

    @property
    def ns_factory(self):
        return OneFNoiseGenerator


class AdvancedPowerLawNoise(AdvancedNoiseBase):
    ns_factory_arg_keys = ("alpha", "div_max_dims", "use_sign")

    @property
    def ns_factory(self):
        return PowerLawNoiseGenerator


class AdvancedDistroNoise(AdvancedNoiseBase):
    distro_params = DistroNoiseGenerator.build_params()
    ns_factory_arg_keys = (
        "distro",
        "quantile_norm",
        "quantile_norm_dim",
        "quantile_norm_flatten",
        "result_index",
        *distro_params.keys(),
    )

    @property
    def ns_factory(self):
        return DistroNoiseGenerator


class AdvancedCollatzNoise(AdvancedNoiseBase):
    ns_factory_arg_keys = (
        "adjust_scale",
        "iteration_sign_flipping",
        "chain_length",
        "iterations",
        "rmin",
        "rmax",
        "flatten",
        "dims",
        "output_mode",
        "noise_dtype",
        "quantile",
        "quantile_strategy",
        "integer_math",
        "add_preserves_sign",
        "even_multiplier",
        "even_addition",
        "odd_multiplier",
        "odd_addition",
        "chain_offset",
        "seed_mode",
        "break_loops",
    )

    @property
    def ns_factory(self):
        return CollatzNoiseGenerator

    def make_noise_sampler(self, x, *args, normalized=True, **kwargs):
        seed_ns = (
            self.seed_custom_noise.make_noise_sampler(
                x,
                *args,
                normalized=False,
                **kwargs,
            )
            if self.seed_custom_noise is not None
            else None
        )
        mix_ns = (
            self.mix_custom_noise.make_noise_sampler(
                x,
                *args,
                normalized=False,
                **kwargs,
            )
            if self.mix_custom_noise is not None
            and self.output_mode.startswith("noise_")
            else None
        )
        return super().make_noise_sampler(
            x,
            *args,
            normalized=normalized,
            seed_noise_sampler=seed_ns,
            mix_noise_sampler=mix_ns,
        )


class AdvancedWaveletNoise(AdvancedNoiseBase):
    ns_factory_arg_keys = (
        "octave_scale_mode",
        "octave_rescale_mode",
        "post_octave_rescale_mode",
        "initial_amplitude",
        "persistence",
        "octaves",
        "octave_height_factor",
        "octave_width_factor",
        "height_factor",
        "width_factor",
        "min_height",
        "min_width",
        "update_blend",
        "update_blend_function",
    )

    @property
    def ns_factory(self):
        return WaveletNoiseGenerator

    def make_noise_sampler(self, x, *args, normalized=True, **kwargs):
        if x.ndim < 4:
            raise ValueError("Can only handle 4+ dimensional latents")
        height, width = x.shape[-2:]
        result = super().make_noise_sampler(x, *args, normalized=normalized, **kwargs)
        wavelet_ng = result.noise_sampler
        max_height = (
            int(max(height, *(od.height for od in wavelet_ng.octave_data)))
            if wavelet_ng.octave_data
            else height
        )
        max_width = (
            int(max(width, *(od.width for od in wavelet_ng.octave_data)))
            if wavelet_ng.octave_data
            else width
        )
        internal_ns = (
            self.custom_noise.make_noise_sampler(
                x.new_zeros(*x.shape[:-2], max_height, max_width)
                if max_width != width or max_height != height
                else x,
                *args,
                normalized=self.normalize_noise,
                **kwargs,
            )
            if self.custom_noise is not None
            else None
        )
        wavelet_ng.set_internal_noise_sampler(internal_ns)
        return result


class AdvancedVoronoiNoise(AdvancedNoiseBase):
    ns_factory_arg_keys = tuple(VoronoiNoiseGenerator.ng_params(no_super=True))

    @property
    def ns_factory(self):
        return VoronoiNoiseGenerator

    def clone_key(self, k):
        if k == "custom_noise" and self.custom_noise is not None:
            return self.custom_noise.clone()
        return super().clone_key(k)

    def make_noise_sampler(self, x, *args, normalized=True, **kwargs):
        if x.ndim != 4:
            raise ValueError("Can only handle 4+ dimensional latents")
        return super().make_noise_sampler(
            x,
            *args,
            normalized=normalized,
            noise_sampler_factory=self.custom_noise,
            **kwargs,
        )


class CompositeNoise(CustomNoiseItemBase):
    def __init__(
        self,
        factor,
        *,
        dst_noise,
        src_noise,
        normalize_dst,
        normalize_src,
        normalize_result,
        mask,
    ):
        super().__init__(
            factor,
            dst_noise=dst_noise.clone(),
            src_noise=src_noise.clone(),
            normalize_dst=normalize_dst,
            normalize_src=normalize_src,
            normalize_result=normalize_result,
            mask=mask.clone(),
        )

    def clone_key(self, k):
        if k in {"mask", "src_noise", "dst_noise"}:
            return getattr(self, k).clone()
        return super().clone_key(k)

    def make_noise_sampler(self, x, *args, normalized=True, **kwargs):
        normalize_src, normalize_dst, normalize_result = (
            self.get_normalize(f"normalize_{k}", normalized)
            for k in ("src", "dst", "result")
        )
        nsd = self.dst_noise.make_noise_sampler(
            x,
            *args,
            normalized=normalize_dst,
            **kwargs,
        )
        nss = self.src_noise.make_noise_sampler(
            x,
            *args,
            normalized=normalize_src,
            **kwargs,
        )
        mask = self.mask.to(x.device, copy=True)
        mask = torch.nn.functional.interpolate(
            mask.reshape((-1, 1, *mask.shape[-2:])),
            size=x.shape[-2:],
            mode="bilinear",
        )
        mask = comfy.utils.repeat_to_batch_size(mask, x.shape[0])
        imask = torch.ones_like(mask) - mask
        factor = self.factor

        def noise_sampler(s, sn):
            noise_dst = nsd(s, sn).mul_(imask)
            noise_src = nss(s, sn).mul_(mask)
            return scale_noise(
                noise_dst.add_(noise_src),
                factor,
                normalized=normalize_result,
            )

        return noise_sampler


class GuidedNoise(CustomNoiseItemBase):
    def __init__(
        self,
        factor,
        *,
        guidance_factor,
        ref_latent,
        method,
        normalize_noise,
        normalize_result,
        noise=None,
    ):
        super().__init__(
            factor,
            normalize_noise=normalize_noise,
            normalize_result=normalize_result,
            ref_latent=ref_latent.clone(),
            noise=noise.clone() if noise is not None else None,
            method=method,
            guidance_factor=guidance_factor,
        )

    def clone_key(self, k):
        if k == "noise" and self.noise is None:
            return None
        if k in {"noise", "ref_latent"}:
            return getattr(self, k).clone()
        return super().clone_key(k)

    def make_noise_sampler(self, x, *args, normalized=True, **kwargs):
        factor, guidance_factor = self.factor, self.guidance_factor
        normalize_noise, normalize_result = (
            self.get_normalize(f"normalize_{k}", normalized)
            for k in ("noise", "result")
        )
        if self.noise is None:
            ns = None
            x_zeros = torch.zeros_like(x)
        else:
            ns = self.noise.make_noise_sampler(
                x,
                *args,
                normalized=normalize_noise,
                **kwargs,
            )
        ref_latent = self.ref_latent.to(x, copy=True)
        if ref_latent.shape[-2:] != x.shape[-2:]:
            ref_latent = torch.nn.functional.interpolate(
                ref_latent,
                size=x.shape[-2:],
                mode="bicubic",
                align_corners=True,
            )
        if self.method == "linear":

            def noise_sampler(s, sn):
                return scale_noise(
                    SonarGuidanceMixin.guidance_linear(
                        x_zeros.clone() if ns is None else ns(s, sn),
                        ref_latent,
                        guidance_factor,
                        do_shift=ns is not None,
                    ),
                    factor,
                    normalized=normalize_result,
                )

        elif self.method == "euler":

            def noise_sampler(s, sn):
                return scale_noise(
                    SonarGuidanceMixin.guidance_euler(
                        s,
                        sn,
                        x_zeros if ns is None else ns(s, sn),
                        x,
                        ref_latent,
                        guidance_factor,
                        do_shift=ns is not None,
                    ),
                    factor,
                    normalized=normalize_result,
                )

        else:
            raise ValueError("Bad method")

        return noise_sampler


class ScheduledNoise(CustomNoiseItemBase):
    def __init__(
        self,
        factor,
        *,
        noise,
        start_sigma,
        end_sigma,
        normalize,
        fallback_noise=None,
    ):
        super().__init__(
            factor,
            noise=noise.clone(),
            start_sigma=start_sigma,
            end_sigma=end_sigma,
            normalize=normalize,
            fallback_noise=None if fallback_noise is None else fallback_noise.clone(),
        )

    def clone_key(self, k):
        if k == "noise":
            return self.noise.clone()
        if k == "fallback_noise":
            return None if self.fallback_noise is None else self.fallback_noise.clone()
        return super().clone_key(k)

    def make_noise_sampler(self, x, *args, normalized=True, **kwargs):
        factor = self.factor
        start_sigma, end_sigma = self.start_sigma, self.end_sigma
        normalize = self.get_normalize("normalize", normalized)
        ns = self.noise.make_noise_sampler(x, *args, normalized=False, **kwargs)
        if self.fallback_noise:
            nsa = self.fallback_noise.make_noise_sampler(
                x,
                *args,
                normalized=False,
                **kwargs,
            )
        else:

            def nsa(_s, _sn):
                return torch.zeros_like(x)

        def noise_sampler(s, sn):
            if s is None or sn is None:
                raise ValueError(
                    "ScheduledNoise requires sigma, sigma_next to be passed",
                )
            noise = (ns if end_sigma <= s <= start_sigma else nsa)(s, sn)
            return scale_noise(noise, factor, normalized=normalize)

        return noise_sampler


class RepeatedNoise(CustomNoiseItemBase):
    def __init__(self, factor, *, noise, **kwargs):
        super().__init__(factor, noise=noise.clone(), **kwargs)

    def clone_key(self, k):
        if k == "noise":
            return self.noise.clone()
        return super().clone_key(k)

    def make_noise_sampler(self, x, *args, normalized=True, **kwargs):
        factor = self.factor
        repeat_length, max_recycle = self.repeat_length, self.max_recycle
        permute = self.permute
        normalize = self.get_normalize("normalize", normalized)
        ns = self.noise.make_noise_sampler(x, *args, normalized=False, **kwargs)
        noise_items = []
        permute_options = 2
        u32_max = 0xFFFF_FFFF
        seed = kwargs.get("seed")
        if seed is None:
            seed = torch.randint(
                -u32_max,
                u32_max,
                (1,),
                device="cpu",
                dtype=torch.int64,
            ).item()
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        last_idx = -1

        def noise_sampler(s, sn):
            nonlocal last_idx
            rands = torch.randint(
                u32_max,
                (4,),
                generator=gen,
                dtype=torch.uint32,
            ).tolist()
            skip_permute = permute == "disabled"
            if len(noise_items) < repeat_length:
                idx = len(noise_items)
                noise = ns(s, sn)
                noise_items.append((1, noise))
                skip_permute = permute != "always"
            else:
                idx = rands[0] % repeat_length
                if idx == last_idx:
                    idx = (idx + 1) % repeat_length
                count, noise = noise_items[idx]
                if count >= max_recycle:
                    noise = ns(s, sn)
                    noise_items[idx] = (1, noise)
                    skip_permute = permute != "always"
                else:
                    noise_items[idx] = (count + 1, noise)

            last_idx = idx
            if skip_permute:
                return noise.clone()
            noise_dims = len(noise.shape)
            rep_mode = rands[1] % permute_options
            if rep_mode == 0:
                if rands[2] <= u32_max // 5:
                    # 10% of the time we return the original tensor instead of flipping or inverting
                    noise = noise.clone()
                    if rands[2] & 1 == 1:
                        noise *= -1.0
                else:
                    dims = tuple({rands[2] % noise_dims, rands[3] % noise_dims})
                    noise = torch.flip(noise, dims)
            elif rep_mode == 1:
                dim = rands[2] % noise_dims
                count = rands[3] % noise.shape[dim]
                noise = torch.roll(noise, count, dims=(dim,)).clone()
            return scale_noise(noise, factor, normalized=normalize)

        return noise_sampler


# Modulated noise functions copied from https://github.com/Clybius/ComfyUI-Extra-Samplers
# They probably don't work correctly for normal sampling.
class ModulatedNoise(CustomNoiseItemBase):
    MODULATION_DIMS = (-3, (-2, -1), (-3, -2, -1))

    def __init__(
        self,
        factor,
        *,
        noise,
        normalize_result,
        normalize_noise,
        normalize_ref,
        modulation_type="none",
        modulation_strength=2.0,
        modulation_dims=3,
        ref_latent_opt=None,
    ):
        super().__init__(
            factor,
            normalize_result=normalize_result,
            normalize_noise=normalize_noise,
            normalize_ref=normalize_ref,
            noise=noise.clone(),
            modulation_dims=modulation_dims,
            modulation_type=modulation_type,
            modulation_strength=modulation_strength,
            ref_latent_opt=None if ref_latent_opt is None else ref_latent_opt.clone(),
        )
        mt = self.modulation_type
        if mt == "intensity":
            self.modulation_function = self.intensity_based_multiplicative_noise
        elif mt == "frequency":
            self.modulation_function = self.frequency_based_noise
        elif mt == "spectral_signum":
            self.modulation_function = self.spectral_modulate_noise
        else:
            self.modulation_function = None

    def clone_key(self, k):
        if k == "ref_latent_opt":
            return None if self.ref_latent_opt is None else self.ref_latent_opt.clone()
        if k == "noise":
            return self.noise.clone()
        return super().clone_key(k)

    def make_noise_sampler(self, x, *args, normalized=True, **kwargs):
        factor, strength = self.factor, self.modulation_strength
        normalize_noise, normalize_result, normalize_ref = (
            self.get_normalize(f"normalize_{k}", normalized)
            for k in ("noise", "result", "ref")
        )

        dims = self.MODULATION_DIMS[self.modulation_dims - 1]
        if not self.modulation_function:
            ns = self.noise.make_noise_sampler(
                x,
                *args,
                normalized=normalize_result or normalize_noise,
                **kwargs,
            )

            def noise_sampler(s, sn):
                return scale_noise(ns(s, sn), factor, normalized=False)

            return noise_sampler

        ns = self.noise.make_noise_sampler(
            x,
            *args,
            normalized=normalize_noise,
            **kwargs,
        )

        ref_latent = (
            None
            if self.ref_latent_opt is None
            else self.ref_latent_opt.to(x, copy=True)
        )
        modulation_function = self.modulation_function

        def noise_sampler(s, sn):
            _sigma_down, sigma_up = sampling.get_ancestral_step(s, sn, eta=1.0)
            noise = modulation_function(
                scale_noise(
                    x if ref_latent is None else ref_latent,
                    normalized=normalize_ref,
                ),
                ns(s, sn),
                1.0,  # s_noise
                sigma_up,
                strength,
                dims,
            )
            return scale_noise(noise, factor, normalized=normalize_result)

        return noise_sampler

    @staticmethod
    def intensity_based_multiplicative_noise(
        x,
        noise,
        s_noise,
        sigma_up,
        intensity,
        dims,
    ) -> torch.Tensor:
        """Scales noise based on the intensities of the input tensor."""
        std = torch.std(
            x - x.mean(),
            dim=dims,
            keepdim=True,
        )  # Average across channels to get intensity
        scaling = (
            1 / (std * abs(intensity) + 1.0)
        )  # Scale std by intensity, as not doing this leads to more noise being left over, leading to crusty/preceivably extremely oversharpened images
        additive_noise = noise * s_noise * sigma_up
        scaled_noise = noise * s_noise * sigma_up * scaling + additive_noise

        noise_norm = torch.norm(additive_noise)
        scaled_noise_norm = torch.norm(scaled_noise)
        scaled_noise *= noise_norm / scaled_noise_norm  # Scale to normal noise strength
        return scaled_noise * intensity + additive_noise * (1 - intensity)

    @staticmethod
    def frequency_based_noise(
        z_k,
        noise,
        s_noise,
        sigma_up,
        intensity,
        channels,
    ) -> torch.Tensor:
        """Scales the high-frequency components of the noise based on the given intensity."""
        additive_noise = noise * s_noise * sigma_up

        std = torch.std(
            z_k - z_k.mean(),
            dim=channels,
            keepdim=True,
        )  # Average across channels to get intensity
        scaling = 1 / (std * abs(intensity) + 1.0)
        # Perform Fast Fourier Transform (FFT)
        z_k_freq = torch.fft.fft2(scaling * additive_noise + additive_noise)

        # Get the magnitudes of the frequency components
        magnitudes = torch.abs(z_k_freq)

        # Create a high-pass filter (emphasize high frequencies)
        h, w = z_k.shape[-2:]
        b = abs(
            intensity,
        )  # Controls the emphasis of the high pass (higher frequencies are boosted)
        high_pass_filter = 1 - torch.exp(
            -((torch.arange(h)[:, None] / h) ** 2 + (torch.arange(w)[None, :] / w) ** 2)
            * b**2,
        )
        high_pass_filter = high_pass_filter.to(z_k.device)

        # Apply the filter to the magnitudes
        magnitudes_scaled = magnitudes * (1 + high_pass_filter)

        # Reconstruct the complex tensor with scaled magnitudes
        z_k_freq_scaled = magnitudes_scaled * torch.exp(1j * torch.angle(z_k_freq))

        # Perform Inverse Fast Fourier Transform (IFFT)
        z_k_scaled = torch.fft.ifft2(z_k_freq_scaled)

        # Return the real part of the result
        z_k_scaled = torch.real(z_k_scaled)

        noise_norm = torch.norm(additive_noise)
        scaled_noise_norm = torch.norm(z_k_scaled)

        z_k_scaled *= noise_norm / scaled_noise_norm  # Scale to normal noise strength

        return z_k_scaled * intensity + additive_noise * (1 - intensity)

    @staticmethod
    def spectral_modulate_noise(
        _unused,
        noise,
        s_noise,
        sigma_up,
        intensity,
        channels,
        spectral_mod_percentile=5.0,
    ) -> torch.Tensor:  # Modified for soft quantile adjustment using a novel:tm::c::r: method titled linalg.
        additive_noise = noise * s_noise * sigma_up
        # Convert image to Fourier domain
        fourier = torch.fft.fftn(
            additive_noise,
            dim=channels,
        )  # Apply FFT along Height and Width dimensions

        log_amp = torch.log(torch.sqrt(fourier.real**2 + fourier.imag**2))

        quantile_low = (
            torch.quantile(
                log_amp.abs().flatten(1),
                spectral_mod_percentile * 0.01,
                dim=1,
            )
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(log_amp.shape)
        )

        quantile_high = (
            torch.quantile(
                log_amp.abs().flatten(1),
                1 - (spectral_mod_percentile * 0.01),
                dim=1,
            )
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(log_amp.shape)
        )

        quantile_max = (
            torch.quantile(log_amp.abs().flatten(1), 1, dim=1)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(log_amp.shape)
        )

        # Decrease high-frequency components
        mask_high = log_amp > quantile_high  # If we're larger than 95th percentile

        additive_mult_high = torch.where(
            mask_high,
            1
            - ((log_amp - quantile_high) / (quantile_max - quantile_high)).clamp_(
                max=0.5,
            ),  # (1) - (0-1), where 0 is 95th %ile and 1 is 100%ile
            torch.tensor(1.0),
        )

        # Increase low-frequency components
        mask_low = log_amp < quantile_low
        additive_mult_low = torch.where(
            mask_low,
            1
            + (1 - (log_amp / quantile_low)).clamp_(
                max=0.5,
            ),  # (1) + (0-1), where 0 is 5th %ile and 1 is 0%ile
            torch.tensor(1.0),
        )

        mask_mult = (additive_mult_low * additive_mult_high) ** intensity
        filtered_fourier = fourier * mask_mult

        # Inverse transform back to spatial domain
        inverse_transformed = torch.fft.ifftn(
            filtered_fourier,
            dim=channels,
        )  # Apply IFFT along Height and Width dimensions

        return inverse_transformed.real.to(additive_noise.device)


class RandomNoise(CustomNoiseItemBase):
    def __init__(self, factor, *, noise, mix_count, normalize):
        if len(noise.items) == 0:
            raise ValueError("RandomNoise requires ta least one noise item")
        super().__init__(
            factor,
            noise=noise.clone(),
            mix_count=mix_count,
            normalize=normalize,
        )

    def clone_key(self, k):
        if k == "noise":
            return self.noise.clone()
        return super().clone_key(k)

    def make_noise_sampler(self, x, *args, normalized=True, **kwargs):
        factor = self.factor
        noise_samplers = tuple(
            ni.make_noise_sampler(x, *args, normalized=False, **kwargs)
            for ni in self.noise.items
        )
        num_samplers = len(noise_samplers)
        mix_count = min(self.mix_count, num_samplers)
        normalize = self.get_normalize("normalize", normalized or mix_count > 1)

        if mix_count == 1:

            def noise_sampler(s, sn):
                idx = torch.randint(num_samplers, (1,)).item()
                return scale_noise(
                    noise_samplers[idx](s, sn),
                    factor,
                    normalized=normalize,
                )

            return noise_sampler

        def noise_sampler(s, sn):
            seen = set()
            while len(seen) < mix_count:
                idx = torch.randint(num_samplers, (1,)).item()
                if idx in seen:
                    continue
                seen.add(idx)
            idxs = tuple(seen)
            noise = noise_samplers[idxs[0]](s, sn)
            for i in idxs[1:]:
                noise += noise_samplers[i](s, sn)
            return scale_noise(noise, factor, normalized=normalize)

        return noise_sampler


class ChannelNoise(CustomNoiseItemBase):
    def __init__(self, factor, *, noise, insufficient_channels_mode, normalize):
        if len(noise.items) == 0:
            raise ValueError("ChannelNoise requires at least one noise item")
        if insufficient_channels_mode not in {"wrap", "repeat", "zero"}:
            raise ValueError("Bad insufficient_channels_mode")
        super().__init__(
            factor,
            noise=noise.clone(),
            insufficient_channels_mode=insufficient_channels_mode,
            normalize=normalize,
        )

    def clone_key(self, k):
        if k == "noise":
            return self.noise.clone()
        return super().clone_key(k)

    def make_noise_sampler(self, x, *args, normalized=True, **kwargs):
        factor = self.factor
        icmode = self.insufficient_channels_mode
        c = x.shape[1]
        noise_items = self.noise.items[:c]
        num_samplers = len(noise_items)

        def make_zero_noise_sampler(x, *_args, **_kwargs):
            return lambda *_args, **_kwargs: torch.zeros_like(x)

        make_zero_noise_sampler.make_noise_sampler = make_zero_noise_sampler

        while len(noise_items) < c:
            if icmode == "wrap":
                item = noise_items[len(noise_items) % num_samplers]
            elif icmode == "repeat":
                item = noise_items[num_samplers - 1]
            elif icmode == "zero":
                item = make_zero_noise_sampler
            else:
                raise ValueError("Bad insufficient_channels_mode")
            noise_items.append(item)
        noise_samplers = tuple(
            ni.make_noise_sampler(
                x[:, ni_channel : ni_channel + 1, ...],
                *args,
                normalized=False,
                **kwargs,
            )
            for ni_channel, ni in enumerate(noise_items)
        )
        normalize = self.get_normalize("normalize", normalized)

        def noise_sampler(s, sn):
            noise = torch.cat(tuple(ns(s, sn) for ns in noise_samplers), dim=1)
            return scale_noise(noise, factor, normalized=normalize)

        return noise_sampler


class RippleFilteredNoise(CustomNoiseItemBase):
    def __init__(
        self,
        factor,
        *,
        noise,
        **kwargs,
    ):
        super().__init__(factor, noise=noise.clone(), **kwargs)

    def clone_key(self, k):
        if k == "noise":
            return self.noise.clone()
        return super().clone_key(k)

    def make_noise_sampler(self, x, *args, normalized=True, **kwargs):
        factor = self.factor
        dim = self.dim
        if dim < 0:
            dim = x.ndim + dim
        if dim < 0 or dim >= x.ndim:
            raise ValueError("Dimension out of range")
        dim_els = math.prod(x.shape[dim:]) if self.flatten else x.shape[dim]
        mode_fun = torch.sin if self.mode.startswith("sin") else torch.cos
        follow_sign = self.mode.endswith("_copysign")
        scaler_shape = [1] * x.ndim
        if self.flatten:
            scaler_shape[dim:] = x.shape[dim:]
        else:
            scaler_shape[dim] = x.shape[dim]
        scaler = mode_fun(
            torch.linspace(
                self.offset,
                self.offset + math.pi * self.period,
                steps=dim_els,
                dtype=x.dtype,
                device=x.device,
            ),
        )
        scaler = (
            1.0
            + torch.where(
                scaler < 0,
                scaler * self.amplitude_low,
                scaler * self.amplitude_high,
            )
        ).reshape(scaler_shape)
        ns = self.noise.make_noise_sampler(
            x,
            *args,
            normalized=self.normalize_noise,
            **kwargs,
        )
        roll = self.roll
        normalize = self.get_normalize("normalize", normalized)
        counter = 0

        def noise_sampler(s, sn):
            nonlocal counter
            noise = ns(s, sn)
            to_roll = int(roll * counter)
            counter += 1
            scaler_curr = scaler.roll(to_roll, dims=dim)
            result = scale_noise(noise, factor, normalized=normalize).mul_(
                scaler_curr,
            )
            return result.copysign(1.0 - scaler_curr) if follow_sign else result

        return noise_sampler


class NormalizeToScaleNoise(CustomNoiseItemBase):
    def __init__(
        self,
        factor,
        *,
        noise,
        min_negative_value: float,
        max_negative_value: float,
        min_positive_value: float,
        max_positive_value: float,
        mode: str,
        **kwargs,
    ):
        if mode == "simple":
            if min_negative_value >= max_positive_value:
                raise ValueError(
                    "In simple mode, min_negative_value can't be greater or equal to max_positive_value",
                )
        elif mode == "advanced":
            if min_negative_value >= max_negative_value:
                raise ValueError(
                    "In advanced mode, min_negative_value can't be greater or equal to max_negative value",
                )
            if min_positive_value >= max_positive_value:
                raise ValueError(
                    "In advanced mode, min_positive_value can't be greater or equal to max_positive value",
                )
        else:
            raise ValueError("Bad mode")
        super().__init__(
            factor,
            noise=noise.clone(),
            min_negative_value=min_negative_value,
            max_negative_value=max_negative_value,
            min_positive_value=min_positive_value,
            max_positive_value=max_positive_value,
            mode=mode,
            **kwargs,
        )

    def clone_key(self, k):
        if k == "noise":
            return self.noise.clone()
        return super().clone_key(k)

    def make_noise_sampler(self, x, *args, normalized=True, **kwargs):
        std_dims, std_multiplier = self.std_dims, self.std_multiplier
        mean_dims, mean_multiplier = self.mean_dims, self.mean_multiplier
        factor = self.factor
        mode = self.mode
        if mode == "simple":
            noise_filter = partial(
                utils.normalize_to_scale,
                target_min=self.min_negative_value,
                target_max=self.max_positive_value,
                dim=self.dims,
            )
        else:
            noise_filter = partial(
                utils.normalize_to_scale_adv,
                min_pos=self.min_positive_value,
                max_pos=self.max_positive_value,
                min_neg=self.min_negative_value,
                max_neg=self.max_negative_value,
                dim=(),
            )

        ns = self.noise.make_noise_sampler(
            x,
            *args,
            normalized=self.normalize_noise,
            **kwargs,
        )
        normalize = self.get_normalize("normalize", normalized)

        def noise_sampler(s, sn):
            noise = ns(s, sn)
            if mode == "simple" or noise.ndim < 2 or not self.dims:
                noise = noise_filter(noise)
            else:
                for bidx in range(noise.shape[0]):
                    noise[bidx] = noise_filter(noise[bidx])
            if mean_multiplier != 0:
                noise -= noise.mean(dim=mean_dims, keepdim=True).mul_(mean_multiplier)
            if std_multiplier != 0:
                noise_std = (
                    noise.std(dim=std_dims, keepdim=True)
                    .sub_(1.0)
                    .mul_(std_multiplier)
                    .add_(1.0)
                )
                noise /= torch.where(noise_std == 0, 1e-07, noise_std)
            return scale_noise(noise, factor, normalized=normalize)

        return noise_sampler


class BlendedNoise(CustomNoiseItemBase):
    def __init__(
        self,
        factor,
        *,
        normalize,
        blend_function,
        custom_noise_1=None,
        custom_noise_2=None,
        custom_noise_mask=None,
        noise_2_percent=0.5,
    ):
        if custom_noise_1 is None and (
            custom_noise_mask is not None or noise_2_percent != 1
        ):
            raise ValueError(
                "When custom_noise_1 is not attached noise_2_percent must be set to 1",
            )
        if custom_noise_2 is None and (
            custom_noise_mask is not None or noise_2_percent != 0
        ):
            raise ValueError(
                "When custom_noise_2 is not attached noise_2_percent must be set to 0",
            )
        if (
            custom_noise_mask is None
            and noise_2_percent == 1
            and custom_noise_1 is None
        ):
            custom_noise_1, custom_noise_2 = custom_noise_2, None
            noise_2_percent = 0.0
        super().__init__(
            factor,
            noise_2_percent=noise_2_percent,
            blend_function=blend_function,
            custom_noise_1=custom_noise_1.clone(),
            custom_noise_2=None if custom_noise_2 is None else custom_noise_2.clone(),
            custom_noise_mask=None
            if custom_noise_mask is None
            else custom_noise_mask.clone(),
            normalize=normalize,
        )

    def clone_key(self, k):
        if k == "custom_noise_1":
            return self.custom_noise_1.clone()
        if k == "custom_noise_2":
            return None if self.custom_noise_2 is None else self.custom_noise_2.clone()
        if k == "custom_noise_mask":
            return (
                None
                if self.custom_noise_mask is None
                else self.custom_noise_mask.clone()
            )
        return super().clone_key(k)

    def make_noise_sampler(self, x, *args, normalized=True, **kwargs):
        factor = self.factor
        normalize = self.get_normalize("normalize", normalized)
        blend_function = self.blend_function
        n2_blend = self.noise_2_percent

        ns_1 = self.custom_noise_1.make_noise_sampler(
            x,
            *args,
            normalized=False,
            **kwargs,
        )
        ns_2 = (
            None
            if self.custom_noise_2 is None
            else self.custom_noise_2.make_noise_sampler(
                x,
                *args,
                normalized=False,
                **kwargs,
            )
        )
        ns_mask = (
            None
            if self.custom_noise_mask is None
            else self.custom_noise_mask.make_noise_sampler(
                x,
                *args,
                normalized=False,
                **kwargs,
            )
        )
        n2_blend_tensor = x.new_full((1,), n2_blend) if ns_mask is None else None

        def noise_sampler(s, sn):
            nonlocal n2_blend_tensor
            noise_1 = ns_1(s, sn)
            noise_2 = None if ns_2 is None else ns_2(s, sn)
            if ns_mask is not None:
                n2_blend_tensor = (
                    utils.normalize_to_scale(ns_mask(s, sn), 0.0, 1.0) + n2_blend
                ).clamp_(0.0, 1.0)
            noise = (
                noise_1
                if noise_2 is None
                else blend_function(noise_1, noise_2, n2_blend_tensor)
            )
            return scale_noise(noise, factor, normalized=normalize)

        return noise_sampler


class ResizedNoise(CustomNoiseItemBase):
    def __init__(
        self,
        factor,
        *,
        custom_noise,
        **kwargs,
    ):
        if len(custom_noise.items) == 0:
            raise ValueError("ResizedNoise requires at least one noise item")
        super().__init__(factor, custom_noise=custom_noise.clone(), **kwargs)

    def clone_key(self, k):
        if k == "custom_noise":
            return self.custom_noise.clone()
        return super().clone_key(k)

    def make_noise_sampler(self, x, *args, normalized=True, **kwargs):
        if x.ndim < 3:
            raise ValueError("ResizedNoise can only handle 3+ dimensional latents")
        factor = self.factor
        normalize = self.get_normalize("normalize", normalized)
        spatial_compression = self.spatial_compression
        spatial_mode = self.spatial_mode
        width, height = self.width, self.height
        xh, xw = x.shape[-2:]
        if spatial_mode != "percentage":
            height //= spatial_compression
            width //= spatial_compression
        if spatial_mode == "absolute":
            nh, nw = int(height), int(width)
        elif spatial_mode == "relative":
            nh, nw = int(xh + height), int(xw + width)
        elif spatial_mode == "percentage":
            nh, nw = max(1, int(xh * height)), max(1, int(xw * width))
        else:
            raise ValueError("Bad spatial_mode")
        offsh = self.crop_offset_vertical // spatial_compression
        offsw = self.crop_offset_horizontal // spatial_compression
        if xh == nh and xw == nw:
            ns = self.custom_noise.make_noise_sampler(
                x,
                *args,
                normalized=normalize,
                **kwargs,
            )

            def noise_sampler(*args, **kwargs):
                return ns(*args, **kwargs).mul_(factor)

            return noise_sampler

        upscale_mode = self.upscale_mode
        downscale_mode = self.downscale_mode
        crop_mode = self.crop_mode
        x_all_bigger = xh >= nh and xw >= nw
        x_any_bigger = xh >= nh or xw >= nw
        if x_all_bigger:
            if self.initial_reference == "prefer_crop":
                x = crop_samples(
                    x,
                    nw,
                    nh,
                    mode=self.crop_mode,
                    offset_width=offsw,
                    offset_height=offsh,
                )
            else:
                x = utils.scale_samples(x, nw, nh, mode=self.downscale_mode)
            output = partial(
                utils.scale_samples,
                width=xw,
                height=xh,
                mode=upscale_mode,
            )
        else:
            x = utils.scale_samples(x, nw, nh, mode=self.upscale_mode)
            if x_any_bigger:
                output = partial(
                    utils.scale_samples,
                    width=xw,
                    height=xh,
                    mode=upscale_mode,
                )
            elif self.downscale_strategy == "scale":
                output = partial(
                    utils.scale_samples,
                    width=xw,
                    height=xh,
                    mode=downscale_mode,
                )
            else:
                output = partial(
                    crop_samples,
                    width=xw,
                    height=xh,
                    mode=crop_mode,
                    offset_width=offsw,
                    offset_height=offsh,
                )
        ns = self.custom_noise.make_noise_sampler(x, *args, normalized=False, **kwargs)
        del x

        def noise_sampler(*args, **kwargs):
            return output(
                scale_noise(ns(*args, **kwargs), factor, normalized=normalize),
            )

        return noise_sampler


class WaveletFilteredNoise(CustomNoiseItemBase):
    def clone_key(self, k):
        if k == "noise" and self.noise is not None:
            return self.noise.clone()
        if k == "noise_high" and self.noise_high is not None:
            return self.noise_high.clone()
        return super().clone_key(k)

    def make_noise_sampler(
        self,
        x,
        sigma_min,
        sigma_max,
        *args,
        normalized=True,
        **kwargs,
    ):
        factor = self.factor
        normalize = self.get_normalize("normalize", normalized)
        internal_ns = (
            self.noise.make_noise_sampler(
                x,
                *args,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                normalized=self.normalize_noise,
                **kwargs,
            )
            if self.noise is not None
            else None
        )
        internal_ns_high = (
            None
            if self.noise_high is None
            else self.noise_high.make_noise_sampler(
                x,
                *args,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                normalized=self.normalize_noise,
                **kwargs,
            )
        )
        ns_kwargs = getattr(self, "ns_kwargs", {}).copy()
        yl_blend_function = ns_kwargs.pop("yl_blend_function", torch.lerp)
        yh_blend_function = ns_kwargs.pop("yh_blend_function", torch.lerp)
        if isinstance(yl_blend_function, str):
            yl_blend_function = utils.BLENDING_MODES[yl_blend_function]
        if isinstance(yh_blend_function, str):
            yh_blend_function = utils.BLENDING_MODES[yh_blend_function]

        kwargs |= ns_kwargs
        ns = WaveletFilteredNoiseGenerator(
            x,
            *args,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            normalized=False,
            noise_sampler=internal_ns,
            noise_sampler_high=internal_ns_high,
            yl_blend_function=yl_blend_function,
            yh_blend_function=yh_blend_function,
            **kwargs,
        )

        def noise_sampler(sigma, sigma_next):
            return scale_noise(
                ns(sigma, sigma_next),
                factor,
                normalized=normalize,
            )

        return noise_sampler


class ScatternetFilteredNoise(CustomNoiseItemBase):
    def clone_key(self, k):
        if k == "noise" and self.noise is not None:
            return self.noise.clone()
        return super().clone_key(k)

    def make_noise_sampler(
        self,
        x,
        sigma_min,
        sigma_max,
        *args,
        normalized=True,
        **kwargs,
    ):
        if x.ndim != 4:
            raise ValueError("Currently can only handle 4 dimensional latents")
        factor = self.factor
        normalize = self.get_normalize("normalize", normalized)
        if self.noise is not None:
            if self.output_mode.endswith("_adjusted") and self.scatternet_order != 0:
                spatial_compensation = 2 ** abs(self.scatternet_order)
            else:
                spatial_compensation = 1
            internal_ns = self.noise.make_noise_sampler(
                x
                if spatial_compensation == 1
                else x.new_zeros(
                    *x.shape[:-2],
                    x.shape[-2] * spatial_compensation,
                    x.shape[-1] * spatial_compensation,
                ),
                *args,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                normalized=self.normalize_noise,
                **kwargs,
            )
        else:
            internal_ns = None
        ns_kwargs = getattr(self, "ns_kwargs", {}).copy()
        kwargs |= ns_kwargs
        ns = ScatternetFilteredNoiseGenerator(
            x,
            *args,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            normalized=False,
            noise_sampler=internal_ns,
            mode=self.padding_mode,
            use_symmetric_filter=self.use_symmetric_filter,
            magbias=self.magbias,
            output_offset=self.output_offset,
            output_mode=self.output_mode,
            scatternet_order=self.scatternet_order,
            per_channel_scatternet=self.per_channel_scatternet,
            **kwargs,
        )

        def noise_sampler(sigma, sigma_next):
            return scale_noise(
                ns(sigma, sigma_next),
                factor,
                normalized=normalize,
            )

        return noise_sampler


class LatentOperationFilteredNoise(CustomNoiseItemBase):
    def clone_key(self, k):
        if k == "noise" and self.noise is not None:
            return self.noise.clone()
        return super().clone_key(k)

    def make_noise_sampler(
        self,
        x,
        sigma_min,
        sigma_max,
        *args,
        normalized=True,
        **kwargs,
    ):
        factor = self.factor
        normalize = self.get_normalize("normalize", normalized)
        ns = self.noise.make_noise_sampler(
            x,
            *args,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            normalized=self.normalize_noise,
            **kwargs,
        )
        ops = self.operations

        def noise_sampler(sigma, sigma_next):
            noise = ns(sigma, sigma_next)
            for op in ops:
                noise = op(latent=noise, sigma=sigma)
            return scale_noise(noise, factor, normalized=normalize)

        return noise_sampler


class BlendFilterNoise(CustomNoiseItemBase):
    def __init__(self, factor, *, noise, **kwargs):
        if len(noise.items) == 0:
            raise ValueError("BlendFilterNoise requires at least one noise item")
        super().__init__(
            factor,
            noise=noise.clone(),
            **kwargs,
        )

    def clone_key(self, k):
        if k == "noise":
            return self.noise.clone()
        return super().clone_key(k)

    def apply_effects(self, noise, sigma):
        blu = external.MODULES.bleh.py.latent_utils
        if self.ffilter:
            noise = blu.ffilter(
                noise,
                self.ffilter_threshold,
                self.ffilter_scale,
                self.ffilter,
                self.ffilter_strength,
            )
        if self.enhance_mode != "none" and self.enhance_strength != 0:
            noise = blu.enhance_tensor(
                noise,
                self.enhance_mode,
                self.enhance_strength,
                sigma=sigma,
                skip_multiplier=0,
                adjust_scale=False,
            )
        return noise

    def make_noise_sampler(self, x, *args, normalized=True, **kwargs):
        factor = self.factor
        noise_items = self.noise.items
        noise_samplers = tuple(
            ni.make_noise_sampler(x, *args, normalized=False, **kwargs)
            for ni in noise_items
        )
        num_samplers = len(noise_samplers)
        normalize_noise = self.get_normalize(
            "normalize_noise",
            normalized or num_samplers > 1,
        )
        normalize_result = self.get_normalize("normalize_result", normalized)
        noise_effects = self.affect in {"noise", "both"}
        result_effects = self.affect in {"result", "both"}
        noise_init = torch.zeros_like(x)

        def noise_sampler(s, sn):
            noise = noise_init.clone()
            for ni, ns in zip(noise_items, noise_samplers):
                curr_noise = scale_noise(ns(s, sn), normalized=normalize_noise)
                if noise_effects:
                    curr_noise = self.apply_effects(curr_noise, s)
                if self.blend_mode == "simple_add":
                    noise += curr_noise.mul_(ni.factor)
                else:
                    noise = utils.BLENDING_MODES[self.blend_mode](
                        noise,
                        curr_noise,
                        ni.factor,
                    )
            del curr_noise
            noise = scale_noise(noise, factor, normalized=normalize_result)
            if result_effects:
                noise = self.apply_effects(noise, s)
            return noise

        return noise_sampler


class QuantileFilteredNoise(CustomNoiseItemBase):
    def clone_key(self, k):
        if k == "noise":
            return self.noise.clone()
        return super().clone_key(k)

    def make_noise_sampler(
        self,
        x,
        sigma_min,
        sigma_max,
        *args,
        normalized=True,
        **kwargs,
    ):
        factor = self.factor
        normalize = self.get_normalize("normalize", normalized)
        ns = self.noise.make_noise_sampler(
            x,
            *args,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            normalized=self.normalize_noise,
            **kwargs,
        )
        noise_filter = partial(
            quantile_normalize,
            quantile=self.quantile,
            dim=self.norm_dim,
            flatten=self.norm_flatten,
            nq_fac=self.norm_fac,
            pow_fac=self.norm_pow,
            strategy=self.strategy,
        )

        def noise_sampler(sigma, sigma_next):
            return scale_noise(
                noise_filter(ns(sigma, sigma_next)),
                factor,
                normalized=normalize,
            )

        return noise_sampler


class PerDimNoise(CustomNoiseItemBase):
    def clone_key(self, k):
        if k == "noise":
            return self.noise.clone()
        return super().clone_key(k)

    def make_noise_sampler(
        self,
        x,
        sigma_min,
        sigma_max,
        *args,
        normalized=True,
        **kwargs,
    ):
        factor = self.factor
        normalize = self.get_normalize("normalize", normalized)
        offset, chunk_size = self.offset, self.chunk_size
        dim = self.dim
        if dim < 0:
            dim = x.ndim + dim
        if dim < 0 or dim >= x.ndim:
            raise ValueError("Dimension out of range")
        dim_size = x.shape[dim]
        if self.shrink_dim:
            if offset + chunk_size > dim_size:
                raise ValueError("Offset or chunk size incompatible with tensor")
            x = x[
                tuple(
                    slice(offset, offset + chunk_size)
                    if d == dim
                    else slice(None, None)
                    for d in range(x.ndim)
                )
            ]
        ns = self.noise.make_noise_sampler(
            x,
            *args,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            normalized=self.normalize_noise,
            **kwargs,
        )
        trim_slice = tuple(
            slice(-dim_size, None) if d == dim else slice(None, None)
            for d in range(x.ndim)
        )
        if self.shrink_dim:

            def noise_sampler(sigma, sigma_next) -> torch.Tensor:
                noise = torch.cat(
                    tuple(ns(sigma, sigma_next) for _ in range(dim_size)),
                    dim=dim,
                )[trim_slice]
                return scale_noise(noise, factor, normalized=normalize)

        else:
            select_dim = [slice(None, None) for d in range(x.ndim)]
            n_chunks = math.ceil(dim_size / chunk_size)
            temp_shape = list(x.shape)
            temp_shape[dim] = int(n_chunks * chunk_size)

            def noise_sampler(sigma, sigma_next) -> torch.Tensor:
                nonlocal select_dim
                result = x.new_zeros(temp_shape)
                # result = torch.zeros_like(x)
                for idx in range(0, dim_size, chunk_size):
                    select_dim[dim] = slice(idx, idx + chunk_size)
                    result[select_dim] = ns(sigma, sigma_next)[select_dim]
                return scale_noise(result[trim_slice], factor, normalized=normalize)

        return noise_sampler


class ShuffledNoise(CustomNoiseItemBase):
    def clone_key(self, k):
        if k == "noise":
            return self.noise.clone()
        return super().clone_key(k)

    def make_noise_sampler(
        self,
        x,
        sigma_min,
        sigma_max,
        *args,
        normalized=True,
        **kwargs,
    ):
        factor = self.factor
        dims = tuple(d if d >= 0 else x.ndim + d for d in self.dims)
        if not all(d >= 0 and d < x.ndim for d in dims):
            raise ValueError("Dimension out of range")
        percentages = self.percentages
        if not all(0.0 <= p <= 1.0 for p in percentages):
            raise ValueError("Percentage out of range, must be between 0 and 1")
        ns = self.noise.make_noise_sampler(
            x,
            *args,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            normalized=normalized,
            **kwargs,
        )
        if not percentages or not dims or all(p == 0 for p in percentages):
            return ns
        n_percentages = len(percentages)
        fork_rng = self.fork_rng
        no_identity = self.no_identity

        def noise_sampler(sigma, sigma_next):
            noise = scale_noise(
                ns(sigma, sigma_next),
                factor,
                normalized=normalized,
            )
            with torch.random.fork_rng(enabled=fork_rng, devices=(noise.device,)):
                for idx, dim in enumerate(dims):
                    noise = utils.elementwise_shuffle_by_dim(
                        noise,
                        dim=dim,
                        prob=percentages[idx % n_percentages],
                        no_identity=no_identity,
                    )
            return noise

        return noise_sampler

    def make_noise_sampler_(
        self,
        x,
        sigma_min,
        sigma_max,
        *args,
        normalized=True,
        **kwargs,
    ):
        factor = self.factor
        dims = {x.ndim + d if d < 0 else d for d in self.dims}
        if not all(-1 < d < x.ndim for d in dims):
            raise ValueError("Dimension out of range")
        min_dim = min(dims) if dims else -1
        max_dim = max(dims) if dims else -1
        if max_dim == min_dim:
            max_dim = -1
        flatten = self.flatten
        percentage = self.percentage
        ns = self.noise.make_noise_sampler(
            x,
            *args,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            normalized=normalized,
            **kwargs,
        )
        if self.percentage == 0:
            return ns

        def noise_sampler(sigma, sigma_next):
            noise = scale_noise(
                ns(sigma, sigma_next),
                factor,
                normalized=normalized,
            )
            if not dims:
                return noise
            orig_shape = noise.shape
            if flatten:
                noise = noise.flatten(start_dim=min_dim, end_dim=max_dim)
                shuffles = ((min_dim, noise.shape[min_dim]),)
            else:
                shuffles = tuple((d, noise.shape[d]) for d in dims)
            for d, sz in shuffles:
                adj_sz = int(sz * percentage)
                if adj_sz == 0:
                    continue
                get_idxs = tuple(
                    torch.randperm(sz)[:adj_sz] if d == i else slice(None)
                    for i in range(noise.ndim)
                )
                set_idxs = (
                    ...
                    if percentage == 1.0
                    else tuple(
                        torch.randperm(sz)[:adj_sz] if d == i else slice(None)
                        for i in range(noise.ndim)
                    )
                )
                noise[set_idxs] = noise[get_idxs]
            return noise.reshape(orig_shape) if flatten else noise

        return noise_sampler


class PatternBreakNoise(CustomNoiseItemBase):
    def __init__(
        self,
        factor,
        *,
        noise,
        detail_level: float,
        percentage: float,
        restore_scale: bool,
        blend_mode: str = "lerp",
        blend_function=None,
    ):
        super().__init__(
            factor,
            noise=noise,
            detail_level=detail_level,
            percentage=percentage,
            restore_scale=restore_scale,
            blend_function=blend_function or utils.BLENDING_MODES[blend_mode],
        )

    def clone_key(self, k):
        if k == "noise":
            return self.noise.clone()
        return super().clone_key(k)

    def make_noise_sampler(
        self,
        x,
        sigma_min,
        sigma_max,
        *args,
        normalized=True,
        **kwargs,
    ):
        factor = self.factor
        ns = self.noise.make_noise_sampler(
            x,
            *args,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            normalized=normalized if self.percentage == 0 else False,
            **kwargs,
        )
        if self.percentage == 0:
            return ns
        noise_filter = partial(
            pattern_break,
            percentage=self.percentage,
            detail_level=self.detail_level,
            blend_function=self.blend_function,
            restore_scale=self.restore_scale,
        )

        def noise_sampler(sigma, sigma_next):
            return scale_noise(
                noise_filter(ns(sigma, sigma_next)),
                factor,
                normalized=normalized,
            )

        return noise_sampler


class CustomNoiseParametersNoise(CustomNoiseItemBase):
    def clone_key(self, k):
        if k == "noise":
            return self.noise.clone()
        return super().clone_key(k)

    def make_noise_sampler(
        self,
        x,
        sigma_min,
        sigma_max,
        *args,
        normalized=True,
        **kwargs,
    ):
        factor = self.factor
        normalize = self.get_normalize("normalize", normalized)
        orig_shape = x.shape
        orig_dtype = x.dtype
        orig_device = x.device
        if self.override_device is not None:
            kwargs["cpu"] = self.override_device == "cpu"
            x = x.to(device=self.override_device)
        if x.ndim == 5 and self.frames_to_channels:
            x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], *x.shape[3:])
        fix_invalid = self.fix_invalid
        if self.override_dtype and x.dtype != self.override_dtype:
            x = x.to(dtype=self.override_dtype)
        fixed_aspect = False
        if self.ensure_square_aspect_ratio:
            if x.ndim == 3:
                height, width = 1, x.shape[-1]
                spatdims = 1
            else:
                spatdims = 2
                height, width = x.shape[-2:]
            hw = (height * width) ** 0.5
            if not hw.is_integer():
                fixed_aspect = True
                hw = math.ceil(hw)
                temp_x = x.new_zeros(*x.shape[:-spatdims], hw**2)
                temp_x[..., : height * width] = x.flatten(start_dim=-spatdims)[
                    ...,
                    : height * width,
                ]
                x = temp_x.reshape(*temp_x.shape[:-1], hw, hw)
        if self.rng_offset_mode in {"override", "add"}:
            seed = (
                self.rng_state_offset
                if self.rng_offset_mode == "override"
                else kwargs.pop("seed", 0) + self.rng_state_offset
            )
            kwargs["seed"] = seed
        else:
            seed = kwargs.get("seed", 0)
        rng_mode = self.rng_mode
        if rng_mode == "separate":
            rng_state = RNGStates(x.device.type)
            if self.rng_offset_mode != "disabled":
                temp_rng_state = rng_state
                try:
                    random.seed(seed)
                    torch.manual_seed(seed)
                    rng_state = RNGStates(x.device.type)
                finally:
                    temp_rng_state.set_states()
                del temp_rng_state
        else:
            rng_state = None
        ns = self.noise.make_noise_sampler(
            x,
            *args,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            normalized=False,
            **kwargs,
        )
        device_type = x.device.type

        def noise_sampler(sigma, sigma_next) -> torch.Tensor:
            if rng_mode != "default":
                temp_rng_state = RNGStates(device_type)
                try:
                    if rng_mode == "separate":
                        rng_state.set_states()
                    noise = ns(sigma, sigma_next)
                    if rng_mode == "separate":
                        rng_state.update()
                finally:
                    temp_rng_state.set_states()
            else:
                noise = ns(sigma, sigma_next)
            if fix_invalid:
                noise_temp = noise.nan_to_num(0, posinf=0, neginf=0)
                noise = noise.nan_to_num_(
                    0,
                    posinf=noise_temp.max(),
                    neginf=noise_temp.min(),
                )
            if fixed_aspect:
                noise = noise.flatten(start_dim=-spatdims)[..., : height * width]
            if noise.shape != orig_shape:
                noise = noise.reshape(orig_shape)
            if noise.dtype != orig_dtype or noise.device != orig_device:
                noise = noise.to(device=orig_device, dtype=orig_dtype)
            return scale_noise(noise, factor, normalized=normalize)

        return noise_sampler


class BlehOpsNoise(CustomNoiseItemBase):
    def __init__(
        self,
        factor,
        *,
        noise,
        rules,
        normalize,
    ):
        if len(noise.items) == 0:
            raise ValueError("BlehOpsNoise requires at least one noise item")
        super().__init__(
            factor,
            noise=noise.clone(),
            rules=rules,
            normalize=normalize,
        )

    def clone_key(self, k):
        if k == "noise":
            return self.noise.clone()
        return super().clone_key(k)

    def make_noise_sampler(self, x, *args, normalized=True, **kwargs):
        bops = external.MODULES.bleh.py.nodes.ops
        factor = self.factor
        normalize = self.get_normalize("normalize", normalized)
        rulegroup = self.rules
        internal_ns = self.noise.make_noise_sampler(
            x,
            *args,
            normalized=False,
            **kwargs,
        )

        def noise_sampler(s, sn):
            noise = internal_ns(s, sn)
            if len(rulegroup.rules):
                state = {
                    bops.CondType.TYPE: bops.PatchType.LATENT,
                    bops.CondType.PERCENT: 0.0,
                    bops.CondType.BLOCK: -1,
                    bops.CondType.STAGE: -1,
                    "sigma": None if s is None else s,
                    "h": noise,
                    "hsp": x.detach().clone(),
                    "target": "h",
                }
                noise = rulegroup.eval(state, toplevel=True)["h"]
            return scale_noise(noise, factor, normalized=normalize)

        return noise_sampler


NOISE_SAMPLERS: dict[NoiseType, Callable] = {
    NoiseType.BROWNIAN: NoiseSampler.wrap(BrownianNoiseGenerator),
    NoiseType.DISTRO: NoiseSampler.wrap(DistroNoiseGenerator),
    NoiseType.GAUSSIAN: NoiseSampler.wrap(GaussianNoiseGenerator),
    NoiseType.UNIFORM: NoiseSampler.wrap(UniformNoiseGenerator),
    NoiseType.PERLIN: NoiseSampler.wrap(PerlinOldNoiseGenerator),
    NoiseType.STUDENTT: NoiseSampler.wrap(StudentTNoiseGenerator),
    NoiseType.ONEF_PINKISH: NoiseSampler.wrap(partial(OneFNoiseGenerator, alpha=-0.5)),
    NoiseType.ONEF_GREENISH: NoiseSampler.wrap(partial(OneFNoiseGenerator, alpha=0.5)),
    NoiseType.ONEF_PINKISHGREENISH: NoiseSampler.wrap(
        partial(
            MixedNoiseGenerator,
            name="onef_pinkishgreenish",
            noise_mix=(
                (OneFNoiseGenerator, {"alpha": 0.5}, None),
                (OneFNoiseGenerator, {"alpha": -0.5}, None),
            ),
            output_fun=lambda t: t.mul_(0.5),
        ),
    ),
    NoiseType.ONEF_PINKISH_MIX: NoiseSampler.wrap(
        partial(
            MixedNoiseGenerator,
            name="onef_pinkish_mix",
            noise_mix=(
                (OneFNoiseGenerator, {"alpha": -0.5}, lambda t: t.mul_(-1.0)),
                (OneFNoiseGenerator, {"alpha": -0.5}, None),
            ),
            output_fun=lambda t: t.mul_(0.5),
        ),
    ),
    NoiseType.ONEF_GREENISH_MIX: NoiseSampler.wrap(
        partial(
            MixedNoiseGenerator,
            name="onef_greenish_mix",
            noise_mix=(
                (OneFNoiseGenerator, {"alpha": 0.5}, lambda t: t.mul_(-1.0)),
                (OneFNoiseGenerator, {"alpha": 0.5}, None),
            ),
            output_fun=lambda t: t.mul_(0.5),
        ),
    ),
    NoiseType.WHITE: NoiseSampler.wrap(
        partial(
            PowerLawNoiseGenerator,
            alpha=0.0,
            use_sign=True,
        ),
    ),
    NoiseType.GREY: NoiseSampler.wrap(
        partial(
            PowerLawNoiseGenerator,
            alpha=0.0,
            use_sign=False,
        ),
    ),
    NoiseType.VELVET: NoiseSampler.wrap(
        partial(
            PowerLawNoiseGenerator,
            alpha=1.0,
            use_sign=True,
            div_max_dims=(-3, -2, -1),
        ),
    ),
    NoiseType.VIOLET: NoiseSampler.wrap(
        partial(
            PowerLawNoiseGenerator,
            alpha=0.5,
            use_sign=True,
            div_max_dims=(-3, -2, -1),
        ),
    ),
    NoiseType.WAVELET: NoiseSampler.wrap(WaveletNoiseGenerator),
    NoiseType.PINK_OLD: NoiseSampler.wrap(PinkOldNoiseGenerator),
    NoiseType.HIGHRES_PYRAMID: NoiseSampler.wrap(HighresPyramidNoiseGenerator),
    NoiseType.PYRAMID: NoiseSampler.wrap(PyramidNoiseGenerator),
    NoiseType.RAINBOW_MILD: NoiseSampler.wrap(
        partial(
            MixedNoiseGenerator,
            name="rainbow_mild",
            noise_mix=(
                (GreenTestNoiseGenerator, {}, lambda t: t.mul_(0.55)),
                (GreenTestNoiseGenerator, {}, lambda t: t.mul_(0.7)),
            ),
            output_fun=lambda t: t.mul_(1.15),
        ),
    ),
    NoiseType.RAINBOW_INTENSE: NoiseSampler.wrap(
        partial(
            MixedNoiseGenerator,
            name="rainbow_intense",
            noise_mix=(
                (GreenTestNoiseGenerator, {}, lambda t: t.mul_(0.75)),
                (GreenTestNoiseGenerator, {}, lambda t: t.mul_(0.5)),
            ),
            output_fun=lambda t: t.mul_(1.15),
        ),
    ),
    NoiseType.LAPLACIAN: NoiseSampler.wrap(LaplacianNoiseGenerator),
    NoiseType.POWER_OLD: NoiseSampler.wrap(PowerOldNoiseGenerator),
    NoiseType.GREEN_TEST: NoiseSampler.wrap(GreenTestNoiseGenerator),
    NoiseType.PYRAMID_OLD: NoiseSampler.wrap(PyramidOldNoiseGenerator),
    NoiseType.PYRAMID_BISLERP: NoiseSampler.wrap(
        partial(PyramidNoiseGenerator, upscale_mode="bislerp"),
    ),
    NoiseType.HIGHRES_PYRAMID_BISLERP: NoiseSampler.wrap(
        partial(HighresPyramidNoiseGenerator, upscale_mode="bislerp"),
    ),
    NoiseType.PYRAMID_AREA: NoiseSampler.wrap(
        partial(PyramidNoiseGenerator, upscale_mode="area"),
    ),
    NoiseType.HIGHRES_PYRAMID_AREA: NoiseSampler.wrap(
        partial(HighresPyramidNoiseGenerator, upscale_mode="area"),
    ),
    NoiseType.PYRAMID_OLD_BISLERP: NoiseSampler.wrap(
        partial(PyramidOldNoiseGenerator, upscale_mode="bislerp"),
    ),
    NoiseType.PYRAMID_OLD_AREA: NoiseSampler.wrap(
        partial(PyramidOldNoiseGenerator, upscale_mode="area"),
    ),
    NoiseType.PYRAMID_DISCOUNT5: NoiseSampler.wrap(
        partial(PyramidNoiseGenerator, discount=0.5),
    ),
    NoiseType.PYRAMID_MIX: NoiseSampler.wrap(
        partial(
            MixedNoiseGenerator,
            name="pyramid_mix",
            noise_mix=(
                (PyramidNoiseGenerator, {"discount": 0.6}, lambda t: t.mul_(0.2)),
                (PyramidNoiseGenerator, {"discount": 0.6}, lambda t: t.mul_(-0.8)),
            ),
        ),
    ),
    NoiseType.PYRAMID_MIX_AREA: NoiseSampler.wrap(
        partial(
            MixedNoiseGenerator,
            name="pyramid_mix_area",
            noise_mix=(
                (
                    PyramidNoiseGenerator,
                    {"discount": 0.5, "upscale_mode": "area"},
                    lambda t: t.mul_(0.2),
                ),
                (
                    PyramidNoiseGenerator,
                    {"discount": 0.5, "upscale_mode": "area"},
                    lambda t: t.mul_(-0.8),
                ),
            ),
        ),
    ),
    NoiseType.PYRAMID_MIX_BISLERP: NoiseSampler.wrap(
        partial(
            MixedNoiseGenerator,
            name="pyramid_mix_bislerp",
            noise_mix=(
                (
                    PyramidNoiseGenerator,
                    {
                        "discount": 0.5,
                        "upscale_mode": "bislerp",
                    },
                    lambda t: t.mul_(0.2),
                ),
                (
                    PyramidNoiseGenerator,
                    {
                        "discount": 0.5,
                        "upscale_mode": "bislerp",
                    },
                    lambda t: t.mul_(-0.8),
                ),
            ),
        ),
    ),
    NoiseType.COLLATZ: NoiseSampler.wrap(CollatzNoiseGenerator),
    NoiseType.VORONOI_FUZZ: NoiseSampler.wrap(
        partial(
            VoronoiNoiseGenerator,
            n_points=(256,),
            octaves=1,
            distance_mode=("fuzz:name=angle_tanh:fuzz=0.1",),
            result_mode=("diff2",),
            z_max=0.0,
        ),
    ),
    NoiseType.VORONOI_MIX: NoiseSampler.wrap(
        partial(
            MixedNoiseGenerator,
            name="voronoi_mix",
            noise_mix=(
                (
                    VoronoiNoiseGenerator,
                    {
                        "n_points": (256,),
                        "octaves": 3,
                        "distance_mode": ("euclidean",),
                        "result_mode": ("diff2",),
                        "octave_mode": "new_features",
                        "lacunarity": 2.0,
                        "gain": 0.75,
                        "z_max": 0.0,
                    },
                    lambda t: t.mul_(0.6),
                ),
                (
                    GaussianNoiseGenerator,
                    {},
                    lambda t: t.mul_(0.4),
                ),
            ),
        ),
    ),
}


def get_noise_sampler(
    noise_type: str | NoiseType | None,
    x: Tensor,
    sigma_min: float | None,
    sigma_max: float | None,
    seed: int | None = None,
    cpu: bool = True,
    factor: float = 1.0,
    normalized=False,
    **kwargs,
) -> Callable:
    if noise_type is None:
        noise_type = NoiseType.GAUSSIAN
    elif isinstance(noise_type, str):
        noise_type = NoiseType[noise_type.upper()]
    if noise_type == NoiseType.BROWNIAN and (sigma_min is None or sigma_max is None):
        raise ValueError("Must pass sigma min/max when using brownian noise")
    mkns = NOISE_SAMPLERS.get(noise_type)
    if mkns is None:
        raise ValueError("Unknown noise sampler")
    return mkns(
        x,
        sigma_min,
        sigma_max,
        seed=seed,
        cpu=cpu,
        factor=factor,
        normalized=normalized,
        **kwargs,
    )
