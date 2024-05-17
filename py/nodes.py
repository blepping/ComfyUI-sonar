from __future__ import annotations

import abc
import inspect
from types import SimpleNamespace
from typing import Any, Callable

import torch
from comfy import samplers

from . import external, noise
from .noise import NoiseType
from .noise_generation import scale_noise
from .sonar import (
    GuidanceConfig,
    GuidanceType,
    HistoryType,
    SonarConfig,
    SonarDPMPPSDE,
    SonarEuler,
    SonarEulerAncestral,
)


class NoisyLatentLikeNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise_type": (tuple(NoiseType.get_names(skip=(NoiseType.BROWNIAN,))),),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "latent": ("LATENT",),
                "multiplier": ("FLOAT", {"default": 1.0}),
                "add_to_latent": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "custom_noise_opt": ("SONAR_CUSTOM_NOISE",),
                "mul_by_sigmas_opt": ("SIGMAS",),
                "model_opt": ("MODEL",),
            },
        }

    RETURN_TYPES = ("LATENT",)
    CATEGORY = "latent/noise"

    FUNCTION = "go"

    def go(
        self,
        noise_type: str,
        seed: None | int,
        latent: dict,
        multiplier: float = 1.0,
        add_to_latent=False,
        custom_noise_opt: object | None = None,
        mul_by_sigmas_opt: None | torch.Tensor = None,
        model_opt: object | None = None,
    ):
        model, sigmas = model_opt, mul_by_sigmas_opt
        if sigmas is not None and len(sigmas) > 0:
            if model is None:
                raise ValueError(
                    "NoisyLatentLike requires a model when sigmas are connected!",
                )
            while hasattr(model, "model"):
                model = model.model
            latent_scale_factor = model.latent_format.scale_factor
            max_denoise = samplers.Sampler().max_denoise(
                SimpleNamespace(inner_model=model),
                sigmas,
            )
            multiplier *= (
                float(
                    torch.sqrt(1.0 + sigmas[0] ** 2.0) if max_denoise else sigmas[0],
                )
                / latent_scale_factor
            )
        if sigmas is not None and sigmas.numel() > 1:
            sigma_min, sigma_max = sigmas[0], sigmas[-1]
            sigma, sigma_next = sigmas[0], sigmas[1]
        else:
            sigma_min, sigma_max, sigma, sigma_next = (None,) * 4
        latent_samples = latent["samples"]
        if custom_noise_opt is not None:
            ns = custom_noise_opt.make_noise_sampler(
                latent_samples,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
            )
        else:
            ns = noise.get_noise_sampler(
                NoiseType[noise_type.upper()],
                latent_samples,
                sigma_min,
                sigma_max,
                seed=seed,
                cpu=True,
            )
        randst = torch.random.get_rng_state()
        try:
            torch.random.manual_seed(seed)
            result = ns(sigma, sigma_next)
        finally:
            torch.random.set_rng_state(randst)
        result = scale_noise(result, multiplier, normalized=True)
        if add_to_latent:
            result += latent_samples.to(result.device)
        return ({"samples": result},)


class SonarCustomNoiseNodeBase(abc.ABC):
    RETURN_TYPES = ("SONAR_CUSTOM_NOISE",)
    CATEGORY = "advanced/noise"
    FUNCTION = "go"

    @abc.abstractmethod
    def get_item_class(self):
        raise NotImplementedError

    @classmethod
    def INPUT_TYPES(cls, *, include_rescale=True, include_chain=True):
        result = {
            "required": {
                "factor": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -100.0,
                        "max": 100.0,
                        "step": 0.001,
                        "round": False,
                    },
                ),
            },
            "optional": {},
        }
        if include_rescale:
            result["required"] |= {
                "rescale": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.001,
                        "round": False,
                    },
                ),
            }
        if include_chain:
            result["optional"] |= {
                "sonar_custom_noise_opt": ("SONAR_CUSTOM_NOISE",),
            }
        return result

    def go(
        self,
        factor=1.0,
        rescale=0.0,
        sonar_custom_noise_opt=None,
        **kwargs: dict[str, Any],
    ):
        nis = (
            sonar_custom_noise_opt.clone()
            if sonar_custom_noise_opt
            else noise.CustomNoiseChain()
        )
        if factor != 0:
            nis.add(self.get_item_class()(factor, **kwargs))
        return (nis if rescale == 0 else nis.rescaled(rescale),)


class SonarCustomNoiseNode(SonarCustomNoiseNodeBase):
    @classmethod
    def INPUT_TYPES(cls):
        result = super().INPUT_TYPES()
        result["required"] |= {
            "noise_type": (tuple(NoiseType.get_names()),),
        }
        return result

    def get_item_class(self):
        return noise.CustomNoiseItem


class SonarNormalizeNoiseNodeMixin:
    @staticmethod
    def get_normalize(val: str) -> None | bool:
        return None if val == "default" else val == "forced"


class SonarModulatedNoiseNode(SonarCustomNoiseNodeBase, SonarNormalizeNoiseNodeMixin):
    @classmethod
    def INPUT_TYPES(cls):
        result = super().INPUT_TYPES(include_rescale=False, include_chain=False)
        result["required"] |= {
            "sonar_custom_noise": ("SONAR_CUSTOM_NOISE",),
            "modulation_type": (
                (
                    "intensity",
                    "frequency",
                    "spectral_signum",
                    "none",
                ),
            ),
            "dims": ("INT", {"default": 3, "min": 1, "max": 3}),
            "strength": ("FLOAT", {"default": 2.0, "min": -100.0, "max": 100.0}),
            "normalize_result": (("default", "forced", "disabled"),),
            "normalize_noise": (("default", "forced", "disabled"),),
            "normalize_ref": (
                "BOOLEAN",
                {"default": True},
            ),
        }
        result["optional"] |= {"ref_latent_opt": ("LATENT",)}
        return result

    def get_item_class(self):
        return noise.ModulatedNoise

    def go(
        self,
        factor,
        sonar_custom_noise,
        modulation_type,
        dims,
        strength,
        normalize_result,
        normalize_noise,
        normalize_ref,
        ref_latent_opt=None,
    ):
        if ref_latent_opt is not None:
            ref_latent_opt = ref_latent_opt["samples"].clone()
        return super().go(
            factor,
            noise=sonar_custom_noise,
            modulation_type=modulation_type,
            modulation_dims=dims,
            modulation_strength=strength,
            normalize_result=self.get_normalize(normalize_result),
            normalize_noise=self.get_normalize(normalize_noise),
            normalize_ref=self.get_normalize(normalize_ref),
            ref_latent_opt=ref_latent_opt,
        )


class SonarRepeatedNoiseNode(SonarCustomNoiseNodeBase, SonarNormalizeNoiseNodeMixin):
    @classmethod
    def INPUT_TYPES(cls):
        result = super().INPUT_TYPES(include_rescale=False, include_chain=False)
        result["required"] |= {
            "sonar_custom_noise": ("SONAR_CUSTOM_NOISE",),
            "repeat_length": ("INT", {"default": 8, "min": 1, "max": 100}),
            "max_recycle": ("INT", {"default": 1000, "min": 1, "max": 1000}),
            "normalize": (("default", "forced", "disabled"),),
            "permute": (("enabled", "disabled", "always"),),
        }
        return result

    def get_item_class(self):
        return noise.RepeatedNoise

    def go(
        self,
        factor,
        sonar_custom_noise,
        repeat_length,
        max_recycle,
        normalize,
        permute=True,
    ):
        return super().go(
            factor,
            noise=sonar_custom_noise,
            repeat_length=repeat_length,
            max_recycle=max_recycle,
            normalize=self.get_normalize(normalize),
            permute=permute,
        )


class SonarScheduledNoiseNode(SonarCustomNoiseNodeBase, SonarNormalizeNoiseNodeMixin):
    @classmethod
    def INPUT_TYPES(cls):
        result = super().INPUT_TYPES(include_rescale=False, include_chain=False)
        result["required"] |= {
            "model": ("MODEL",),
            "sonar_custom_noise": ("SONAR_CUSTOM_NOISE",),
            "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
            "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
            "normalize": (("default", "forced", "disabled"),),
        }
        result["optional"] |= {"fallback_sonar_custom_noise": ("SONAR_CUSTOM_NOISE",)}
        return result

    def get_item_class(self):
        return noise.ScheduledNoise

    def go(
        self,
        model,
        factor,
        sonar_custom_noise,
        start_percent,
        end_percent,
        normalize,
        fallback_sonar_custom_noise=None,
    ):
        ms = model.get_model_object("model_sampling")
        start_sigma = ms.percent_to_sigma(start_percent)
        end_sigma = ms.percent_to_sigma(end_percent)
        return super().go(
            factor,
            noise=sonar_custom_noise,
            start_sigma=start_sigma,
            end_sigma=end_sigma,
            normalize=self.get_normalize(normalize),
            fallback_noise=fallback_sonar_custom_noise,
        )


class SonarCompositeNoiseNode(SonarCustomNoiseNodeBase, SonarNormalizeNoiseNodeMixin):
    @classmethod
    def INPUT_TYPES(cls):
        result = super().INPUT_TYPES(include_rescale=False, include_chain=False)
        result["required"] |= {
            "sonar_custom_noise_dst": ("SONAR_CUSTOM_NOISE",),
            "sonar_custom_noise_src": ("SONAR_CUSTOM_NOISE",),
            "normalize_dst": (("default", "forced", "disabled"),),
            "normalize_src": (("default", "forced", "disabled"),),
            "normalize_result": (("default", "forced", "disabled"),),
            "mask": ("MASK",),
        }
        return result

    def get_item_class(self):
        return noise.CompositeNoise

    def go(
        self,
        factor,
        sonar_custom_noise_dst,
        sonar_custom_noise_src,
        normalize_src,
        normalize_dst,
        normalize_result,
        mask,
    ):
        return super().go(
            factor,
            dst_noise=sonar_custom_noise_dst,
            src_noise=sonar_custom_noise_src,
            normalize_dst=self.get_normalize(normalize_src),
            normalize_src=self.get_normalize(normalize_dst),
            normalize_result=self.get_normalize(normalize_result),
            mask=mask,
        )


class SonarGuidedNoiseNode(SonarCustomNoiseNodeBase, SonarNormalizeNoiseNodeMixin):
    @classmethod
    def INPUT_TYPES(cls):
        result = super().INPUT_TYPES(include_rescale=False, include_chain=False)
        result["required"] |= {
            "latent": ("LATENT",),
            "sonar_custom_noise": ("SONAR_CUSTOM_NOISE",),
            "method": (("euler", "linear"),),
            "guidance_factor": (
                "FLOAT",
                {
                    "default": 0.0125,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 0.001,
                    "round": False,
                },
            ),
            "normalize_noise": (("default", "forced", "disabled"),),
            "normalize_result": (("default", "forced", "disabled"),),
            "normalize_ref": (
                "BOOLEAN",
                {"default": True},
            ),
        }
        return result

    def get_item_class(self):
        return noise.GuidedNoise

    def go(
        self,
        factor,
        latent,
        sonar_custom_noise,
        normalize_noise,
        normalize_result,
        normalize_ref=True,
        method="euler",
        guidance_factor=0.5,
    ):
        from .sonar import SonarGuidanceMixin

        return super().go(
            factor,
            ref_latent=scale_noise(
                SonarGuidanceMixin.prepare_ref_latent(latent["samples"].clone()),
                normalized=normalize_ref,
            ),
            guidance_factor=guidance_factor,
            noise=sonar_custom_noise.clone(),
            method=method,
            normalize_noise=self.get_normalize(normalize_noise),
            normalize_result=self.get_normalize(normalize_result),
        )


class SonarRandomNoiseNode(SonarCustomNoiseNodeBase, SonarNormalizeNoiseNodeMixin):
    @classmethod
    def INPUT_TYPES(cls):
        result = super().INPUT_TYPES(include_rescale=False, include_chain=False)
        result["required"] |= {
            "sonar_custom_noise": ("SONAR_CUSTOM_NOISE",),
            "mix_count": ("INT", {"default": 1, "min": 1, "max": 100}),
            "normalize": (("default", "forced", "disabled"),),
        }

        return result

    def get_item_class(self):
        return noise.RandomNoise

    def go(
        self,
        factor,
        sonar_custom_noise,
        mix_count,
        normalize,
    ):
        return super().go(
            factor,
            noise=sonar_custom_noise,
            mix_count=mix_count,
            normalize=self.get_normalize(normalize),
        )


class GuidanceConfigNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "factor": (
                    "FLOAT",
                    {
                        "default": 0.01,
                        "min": -2.0,
                        "max": 2.0,
                        "step": 0.001,
                        "round": False,
                    },
                ),
                "guidance_type": (tuple(t.name.lower() for t in GuidanceType),),
                "start_step": ("INT", {"default": 1, "min": 1}),
                "end_step": ("INT", {"default": 9999, "min": 1}),
                "latent": ("LATENT",),
            },
        }

    RETURN_TYPES = ("SONAR_GUIDANCE_CFG",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "make_guidance_cfg"

    def make_guidance_cfg(
        self,
        guidance_type,
        factor,
        start_step,
        end_step,
        latent,
    ):
        return (
            GuidanceConfig(
                guidance_type=GuidanceType[guidance_type.upper()],
                factor=factor,
                start_step=start_step,
                end_step=end_step,
                latent=latent.get("samples"),
            ),
        )


class SamplerNodeSonarBase:
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
                "momentum_init": (tuple(t.name for t in HistoryType),),
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
                "rand_init_noise_type": (
                    tuple(NoiseType.get_names(skip=(NoiseType.BROWNIAN,))),
                ),
            },
            "optional": {
                "guidance_cfg_opt": ("SONAR_GUIDANCE_CFG",),
            },
        }

    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"


class SamplerNodeSonarEuler(SamplerNodeSonarBase):
    @classmethod
    def INPUT_TYPES(cls):
        result = super().INPUT_TYPES()
        result["required"].update(
            {
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
        )
        return result

    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(
        self,
        momentum,
        momentum_hist,
        momentum_init,
        direction,
        rand_init_noise_type,
        s_noise,
        guidance_cfg_opt=None,
    ):
        cfg = SonarConfig(
            momentum=momentum,
            init=HistoryType[momentum_init.upper()],
            momentum_hist=momentum_hist,
            direction=direction,
            rand_init_noise_type=NoiseType[rand_init_noise_type.upper()],
            guidance=guidance_cfg_opt,
        )
        return (
            samplers.KSAMPLER(
                SonarEuler.sampler,
                {
                    "s_noise": s_noise,
                    "sonar_config": cfg,
                },
            ),
        )


class SamplerNodeSonarEulerAncestral(SamplerNodeSonarEuler):
    @classmethod
    def INPUT_TYPES(cls):
        result = super().INPUT_TYPES()
        result["required"].update(
            {
                "eta": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.01,
                        "round": False,
                    },
                ),
                "noise_type": (tuple(NoiseType.get_names()),),
            },
        )
        result["optional"].update(
            {
                "custom_noise_opt": ("SONAR_CUSTOM_NOISE",),
            },
        )
        return result

    def get_sampler(
        self,
        momentum,
        momentum_hist,
        momentum_init,
        direction,
        rand_init_noise_type,
        noise_type,
        eta,
        s_noise,
        guidance_cfg_opt=None,
        custom_noise_opt=None,
    ):
        cfg = SonarConfig(
            momentum=momentum,
            init=HistoryType[momentum_init.upper()],
            momentum_hist=momentum_hist,
            direction=direction,
            rand_init_noise_type=NoiseType[rand_init_noise_type.upper()],
            noise_type=NoiseType[noise_type.upper()],
            custom_noise=custom_noise_opt.clone() if custom_noise_opt else None,
            guidance=guidance_cfg_opt,
        )
        return (
            samplers.KSAMPLER(
                SonarEulerAncestral.sampler,
                {
                    "sonar_config": cfg,
                    "eta": eta,
                    "s_noise": s_noise,
                },
            ),
        )


class SamplerNodeSonarDPMPPSDE(SamplerNodeSonarEuler):
    @classmethod
    def INPUT_TYPES(cls):
        result = super().INPUT_TYPES()
        result["required"].update(
            {
                "eta": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.01,
                        "round": False,
                    },
                ),
                "noise_type": (tuple(NoiseType.get_names(default=NoiseType.BROWNIAN)),),
            },
        )
        result["optional"].update(
            {
                "custom_noise_opt": ("SONAR_CUSTOM_NOISE",),
            },
        )
        return result

    def get_sampler(
        self,
        momentum,
        momentum_hist,
        momentum_init,
        direction,
        rand_init_noise_type,
        noise_type,
        eta,
        s_noise,
        guidance_cfg_opt=None,
        custom_noise_opt=None,
    ):
        cfg = SonarConfig(
            momentum=momentum,
            init=HistoryType[momentum_init.upper()],
            momentum_hist=momentum_hist,
            direction=direction,
            rand_init_noise_type=NoiseType[rand_init_noise_type.upper()],
            noise_type=NoiseType[noise_type.upper()],
            custom_noise=custom_noise_opt.clone() if custom_noise_opt else None,
            guidance=guidance_cfg_opt,
        )
        return (
            samplers.KSAMPLER(
                SonarDPMPPSDE.sampler,
                {
                    "sonar_config": cfg,
                    "eta": eta,
                    "s_noise": s_noise,
                },
            ),
        )


class SamplerNodeConfigOverride:
    KWARG_OVERRIDES = ("s_noise", "eta", "s_churn", "r", "solver_type")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampler": ("SAMPLER",),
                "eta": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "step": 0.01,
                        "round": False,
                    },
                ),
                "s_noise": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "step": 0.01,
                        "round": False,
                    },
                ),
                "s_churn": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "step": 0.01,
                        "round": False,
                    },
                ),
                "r": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "step": 0.01,
                        "round": False,
                    },
                ),
                "sde_solver": (("midpoint", "heun"),),
                "cpu_noise": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "noise_type": (tuple(NoiseType.get_names()),),
                "custom_noise_opt": ("SONAR_CUSTOM_NOISE",),
            },
        }

    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(
        self,
        sampler,
        eta,
        s_noise,
        s_churn,
        r,
        sde_solver,
        cpu_noise=True,
        noise_type=None,
        custom_noise_opt=None,
    ):
        return (
            samplers.KSAMPLER(
                self.sampler_function,
                extra_options=sampler.extra_options
                | {
                    "override_sampler_cfg": {
                        "sampler": sampler,
                        "noise_type": NoiseType[noise_type.upper()]
                        if noise_type is not None
                        else None,
                        "custom_noise": custom_noise_opt,
                        "s_noise": s_noise,
                        "eta": eta,
                        "s_churn": s_churn,
                        "r": r,
                        "solver_type": sde_solver,
                        "cpu_noise": cpu_noise,
                    },
                },
                inpaint_options=sampler.inpaint_options | {},
            ),
        )

    @classmethod
    @torch.no_grad()
    def sampler_function(
        cls,
        model,
        x,
        sigmas,
        *args: list[Any],
        override_sampler_cfg: dict[str, Any] | None = None,
        noise_sampler: Callable | None = None,
        extra_args: dict[str, Any] | None = None,
        **kwargs: dict[str, Any],
    ):
        if not override_sampler_cfg:
            raise ValueError("Override sampler config missing!")
        if extra_args is None:
            extra_args = {}
        cfg = override_sampler_cfg
        sampler, noise_type, custom_noise, cpu = (
            cfg["sampler"],
            cfg.get("noise_type"),
            cfg.get("custom_noise"),
            cfg.get("cpu_noise", True),
        )
        sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
        seed = extra_args.get("seed")
        if custom_noise is not None:
            noise_sampler = custom_noise.make_noise_sampler(
                x,
                sigma_min,
                sigma_max,
                seed=seed,
                cpu=cpu,
            )
        elif noise_type is not None:
            noise_sampler = noise.get_noise_sampler(
                noise_type,
                x,
                sigma_min,
                sigma_max,
                seed=seed,
                cpu=cpu,
                normalized=True,
            )
        sig = inspect.signature(sampler.sampler_function)
        params = sig.parameters
        kwargs = kwargs | {}
        if "noise_sampler" in params:
            kwargs["noise_sampler"] = noise_sampler
        for k in cls.KWARG_OVERRIDES:
            if k not in params or cfg.get(k) is None:
                continue
            kwargs[k] = cfg[k]
        return sampler.sampler_function(
            model,
            x,
            sigmas,
            *args,
            extra_args=extra_args,
            **kwargs,
        )


NODE_CLASS_MAPPINGS = {
    "SamplerSonarEuler": SamplerNodeSonarEuler,
    "SamplerSonarEulerA": SamplerNodeSonarEulerAncestral,
    "SamplerSonarDPMPPSDE": SamplerNodeSonarDPMPPSDE,
    "SamplerConfigOverride": SamplerNodeConfigOverride,
    "NoisyLatentLike": NoisyLatentLikeNode,
    "SonarCustomNoise": SonarCustomNoiseNode,
    "SonarCompositeNoise": SonarCompositeNoiseNode,
    "SonarModulatedNoise": SonarModulatedNoiseNode,
    "SonarRepeatedNoise": SonarRepeatedNoiseNode,
    "SonarScheduledNoise": SonarScheduledNoiseNode,
    "SonarGuidedNoise": SonarGuidedNoiseNode,
    "SonarRandomNoise": SonarRandomNoiseNode,
    "SonarGuidanceConfig": GuidanceConfigNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {}


if "bleh" in external.MODULES:
    bleh = external.MODULES["bleh"]
    bleh_latentutils = bleh.py.latent_utils

    class SonarBlendFilterNoiseNode(
        SonarCustomNoiseNodeBase,
        SonarNormalizeNoiseNodeMixin,
    ):
        @classmethod
        def INPUT_TYPES(cls):
            result = super().INPUT_TYPES(include_rescale=False, include_chain=False)
            result["required"] |= {
                "sonar_custom_noise": ("SONAR_CUSTOM_NOISE",),
                "blend_mode": (
                    ("simple_add", *bleh_latentutils.BLENDING_MODES.keys()),
                ),
                "ffilter": (tuple(bleh_latentutils.FILTER_PRESETS.keys()),),
                "ffilter_custom": ("STRING", {"default": ""}),
                "ffilter_scale": (
                    "FLOAT",
                    {"default": 1.0, "min": -100.0, "max": 100.0},
                ),
                "ffilter_strength": (
                    "FLOAT",
                    {"default": 0.0, "min": -100.0, "max": 100.0},
                ),
                "ffilter_threshold": (
                    "INT",
                    {"default": 1, "min": 1, "max": 32},
                ),
                "enhance_mode": (("none", *bleh_latentutils.ENHANCE_METHODS),),
                "enhance_strength": (
                    "FLOAT",
                    {"default": 0.0, "min": -100.0, "max": 100.0},
                ),
                "affect": (("result", "noise", "both"),),
                "normalize_result": (("default", "forced", "disabled"),),
                "normalize_noise": (("default", "forced", "disabled"),),
            }
            return result

        def get_item_class(self):
            return noise.BlendFilterNoise

        def go(
            self,
            factor,
            sonar_custom_noise,
            blend_mode,
            ffilter,
            ffilter_custom,
            ffilter_scale,
            ffilter_strength,
            ffilter_threshold,
            enhance_mode,
            enhance_strength,
            affect,
            normalize_result,
            normalize_noise,
        ):
            import ast

            ffilter_custom = ffilter_custom.strip()
            normalize_result = (
                None if normalize_result == "default" else normalize_result == "forced"
            )
            normalize_noise = (
                None if normalize_noise == "default" else normalize_noise == "forced"
            )
            if ffilter_custom:
                ffilter = ast.literal_eval(f"[{ffilter_custom}]")
            else:
                ffilter = bleh_latentutils.FILTER_PRESETS[ffilter]
            return super().go(
                factor,
                noise=sonar_custom_noise.rescaled(1.0),
                blend_mode=blend_mode,
                ffilter=ffilter,
                ffilter_scale=ffilter_scale,
                ffilter_strength=ffilter_strength,
                ffilter_threshold=ffilter_threshold,
                enhance_mode=enhance_mode,
                enhance_strength=enhance_strength,
                affect=affect,
                normalize_noise=self.get_normalize(normalize_noise),
                normalize_result=self.get_normalize(normalize_result),
            )

    NODE_CLASS_MAPPINGS["SonarBlendFilterNoise"] = SonarBlendFilterNoiseNode

if "restart" in external.MODULES:
    rs = external.MODULES["restart"]

    class KRestartSamplerCustomNoise:
        @classmethod
        def INPUT_TYPES(cls):
            get_normal_schedulers = getattr(
                rs.nodes,
                "get_supported_normal_schedulers",
                rs.nodes.get_supported_restart_schedulers,
            )
            return {
                "required": {
                    "model": ("MODEL",),
                    "add_noise": (["enable", "disable"],),
                    "noise_seed": (
                        "INT",
                        {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF},
                    ),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "sampler": ("SAMPLER",),
                    "scheduler": (get_normal_schedulers(),),
                    "positive": ("CONDITIONING",),
                    "negative": ("CONDITIONING",),
                    "latent_image": ("LATENT",),
                    "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                    "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                    "return_with_leftover_noise": (["disable", "enable"],),
                    "segments": (
                        "STRING",
                        {
                            "default": rs.restart_sampling.DEFAULT_SEGMENTS,
                            "multiline": False,
                        },
                    ),
                    "restart_scheduler": (rs.nodes.get_supported_restart_schedulers(),),
                    "chunked_mode": ("BOOLEAN", {"default": True}),
                },
                "optional": {
                    "custom_noise_opt": ("SONAR_CUSTOM_NOISE",),
                },
            }

        RETURN_TYPES = ("LATENT", "LATENT")
        RETURN_NAMES = ("output", "denoised_output")
        FUNCTION = "sample"
        CATEGORY = "sampling"

        def sample(
            self,
            model,
            add_noise,
            noise_seed,
            steps,
            cfg,
            sampler,
            scheduler,
            positive,
            negative,
            latent_image,
            start_at_step,
            end_at_step,
            return_with_leftover_noise,
            segments,
            restart_scheduler,
            chunked_mode=False,
            custom_noise_opt=None,
        ):
            return rs.restart_sampling.restart_sampling(
                model,
                noise_seed,
                steps,
                cfg,
                sampler,
                scheduler,
                positive,
                negative,
                latent_image,
                segments,
                restart_scheduler,
                disable_noise=add_noise == "disable",
                step_range=(start_at_step, end_at_step),
                force_full_denoise=return_with_leftover_noise != "enable",
                output_only=False,
                chunked_mode=chunked_mode,
                custom_noise=custom_noise_opt.make_noise_sampler
                if custom_noise_opt
                else None,
            )

    NODE_CLASS_MAPPINGS["KRestartSamplerCustomNoise"] = KRestartSamplerCustomNoise

    if hasattr(rs.restart_sampling, "RestartSampler"):

        class RestartSamplerCustomNoise:
            @classmethod
            def INPUT_TYPES(cls):
                return {
                    "required": {
                        "sampler": ("SAMPLER",),
                        "chunked_mode": ("BOOLEAN", {"default": True}),
                    },
                    "optional": {
                        "custom_noise_opt": ("SONAR_CUSTOM_NOISE",),
                    },
                }

            RETURN_TYPES = ("SAMPLER",)
            FUNCTION = "go"
            CATEGORY = "sampling/custom_sampling/samplers"

            def go(self, sampler, chunked_mode, custom_noise_opt=None):
                restart_options = {
                    "restart_chunked": chunked_mode,
                    "restart_wrapped_sampler": sampler,
                    "restart_custom_noise": None
                    if custom_noise_opt is None
                    else custom_noise_opt.make_noise_sampler,
                }
                restart_sampler = samplers.KSAMPLER(
                    rs.restart_sampling.RestartSampler.sampler_function,
                    extra_options=sampler.extra_options | restart_options,
                    inpaint_options=sampler.inpaint_options,
                )
                return (restart_sampler,)

        NODE_CLASS_MAPPINGS["RestartSamplerCustomNoise"] = RestartSamplerCustomNoise
