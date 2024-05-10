from __future__ import annotations

import abc
import inspect
from types import SimpleNamespace
from typing import Any, Callable

import torch
from comfy import samplers

from . import noise
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
        latent_samples = latent["samples"]
        if custom_noise_opt is not None:
            ns = custom_noise_opt.make_noise_sampler(latent_samples)
        else:
            ns = noise.get_noise_sampler(
                NoiseType[noise_type.upper()],
                latent_samples,
                None,
                None,
                seed=seed,
                cpu=True,
            )
        randst = torch.random.get_rng_state()
        try:
            torch.random.manual_seed(seed)
            result = ns(None, None)
        finally:
            torch.random.set_rng_state(randst)
        result = scale_noise(result, multiplier, normalized=True)
        if add_to_latent:
            result += latent_samples.to(result.device)
        return ({"samples": result},)


class SonarCustomNoiseNodeBase(abc.ABC):
    @abc.abstractmethod
    def get_item_class(self):
        raise NotImplementedError

    @classmethod
    def INPUT_TYPES(cls):
        return {
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
            },
            "optional": {
                "sonar_custom_noise_opt": ("SONAR_CUSTOM_NOISE",),
            },
        }

    RETURN_TYPES = ("SONAR_CUSTOM_NOISE",)
    CATEGORY = "advanced/noise"
    FUNCTION = "go"

    def go(
        self,
        factor,
        rescale,
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


class SonarModulatedNoiseNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
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
            },
            "optional": {"ref_latent_opt": ("LATENT",)},
        }

    RETURN_TYPES = ("SONAR_CUSTOM_NOISE",)
    CATEGORY = "advanced/noise"
    FUNCTION = "go"

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
        normalize_result = (
            None if normalize_result == "default" else normalize_result == "forced"
        )
        normalize_noise = (
            None if normalize_noise == "default" else normalize_noise == "forced"
        )
        if ref_latent_opt is not None:
            ref_latent_opt = ref_latent_opt["samples"].clone()
        nis = noise.CustomNoiseChain()
        nis.add(
            noise.ModulatedNoise(
                factor,
                sonar_custom_noise.rescaled(1.0).make_noise_sampler,
                normalize_result,
                normalize_noise,
                normalize_ref,
                modulation_type=modulation_type,
                modulation_strength=strength,
                modulation_dims=dims,
                ref_latent_opt=ref_latent_opt,
            ),
        )
        return (nis,)


class SonarRepeatedNoiseNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
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
                "sonar_custom_noise": ("SONAR_CUSTOM_NOISE",),
                "repeat_length": ("INT", {"default": 8, "min": 1, "max": 100}),
                "normalize": (("default", "forced", "disabled"),),
                "permute": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("SONAR_CUSTOM_NOISE",)
    CATEGORY = "advanced/noise"
    FUNCTION = "go"

    def go(self, factor, sonar_custom_noise, repeat_length, normalize, permute=True):
        normalize = None if normalize == "default" else normalize == "forced"
        nis = noise.CustomNoiseChain()
        nis.add(
            noise.RepeatedNoise(
                factor,
                sonar_custom_noise.rescaled(1.0).make_noise_sampler,
                repeat_length,
                normalize,
                permute=permute,
            ),
        )
        return (nis,)


class SonarScheduledNoiseNode:
    RETURN_TYPES = ("SONAR_CUSTOM_NOISE",)
    CATEGORY = "advanced/noise"
    FUNCTION = "go"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
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
                "sonar_custom_noise": ("SONAR_CUSTOM_NOISE",),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
                "normalize": (("default", "forced", "disabled"),),
            },
            "optional": {"fallback_sonar_custom_noise": ("SONAR_CUSTOM_NOISE",)},
        }

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
        normalize = None if normalize == "default" else normalize == "forced"
        ms = model.get_model_object("model_sampling")
        start_sigma = ms.percent_to_sigma(start_percent)
        end_sigma = ms.percent_to_sigma(end_percent)
        return (
            noise.CustomNoiseChain(
                [
                    noise.ScheduledNoise(
                        factor,
                        sonar_custom_noise.rescaled(1.0).make_noise_sampler,
                        start_sigma,
                        end_sigma,
                        normalize,
                        fallback_noise_sampler=fallback_sonar_custom_noise.rescaled(
                            1.0,
                        ).make_noise_sampler
                        if fallback_sonar_custom_noise is not None
                        else None,
                    ),
                ],
            ),
        )


class SonarCompositeNoiseNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
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
                "sonar_custom_noise_dst": ("SONAR_CUSTOM_NOISE",),
                "sonar_custom_noise_src": ("SONAR_CUSTOM_NOISE",),
                "normalize_dst": (("default", "forced", "disabled"),),
                "normalize_src": (("default", "forced", "disabled"),),
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("SONAR_CUSTOM_NOISE",)
    CATEGORY = "advanced/noise"
    FUNCTION = "go"

    def go(
        self,
        factor,
        sonar_custom_noise_dst,
        sonar_custom_noise_src,
        normalize_src,
        normalize_dst,
        mask,
    ):
        normalize_src = (
            None if normalize_src == "default" else normalize_src == "forced"
        )
        normalize_dst = (
            None if normalize_dst == "default" else normalize_dst == "forced"
        )
        nis = noise.CustomNoiseChain()

        nis.add(
            noise.CompositeNoise(
                factor,
                sonar_custom_noise_dst.rescaled(1.0).make_noise_sampler,
                sonar_custom_noise_src.rescaled(1.0).make_noise_sampler,
                normalize_src,
                normalize_dst,
                mask.clone(),
            ),
        )
        return (nis,)


class SonarGuidedNoiseNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "sonar_custom_noise": ("SONAR_CUSTOM_NOISE",),
                "method": (("euler", "linear"),),
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
                "normalize": (("default", "forced", "disabled"),),
                "normalize_ref": (
                    "BOOLEAN",
                    {"default": True},
                ),
            },
        }

    RETURN_TYPES = ("SONAR_CUSTOM_NOISE",)
    CATEGORY = "advanced/noise"
    FUNCTION = "go"

    def go(
        self,
        latent,
        sonar_custom_noise,
        normalize,
        normalize_ref=True,
        method="euler",
        factor=1.0,
        guidance_factor=0.5,
    ):
        from .sonar import SonarGuidanceMixin

        normalize = None if normalize == "default" else normalize == "forced"
        nis = noise.CustomNoiseChain()
        nis.add(
            noise.GuidedNoise(
                factor,
                guidance_factor,
                SonarGuidanceMixin.prepare_ref_latent(latent["samples"].clone()),
                sonar_custom_noise.rescaled(1.0).make_noise_sampler,
                method,
                normalize,
                normalize_ref,
            ),
        )
        return (nis,)


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
        sampler, noise_type, custom_noise = (
            cfg["sampler"],
            cfg.get("noise_type"),
            cfg.get("custom_noise"),
        )
        sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
        seed = extra_args.get("seed")
        if custom_noise is not None:
            noise_sampler = custom_noise.make_noise_sampler(
                x,
                sigma_min,
                sigma_max,
                seed=seed,
            )
        elif noise_type is not None:
            noise_sampler = noise.get_noise_sampler(
                noise_type,
                x,
                sigma_min,
                sigma_max,
                seed=seed,
                cpu=True,
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
    "SonarGuidanceConfig": GuidanceConfigNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    import custom_nodes.ComfyUI_restart_sampling as rs

    if not hasattr(rs.restart_sampling, "DEFAULT_SEGMENTS"):
        # Dumb test but this should only exist in restart sampling versions that
        # support plugging in custom noise.
        raise NotImplementedError  # noqa: TRY301

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

    if not hasattr(rs.restart_sampling, "RestartSampler"):
        # Dumb test part II: The Dumbening
        raise NotImplementedError  # noqa: TRY301

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
except (ImportError, NotImplementedError):
    pass
