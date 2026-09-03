from __future__ import annotations

from enum import Enum, auto

import torch

from ..utils import (
    fallback,
    scale_noise,
    tensor_to,
)

# ruff: noqa: ANN002, ANN003


class NoiseType(Enum):
    BROWNIAN = auto()
    COLLATZ = auto()
    DISTRO = auto()
    GAUSSIAN = auto()
    GREEN_TEST = auto()
    GREY = auto()
    HIGHRES_PYRAMID = auto()
    HIGHRES_PYRAMID_AREA = auto()
    HIGHRES_PYRAMID_BISLERP = auto()
    LAPLACIAN = auto()
    ONEF_GREENISH = auto()
    ONEF_GREENISH_MIX = auto()
    ONEF_PINKISH = auto()
    ONEF_PINKISH_MIX = auto()
    ONEF_PINKISHGREENISH = auto()
    PERLIN = auto()
    PINK_OLD = auto()
    POWER_OLD = auto()
    PYRAMID = auto()
    PYRAMID_AREA = auto()
    PYRAMID_BISLERP = auto()
    PYRAMID_DISCOUNT5 = auto()
    PYRAMID_MIX = auto()
    PYRAMID_MIX_AREA = auto()
    PYRAMID_MIX_BISLERP = auto()
    PYRAMID_OLD = auto()
    PYRAMID_OLD_AREA = auto()
    PYRAMID_OLD_BISLERP = auto()
    RAINBOW_INTENSE = auto()
    RAINBOW_MILD = auto()
    STUDENTT = auto()
    UNIFORM = auto()
    VELVET = auto()
    VIOLET = auto()
    VORONOI_FUZZ = auto()
    VORONOI_MIX = auto()
    WAVELET = auto()
    WHITE = auto()

    @classmethod
    def get_names(cls, default=GAUSSIAN, skip=None):
        if default is not None:
            if isinstance(default, int):
                default = cls(default)
            yield default.name.lower()
        for nt in cls:
            if nt == default or (skip and nt in skip):
                continue
            yield nt.name.lower()


class NoiseError(Exception):
    pass


class NoiseGenerator:
    name = "unknown"
    MIN_DIMS = 1
    MAX_DIMS = 0

    def __init__(
        self,
        x,
        **kwargs,
    ):
        if x.ndim < self.MIN_DIMS:
            errstr = f"Noise generator {self.name} requires at least {self.MIN_DIMS} dimension(s) but got input with shape {x.shape}"
            raise ValueError(errstr)
        if self.MAX_DIMS > 0 and x.ndim > self.MAX_DIMS:
            errstr = f"Noise generator {self.name} requires at most {self.MAX_DIMS} dimension(s) but got input with shape {x.shape}"
            raise ValueError(errstr)
        params = self.ng_params()
        kwarg_params = params | kwargs
        for k in params:
            setattr(self, k, kwarg_params.pop(k))
        self.options = kwarg_params
        self.update_x(x)

    @classmethod
    def ng_params(cls):
        return {
            "normalized": True,
            "force_normalize": None,
            "normalize_dims": None,
            "cpu": True,
            "generator": None,
        }

    def update_x(self, x):
        self.shape = x.shape
        self.batch = self.channels = self.frames = self.height = self.width = None
        if x.ndim >= 2:
            self.batch, self.channels = x.shape[:2]
        if x.ndim > 2:
            self.width = x.shape[-1]
        if x.ndim > 3:
            self.height = x.shape[-2]
        if x.ndim == 5:
            self.frames = x.shape[-3]
        self.device = x.device
        self.gen_device = torch.device("cpu") if self.cpu else self.device
        self.layout = x.layout
        self.dtype = x.dtype

    def rand_like(
        self,
        *,
        fun=torch.randn,
        cpu=None,
        to_device=True,
        shape=None,
        dtype=None,
        layout=None,
        device=None,
        generator=None,
    ):
        cpu = fallback(cpu, self.cpu)
        noise = fun(
            *fallback(shape, self.shape),
            generator=fallback(generator, self.generator),
            dtype=fallback(dtype, self.dtype),
            layout=fallback(layout, self.layout),
            device=fallback(device, "cpu" if cpu else self.gen_device),
        )
        if to_device and noise.device != self.device:
            noise = tensor_to(noise, self.device)
        return noise

    def output_hook(self, noise):
        if noise.device != self.device:
            noise = tensor_to(noise, self.device)
        return scale_noise(
            noise,
            normalized=self.normalized
            and (self.force_normalize is None or self.force_normalize is True),
            normalize_dims=self.normalize_dims,
        )

    def pre_hook(self):
        pass

    def generate(self):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        self.pre_hook()
        return self.output_hook(self.generate(*args, **kwargs))

    def __str__(self):
        pretty_params = ", ".join(f"{k}={getattr(self, k)!s}" for k in self.ng_params())
        return f"<NoiseGenerator({self.name}): device={self.device}, shape={self.shape}, dtype={self.dtype}, {pretty_params}>"


class FramesToChannelsNoiseGenerator(NoiseGenerator):
    MIN_DIMS = 4
    MAX_DIMS = 5

    def get_adjusted_shape(self):
        if self.frames:
            return (self.batch, self.channels * self.frames, self.height, self.width)
        return (self.batch, self.channels, self.height, self.width)

    def fix_output_frames(self, noise):
        if not self.frames:
            return noise
        return noise.reshape(
            self.batch,
            self.channels,
            self.frames,
            self.height,
            self.width,
        )

    def rand_like(self, *args, shape=None, **kwargs):
        noise = super().rand_like(*args, shape=shape, **kwargs)
        if shape is not None:
            return noise
        adjusted_shape = self.get_adjusted_shape()
        if noise.shape != adjusted_shape:
            return noise.reshape(*adjusted_shape)
        return noise


class MixedNoiseGenerator(NoiseGenerator):
    @classmethod
    def ng_params(cls):
        return super().ng_params() | {
            "name": "mixed_noise",
            "normalized": True,
            "pass_args": frozenset(("cpu",)),
            "noise_mix": (),
            "output_fun": None,
        }

    def __init__(self, x, *args, **kwargs):
        min_dim = max_dim = None
        self.name = kwargs["name"]
        for item in kwargs["noise_mix"]:
            ng_class = item[0] if isinstance(item, (tuple, list)) else item
            cmin, cmax = ng_class.MIN_DIMS, ng_class.MAX_DIMS
            min_dim = max(min_dim if min_dim is not None else cmin, cmin)
            max_dim = min(max_dim if max_dim is not None else cmax, cmax)
        self.MIN_DIMS = min_dim
        self.MAX_DIMS = max_dim
        super().__init__(x, *args, **kwargs)
        ng_list = []
        for ng_class, ng_class_kwargs, transform_fun in self.noise_mix:
            ng_kwargs = {k: v for k, v in kwargs.items() if k in self.pass_args}
            ng_list.append((ng_class(x, **ng_class_kwargs, **ng_kwargs), transform_fun))
        self.ng_list = ng_list

    def generate(self, *args):
        noise = None
        for ng, transform_fun in self.ng_list:
            new_noise = ng(*args)
            if transform_fun is not None:
                new_noise = transform_fun(new_noise)
            noise = new_noise if noise is None else noise.add_(new_noise)
        if self.output_fun is not None:
            noise = self.output_fun(noise)
        return noise
