# Initial implementation by https://github.com/elias-gaeros/
# He also provided a lot of help with refactoring and other improvements. Thanks!
# (But if anything is broken in here, I'm almost certainly the one to blame.)

from __future__ import annotations

import math
import os
import random

import comfy
import folder_paths
import latent_preview
import torch
from comfy.k_diffusion.sampling import BrownianTreeNoiseSampler
from PIL import Image
from torch import Tensor

from ..noise import CustomNoiseItemBase
from ..utils import scale_noise
from .base import (
    NOISE_INPUT_TYPES_HINT,
    WILDCARD_NOISE,
    NoiseChainInputTypes,
    SonarCustomNoiseNodeBase,
    SonarInputTypes,
    SonarNormalizeNoiseNodeMixin,
)

PREVIEW_FORMAT = comfy.latent_formats.SD15()


def make_preview_result(img, result, prefix="sonar_temp"):
    output_dir = folder_paths.get_temp_directory()
    prefix_append = f"{prefix}_" + "".join(
        random.choice("abcdefghijklmnopqrstupvxyz")  # noqa: S311
        for x in range(5)
    )
    full_output_folder, filename, counter, subfolder, _ = (
        folder_paths.get_save_image_path(prefix_append, output_dir)
    )
    filename = f"{filename}_{counter:05}_.png"
    file_path = os.path.join(full_output_folder, filename)  # noqa: PTH118
    img.save(file_path, compress_level=1)

    return {
        "ui": {
            "images": [
                {"filename": filename, "subfolder": subfolder, "type": "temp"},
            ],
        },
        "result": result,
    }


class ChannelMixer:
    def __init__(self, channel_count, common_mode, channel_correlation):
        self.channel_count = channel_count
        self.common_mode = common_mode
        self.channel_correlation = channel_correlation
        self.mixer = self.build() if common_mode is not None else None

    def build(self):
        c = self.channel_count
        common_mode = self.common_mode
        correlation_count = c * (c - 1) // 2
        channel_correlation = self.channel_correlation[:correlation_count]
        channel_correlation = torch.cat(
            (
                channel_correlation * common_mode,
                torch.full(
                    (correlation_count - channel_correlation.numel(),),
                    common_mode,
                ),
            ),
        )
        channel_mixer = torch.eye(c).index_put_(
            tuple(torch.tril_indices(c, c, offset=-1)),
            channel_correlation,
        )
        channel_mixer += channel_mixer.tril(-1).mT
        channel_mixer = torch.linalg.ldl_factor(channel_mixer).LD
        dc = torch.diagonal_copy(channel_mixer)
        torch.diagonal(channel_mixer)[:] = 1.0
        channel_mixer *= dc.clamp_min(0).sqrt().unsqueeze(0)
        channel_mixer /= channel_mixer.norm(dim=1, keepdim=True)
        return channel_mixer

    def to(self, *args: list, **kwargs: dict):
        if self.mixer is not None:
            self.mixer = self.mixer.to(*args, **kwargs)
        return self

    def apply(self, noise, shape, copy=False):
        if self.mixer is None:
            return noise if not copy else noise.clone()
        b, c, h, w = shape
        if c != self.channel_count:
            raise ValueError("Channel count mismatch")
        noise = self.mixer @ noise.swapaxes(0, 1).reshape(c, -1)
        return noise.reshape(c, b, h, w).swapaxes(1, 0)

    def __call__(self, *args: list, **kwargs: dict):
        return self.apply(*args, **kwargs)


class PowerFilter:
    def __init__(
        self,
        *,
        min_freq=0.0,
        max_freq=0.7071,
        stretch=1.0,
        rotate=0.0,
        pnorm=2.0,
        alpha=0.0,
        scale=1.0,
        rel_bw=0.125,
        oversample=4,
        compose_with: PowerFilter | None = None,
        compose_mode="max",
    ):
        self.min_freq = min_freq
        self.max_freq = max(max_freq, min_freq)
        self.stretch = stretch
        self.rotate = rotate
        self.pnorm = pnorm
        self.alpha = alpha
        self.scale = scale
        self.rel_bw = rel_bw
        self.oversample = oversample
        self.compose_with = compose_with
        self.compose_mode = compose_mode

    def clone(self):
        fargs = {
            k: getattr(self, k)
            for k in (
                "min_freq",
                "max_freq",
                "stretch",
                "rotate",
                "pnorm",
                "alpha",
                "scale",
                "rel_bw",
                "oversample",
                "compose_mode",
            )
        }
        fargs["compose_with"] = (
            self.compose_with.clone() if self.compose_with is not None else None
        )
        return self.__class__(**fargs)

    @classmethod
    def compose(cls, a, b, compose_mode="max"):
        if a.shape != b.shape:
            raise ValueError("Filter compose size mismatch!")
        cf = {
            "max": torch.max,
            "min": torch.min,
            "add": torch.add,
            "sub": torch.sub,
            "mul": torch.mul,
        }.get(compose_mode, torch.max)
        return cf(a, b).clamp_(min=0.0)

    @classmethod
    def normalize(cls, op, shape, mix=1.0, normalization_factor=1.0):
        height, width = shape[-2:]
        hfreq_bins = width // 2 + 1

        # Flat unit gain frequency response
        if mix < 1.0:
            flat = torch.ones(1, 1, height, hfreq_bins)
            if mix <= 0.0:
                return flat
        if normalization_factor != 0:
            op *= torch.lerp(
                torch.scalar_tensor(1.0),
                1.0 / op.square().mean().sqrt(),
                normalization_factor,
            )
        if mix < 1.0:
            op = torch.lerp(flat, op, mix, out=op)
        return op

    def build(self, shape, override_oversample=None, composed=True):
        """Construct a band-pass * 1/f^alpha filter in rfft space."""
        oversample = (
            override_oversample if override_oversample is not None else self.oversample
        )
        rel_bw = self.rel_bw
        height, width = shape[-2:]
        hfreq_bins = width // 2 + 1

        # Start with an over-sampled fftshift(rfft2freq()) grid. uses complex
        # numbers for convenient 2d rotation (unrelated to the fft complex phase
        # space)
        fc = torch.complex(
            # real-fftfreq
            torch.linspace(0, 0.5, oversample * hfreq_bins),
            # normal fftfreq
            torch.linspace(
                -(height // 2) / height,
                ((height - 1) // 2) / height,
                oversample * height,
            ).unsqueeze(1),
        )
        # Rotate, stretch and p-norm
        if abs(self.rotate) >= 1e-3:
            fc *= torch.exp(1.0j * torch.deg2rad(torch.scalar_tensor(self.rotate)))
        if self.stretch > 1.0:
            fc.real *= self.stretch
        else:
            fc.imag *= 1.0 / self.stretch
        if abs(self.pnorm - 2.0) < 1e-3:
            d = fc.abs()
        else:
            d = (
                torch.view_as_real(fc)
                .abs()
                .pow(self.pnorm)
                .sum(-1)
                .pow(1.0 / self.pnorm)
            )

        # filter gain function
        op = torch.empty_like(d)
        m_highpass = d >= self.min_freq
        m_lowpass = d < self.max_freq
        m_band = m_highpass & m_lowpass
        # 1 / f^alpha for the band-pass region
        op[m_band] = d[m_band].pow(-self.alpha)
        # easing gaussian (TODO: try cosine windows)
        m_lowpass = ~m_lowpass
        op[m_lowpass] = math.pow(self.max_freq, -self.alpha) * torch.exp(
            -(d[m_lowpass] - self.max_freq).square() / (rel_bw * self.max_freq) ** 2,
        )
        if self.min_freq > 0.0:
            m_highpass = ~m_highpass
            op[m_highpass] = math.pow(self.min_freq, -self.alpha) * torch.exp(
                -(d[m_highpass] - self.min_freq).square()
                / (rel_bw * self.min_freq) ** 2,
            )
        op = torch.nn.functional.interpolate(
            op[None, None, ...],
            (height, hfreq_bins),
            mode="bilinear",
            align_corners=True,
        )
        op = op.roll(-(height // 2), -2)  # ifftshift
        if self.alpha > 0:
            # In general, the mean offset should be kept as is, sampled from
            # N(0, 1 / sqrt(H*W) ). However, gain goes to inf when alpha>0.
            op[..., 0, 0] = 0
        if self.scale != 1.0:
            op *= self.scale
        if composed and self.compose_with is not None:
            return self.compose(
                op,
                self.compose_with.build(shape, override_oversample=override_oversample),
                self.compose_mode,
            )
        return op

    def preview(
        self,
        size=(128, 128),
        mix=1.0,
        normalization_factor=1.0,
        raw=False,
        kernel_gain=1 / 3,
        filter_gain=1 / 3,
    ):
        shape = (1, 4, *size)
        filter_rfft = self.__class__.normalize(
            self.build(size),
            shape,
            mix=mix,
            normalization_factor=normalization_factor,
        )
        filter_fft = rfft2_to_fft2(filter_rfft)
        kernel = torch.fft.irfft2(filter_rfft, s=size, norm="ortho")
        kernel = kernel.roll((size[0] // 2, size[1] // 2), (-2, -1))
        img = (
            filter_fft.mul_(filter_gain).tanh_().mul_(256.0),
            kernel.mul_(kernel_gain).tanh_().add_(1.0).mul_(128.0),
        )
        if raw:
            return img
        img = torch.cat(img, dim=-1).clamp(0, 255).to(torch.uint8)
        return Image.fromarray(img[0, 0].numpy())


class PowerNoiseItem(CustomNoiseItemBase):
    def __init__(
        self,
        factor,
        *,
        channel_correlation,
        power_filter=None,
        **kwargs: dict,
    ):
        if isinstance(channel_correlation, str):
            channel_correlation = torch.tensor(
                tuple(
                    float(val)
                    for val in (val.strip() for val in channel_correlation.split(","))
                    if val
                ),
                device="cpu",
                dtype=torch.float,
            )
        if power_filter is None:
            fargs = {
                k: kwargs.pop(k)
                for k in ("min_freq", "max_freq", "stretch", "rotate", "pnorm", "alpha")
                if k in kwargs
            }
            power_filter = PowerFilter(**fargs)
        super().__init__(
            factor,
            power_filter=power_filter,
            channel_correlation=channel_correlation,
            **kwargs,
        )

    def make_filter(self, shape, oversample=None):
        return PowerFilter.normalize(
            self.power_filter.build(shape, override_oversample=oversample),
            shape,
            mix=self.mix,
            normalization_factor=getattr(self, "filter_norm_factor", 1.0),
        )

    def make_noise_sampler_internal(
        self,
        x: Tensor,
        noise_sampler,
        filter_rfft,
        normalized=True,
    ):
        shape = x.shape
        device = x.device
        time_brownian = self.time_brownian

        channel_mixer = ChannelMixer(
            shape[1],
            self.common_mode,
            self.channel_correlation,
        ).to(device, non_blocking=True)

        def sampler(sigma, sigma_next):
            noise = noise_sampler(sigma, sigma_next).to(device)
            noise_rfft = (
                torch.fft.rfft2(noise, norm="ortho") if time_brownian else noise
            )
            noise = torch.fft.irfft2(
                noise_rfft.mul_(filter_rfft),
                s=shape[-2:],
                norm="ortho",
            )
            noise = channel_mixer(noise, shape)
            return scale_noise(noise, self.factor, normalized=normalized)

        return sampler

    def make_noise_sampler(
        self,
        x: Tensor,
        sigma_min: float | None,
        sigma_max: float | None,
        *,
        seed: int | None,
        cpu: bool = True,
        normalized=True,
    ):
        shape, device = x.shape, x.device
        filter_rfft = self.make_filter(shape).to(device, non_blocking=True)
        if self.time_brownian:
            if sigma_min is None:
                raise ValueError(
                    "time correlated brownian mode is valid only for stochastic samplers",
                )
            noise_sampler = BrownianTreeNoiseSampler(
                x,
                sigma_min,
                sigma_max,
                seed=seed,
                cpu=cpu,
            )
        else:

            def noise_sampler(_s, _sn):
                return torch.randn(
                    (*shape[:-1], filter_rfft.shape[-1]),
                    dtype=torch.complex64,
                    device=device,
                )

        return self.make_noise_sampler_internal(
            x,
            noise_sampler,
            filter_rfft,
            normalized=normalized,
        )

    def preview(
        self,
        size=(128, 128),
        noise=None,
        kernel_gain=1 / 3,
        filter_gain=1 / 3,
    ):
        filter_rfft = self.make_filter(size, oversample=1)
        if noise is None:
            noise = torch.fft.irfft2(
                filter_rfft
                * torch.randn(
                    filter_rfft.shape,
                    dtype=torch.complex64,
                    generator=torch.Generator().manual_seed(0),
                ),
                s=size,
                norm="ortho",
            )
        else:
            noise_rfft = torch.fft.rfft2(noise, norm="ortho")
            noise = torch.fft.irfft2(
                noise_rfft.mul_(filter_rfft),
                s=noise.shape[-2:],
                norm="ortho",
            )
        filter_preview = self.power_filter.preview(
            size=size,
            normalization_factor=getattr(self, "filter_norm_factor", 1.0),
            filter_gain=filter_gain,
            kernel_gain=kernel_gain,
            raw=True,
        )
        img = (
            torch.cat(
                (
                    *filter_preview,
                    noise.mul_(1 / 3).tanh_().add_(1.0).mul_(128.0),
                ),
                dim=-1,
            )
            .clamp(0, 255)
            .to(torch.uint8)
        )
        return Image.fromarray(img[0, 0].numpy())


def rfft2_to_fft2(x):
    """Apply hermitian-summetry to reconstruct the second half of a fft.

    Only for previews.
    """
    height, width = x.shape[-2:]
    x_r = x.roll(height // 2, -2)  # torch.fft.fftshift(x, -2)
    x_l = x_r[..., 1 : -1 if width & 1 else None]
    x_l = torch.flip(x_l.conj(), dims=(-2, -1))
    if height & 1 == 0:
        x_l = x_l.roll(1, -2)
    return torch.cat((x_l, x_r), dim=-1)


class PowerFilterNoiseItem(PowerNoiseItem):
    def __init__(
        self,
        factor,
        *,
        noise,
        normalize_noise,
        normalize_result,
        **kwargs: dict,
    ):
        super().__init__(
            factor,
            noise=noise.clone(),
            normalize_noise=normalize_noise,
            normalize_result=normalize_result,
            **kwargs,
        )

    def clone_key(self, k):
        if k == "noise":
            return self.noise.clone()
        return super().clone_key(k)

    def make_noise_sampler(
        self,
        x: Tensor,
        sigma_min: float | None,
        sigma_max: float | None,
        *,
        seed: int | None,
        cpu: bool = True,
        normalized=True,
    ):
        shape, device = x.shape, x.device
        normalize_noise = self.get_normalize("normalize_noise", False)  # noqa: FBT003
        normalize_result = self.get_normalize("normalize_result", normalized)
        filter_rfft = self.make_filter(shape).to(device, non_blocking=True)
        noise_sampler = self.noise.make_noise_sampler(
            x,
            sigma_min,
            sigma_max,
            seed,
            cpu,
            normalized=normalize_noise,
        )

        return self.make_noise_sampler_internal(
            x,
            noise_sampler,
            filter_rfft,
            normalized=normalize_result,
        )

    def preview(self, size=(128, 128)):
        if getattr(self, "preview_type", None) != "custom":
            return super().preview(size=size)
        torch.manual_seed(0)
        x = torch.randn((1, 4, *size), dtype=torch.float, device="cpu")
        ns = self.noise.make_noise_sampler(
            x,
            torch.scalar_tensor(0.0),
            torch.scalar_tensor(14.0),
            0,
            True,  # noqa: FBT003
            normalized=self.normalize_noise is True,
        )
        filtered_ns = self.make_noise_sampler_internal(
            x,
            ns,
            self.make_filter(x.shape),
            self.normalize_result in {True, None},
        )
        filtered_noise = filtered_ns(
            torch.scalar_tensor(14.0),
            torch.scalar_tensor(10.0),
        )
        previewer = latent_preview.get_previewer(None, PREVIEW_FORMAT)
        default_preview = super().preview(size=size).convert("RGB")
        preview = previewer.decode_latent_to_preview(filtered_noise.cpu())
        default_preview.paste(
            preview.resize((size[-1], size[-2])),
            box=(size[-1] * 2, 0),
        )
        return default_preview


class SonarPowerNoiseNode(SonarCustomNoiseNodeBase):
    DESCRIPTION = "Custom noise type that applies a filter to generated noise."

    INPUT_TYPES = (
        NoiseChainInputTypes()
        .req_bool_time_brownian(
            tooltip="Controls whether brownian noise is used when mix isn't 1.0.",
        )
        .req_float_alpha(
            default=0.0,
            min=-5.0,
            max=5.0,
            tooltip="Values above 0 will amplify low frequencies, negative values will amplify high frequencies.",
        )
        .req_float_max_freq(
            default=0.7071,
            min=0.0,
            max=0.7071,
            tooltip="Maximum frequency to pass through the filter.",
        )
        .req_float_min_freq(
            default=0.0,
            min=0.0,
            max=0.7071,
            tooltip="Minimum frequency to pass through the filter.",
        )
        .req_float_stretch(
            default=1.0,
            min=0.01,
            max=100.0,
            tooltip="Stretches the filter's shape by the specified factor.",
        )
        .req_float_rotate(
            default=0.0,
            min=-90.0,
            max=90.0,
            step=5.0,
            tooltip="Rotates the filter.",
        )
        .req_float_pnorm(
            default=2.0,
            min=0.125,
            max=100.0,
            step=0.1,
            tooltip="Factor used for cushioning the band-pass region.",
        )
        .req_floatpct_mix(
            default=1.0,
            tooltip="Controls the ratio of filtered noise. For example, 0.75 means 75% noise with the filter effects applied, 25% raw noise.",
        )
        .req_float_common_mode(
            default=0.0,
            min=-100.0,
            max=100.0,
            tooltip="Attempts to desaturate the latent by injecting the average across channels (controlled by channel_correction). Applied after mix.",
        )
        .req_string_channel_correlation(
            default="1, 1, 1, 1, 1, 1",
            tooltip="Comma-separated list of channel correlation strengths.",
        )
        .req_field_preview(
            ("none", "no_mix", "mix"),
            default="none",
            tooltip="When enabled, displays a preview of the filter shape and a sample of noise. Mix - previews noise after mix is applied. no_mix - only previews the filtered noise.",
        )
    )

    @classmethod
    def get_item_class(cls):
        return PowerNoiseItem

    def go(
        self,
        preview="none",
        **kwargs: dict,
    ):
        result = super().go(**kwargs)
        if preview == "none":
            return result
        if preview == "no_mix":
            kwargs["mix"] = 1.0
        img = self.get_item_class()(preview_type=preview, **kwargs).preview()
        return make_preview_result(img, result)


class SonarPowerFilterNoiseNode(SonarPowerNoiseNode, SonarNormalizeNoiseNodeMixin):
    DESCRIPTION = "Custom noise type that allows applying a Power Filter to another custom noise generator."

    @classmethod
    def INPUT_TYPES(cls):
        result = super().INPUT_TYPES()
        for k in (
            "min_freq",
            "max_freq",
            "stretch",
            "rotate",
            "pnorm",
            "alpha",
            "time_brownian",
        ):
            del result["required"][k]
        result["required"] |= {
            "sonar_custom_noise": (
                WILDCARD_NOISE,
                {
                    "tooltip": f"Custom noise type to filter.\n{NOISE_INPUT_TYPES_HINT}",
                },
            ),
            "sonar_power_filter": (
                "SONAR_POWER_FILTER",
                {
                    "tooltip": "Filter to use.",
                },
            ),
            "filter_norm_factor": (
                "FLOAT",
                {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "round": False,
                    "tooltip": "Normalization factor applied to the specified filter. 1.0 means 100% normalized.",
                },
            ),
            "normalize_result": (
                ("default", "forced", "disabled"),
                {
                    "tooltip": "Controls whether the final result is normalized to 1.0 strength.",
                },
            ),
            "normalize_noise": (
                ("default", "forced", "disabled"),
                {
                    "tooltip": "Controls whether the generated noise is normalized to 1.0 strength.",
                },
            ),
        }
        result["required"]["preview"] = (
            (*result["required"]["preview"][0], "custom"),
            {
                "tooltip": "When enabled, displays a preview of the filter shape and a sample of noise. Mix - previews noise after mix is applied. no_mix - only previews the filtered noise. custom - Like no_mix, but will use a latent previewer to display a color preview of the generated noise. Works best when previewer is set to TAESD.",
            },
        )
        return result

    @classmethod
    def get_item_class(cls):
        return PowerFilterNoiseItem

    def go(
        self,
        factor,
        sonar_custom_noise,
        sonar_power_filter,
        filter_norm_factor,
        normalize_noise,
        normalize_result,
        preview="none",
        **kwargs: dict,
    ):
        return super().go(
            factor=factor,
            noise=sonar_custom_noise,
            normalize_noise=self.get_normalize(normalize_noise),
            normalize_result=self.get_normalize(normalize_result),
            preview=preview,
            time_brownian=True,
            power_filter=sonar_power_filter,
            filter_norm_factor=filter_norm_factor,
            **kwargs,
        )


class SonarPowerFilterNode:
    RETURN_TYPES = ("SONAR_POWER_FILTER",)
    CATEGORY = "advanced/noise"
    FUNCTION = "go"

    @classmethod
    def INPUT_TYPES(cls):
        include_keys = {"alpha", "max_freq", "min_freq", "stretch", "rotate", "pnorm"}
        return {
            "required": {
                k: v
                for k, v in SonarPowerNoiseNode.INPUT_TYPES()["required"].items()
                if k in include_keys
            }
            | {
                "oversample": (
                    "INT",
                    {
                        "default": 4,
                        "min": 1,
                        "max": 128,
                        "tooltip": "Oversampling factor used for the filter size.",
                    },
                ),
                "blur": (
                    "FLOAT",
                    {
                        "default": 0.125,
                        "min": -10.0,
                        "max": 10.0,
                        "step": 0.01,
                        "round": False,
                        "tooltip": "Slightly blurs the filter to reduce artifacts.",
                    },
                ),
                "scale": (
                    "FLOAT",
                    {
                        "default": 1,
                        "min": -100.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": False,
                        "tooltip": "Scales the filter to the specified strength. May be negative.",
                    },
                ),
                "compose_mode": (
                    ("max", "min", "add", "sub", "mul"),
                    {
                        "tooltip": "Controls composition of the option attached filter. For example, when set to MUL the result will be this filter multiplied by the attached filter. No effect if the optional filter input is not attached.",
                    },
                ),
            },
            "optional": {
                "power_filter_opt": ("SONAR_POWER_FILTER",),
            },
        }

    @classmethod
    def go(
        cls,
        min_freq=0.0,
        max_freq=0.7071,
        stretch=1.0,
        rotate=0.0,
        pnorm=2.0,
        alpha=0.0,
        blur=0.125,
        oversample=4,
        scale=1.0,
        compose_mode="max",
        power_filter_opt=None,
    ):
        return (
            PowerFilter(
                min_freq=min_freq,
                max_freq=max_freq,
                stretch=stretch,
                rotate=rotate,
                pnorm=pnorm,
                alpha=alpha,
                scale=scale,
                rel_bw=blur,
                oversample=oversample,
                compose_mode=compose_mode,
                compose_with=power_filter_opt,
            ),
        )


class SonarPreviewFilterNode:
    DESCRIPTION = "Allows previewing a Power Filter."
    RETURN_TYPES = ("SONAR_POWER_FILTER",)
    CATEGORY = "advanced/noise"
    FUNCTION = "go"
    OUTPUT_NODE = True

    INPUT_TYPES = (
        SonarInputTypes()
        .req_field_sonar_power_filter(
            "SONAR_POWER_FILTER",
            tooltip="Power Filter to preview.",
        )
        .req_float_filter_gain(
            default=1 / 3,
            min=0.0,
            tooltip="Gain factor applied to the filter part of the preview.",
        )
        .req_float_kernel_gain(
            default=1 / 3,
            min=0.0,
            tooltip="Gain factor applied to the kernel part of the preview.",
        )
        .req_floatpct_norm_factor(
            default=1.0,
            tooltip="Normalization factor applied to the filter before previewing. 1.0 means 100% normalized.",
        )
        .req_field_preview_size(
            (
                "128x128",
                "256x256",
                "384x256",
                "256x384",
                "768x512",
                "512x768",
                "768x768",
                "128x127",
                "127x128",
            ),
            default="128x128",
            tooltip="Controls the size of the generated preview. Note: Sizes are in latent pixels. For most models, one latent pixel equals eight pixels",
        )
    )

    @classmethod
    def go(
        cls,
        sonar_power_filter,
        filter_gain=1 / 3,
        kernel_gain=1 / 3,
        norm_factor=1.0,
        preview_size="256x256",
    ):
        filt = sonar_power_filter.clone()
        filt.preview_type = "custom"
        preview_size = tuple(int(val) for val in preview_size.split("x", 1))
        return make_preview_result(
            filt.preview(
                size=(preview_size[1], preview_size[0]),
                filter_gain=filter_gain,
                kernel_gain=kernel_gain,
                normalization_factor=norm_factor,
            ),
            (filt,),
        )


NODE_CLASS_MAPPINGS = {
    "SonarPowerNoise": SonarPowerNoiseNode,
    "SonarPowerFilterNoise": SonarPowerFilterNoiseNode,
    "SonarPowerFilter": SonarPowerFilterNode,
    "SonarPreviewFilter": SonarPreviewFilterNode,
}
