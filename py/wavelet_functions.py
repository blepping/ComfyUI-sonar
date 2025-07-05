from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch

from .utils import fallback

if TYPE_CHECKING:
    from collections.abc import Sequence

try:
    import pytorch_wavelets as ptwav

    HAVE_WAVELETS = True
except ImportError:
    ptwav = None
    HAVE_WAVELETS = False


class Wavelet:
    DEFAULT_MODE = "periodization"
    DEFAULT_LEVEL = 3
    DEFAULT_WAVE = "haar"
    DEFAULT_USE_1D_DWT = False
    DEFAULT_USE_DTCWT = False
    DEFAULT_QSHIFT = "qshift_a"
    DEFAULT_BIORT = "near_sym_a"

    def __init__(
        self,
        *,
        wave: str = DEFAULT_WAVE,
        level: int = DEFAULT_LEVEL,
        mode: str = DEFAULT_MODE,
        use_1d_dwt: bool = DEFAULT_USE_1D_DWT,
        use_dtcwt: bool = DEFAULT_USE_DTCWT,
        biort: str = DEFAULT_BIORT,
        qshift: str = DEFAULT_QSHIFT,
        inv_wave: str | None = None,
        inv_mode: str | None = None,
        inv_biort: str | None = None,
        inv_qshift=None,
        device: str | torch.device | None = None,
    ):
        if not HAVE_WAVELETS:
            raise RuntimeError(
                "Wavelet noise requires the pytorch_wavelets package to be installed in your Python environment",
            )
        inv_wave = fallback(inv_wave, wave)
        inv_mode = fallback(inv_mode, mode)
        inv_biort = fallback(inv_biort, biort)
        inv_qshift = fallback(inv_qshift, qshift)
        if use_dtcwt:
            fwdfun, invfun = ptwav.DTCWTForward, ptwav.DTCWTInverse
        elif use_1d_dwt:
            fwdfun, invfun = ptwav.DWT1DForward, ptwav.DWT1DInverse
        else:
            fwdfun, invfun = ptwav.DWTForward, ptwav.DWTInverse
        if use_dtcwt:
            self._wavelet_forward = fwdfun(
                J=level,
                mode=mode,
                biort=biort,
                qshift=qshift,
            )
            self._wavelet_inverse = invfun(
                mode=inv_mode,
                biort=inv_biort,
                qshift=inv_qshift,
            )
        else:
            self._wavelet_forward = fwdfun(J=level, wave=wave, mode=mode)
            self._wavelet_inverse = invfun(wave=inv_wave, mode=inv_mode)
        if device is not None:
            self._wavelet_forward = self._wavelet_forward.to(device=device)
            self._wavelet_inverse = self._wavelet_inverse.to(device=device)

    def forward(
        self,
        t: torch.Tensor,
        *,
        forward_function: Callable | None = None,
    ) -> tuple[torch.Tensor, tuple]:
        return fallback(forward_function, self._wavelet_forward)(t)

    def inverse(
        self,
        yl: torch.Tensor,
        yh: tuple,
        *,
        inverse_function: Callable | None = None,
        two_step_inverse: bool = False,
    ) -> torch.Tensor:
        inverse_function = fallback(inverse_function, self._wavelet_inverse)
        if not two_step_inverse:
            return inverse_function((yl, yh))
        result = inverse_function((torch.zeros_like(yl), yh))
        result += inverse_function((
            yl,
            tuple(torch.zeros_like(yh_band) for yh_band in yh),
        ))
        return result

    def to(self, *args: list, **kwargs: dict) -> None:
        self._wavelet_forward = self._wavelet_forward.to(*args, **kwargs)
        self._wavelet_inverse = self._wavelet_inverse.to(*args, **kwargs)


def wavelet_scaling(
    yl: torch.Tensor,
    yh: Sequence,
    yl_scale: float | torch.Tensor,
    yh_scales: float | torch.Tensor | None,
    *,
    in_place: bool = False,
) -> tuple:
    if not in_place:
        yl = yl.clone()
        yh = tuple(yhband.clone() for yhband in yh)
    if yl_scale != 1.0:
        yl *= yl_scale
    if yh_scales is None or yh_scales == 1.0:
        return (yl, yh)
    if isinstance(yh_scales, (int, float)):
        yh_scales = (yh_scales,) * len(yh)
    # print("SCALES", self.yl_scale, yh_scales)
    for hscale, ht in zip(yh_scales, yh):
        # print(">> SCALING", hscale)
        if isinstance(hscale, (int, float)):
            ht *= hscale  # noqa: PLW2901
            continue
        for lidx in range(min(ht.shape[2], len(hscale))):
            # print(">>    SCALE IDX", lidx)
            ht[:, :, lidx, :, :] *= hscale[lidx]
    return (yl, yh)


def wavelet_blend(
    a: tuple,
    b: tuple,
    *,
    yl_factor: torch.Tensor | float,
    blend_function: Callable,
    yh_factor: torch.Tensor | float | None = None,
    yh_blend_function: Callable | None = None,
) -> tuple:
    if not isinstance(yl_factor, torch.Tensor):
        yl_factor = a[0].new_full((1,), yl_factor)
    if yh_factor is None:
        yh_factor = yl_factor
    elif not isinstance(yh_factor, torch.Tensor):
        yh_factor = a[0].new_full((1,), yh_factor)
    yh_blend_function = fallback(yh_blend_function, blend_function)
    return (
        blend_function(a[0], b[0], yl_factor),
        tuple(yh_blend_function(ta, tb, yh_factor) for ta, tb in zip(a[1], b[1])),
    )
