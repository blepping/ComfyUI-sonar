# ruff: noqa: ANN002, ANN003
from __future__ import annotations

from typing import Callable

import torch

from .. import utils
from .base import NoiseGenerator

F = torch.nn.functional


# With help from ChatGPT.
class VoronoiNoiseGenerator(NoiseGenerator):
    name = "voronoi"
    MIN_DIMS = 4
    MAX_DIMS = 4

    voronoi_distance_modes = frozenset((
        "angle_sigmoid",
        "angle_tanh",
        "angle",
        "chebyshev",
        "euclidean",
        "fractal_norm",
        "fuzz",
        "manhatten",
        "minkowski",
        "quadratic",
        "weight",
    ))

    voronoi_result_modes = frozenset((
        "cellid",
        "diff",
        "diff2",
        "f",
        "f1",
        "f2",
        "f3",
        "f4",
        "fractal_norm",
        "fuzz",
        "inv_f",
        "inv_f1",
        "inv_f2",
        "inv_f3",
        "inv_f4",
        "gradient_magnitude",
        "median_distance",
        "ridge",
        "softmin",
    ))

    @classmethod
    def ng_params(cls, *, no_super: bool = False):
        result = {
            "n_points": (32,),
            "distance_mode": ("euclidean",),
            "z_initial": 0.0,
            "z_increment": 1.0,
            "z_max": 100000,
            "z_max_mode": "reset",
            # None or numeric
            "z_range": None,
            "result_mode": ("f1",),
            "octaves": 1,
            # same_features or new_features
            "octave_mode": "same_features",
            "lacunarity": 2.0,  # scale increase per octave
            "gain": 0.5,  # amplitude decrease per octave
            "initial_amplitude": 1.0,
            "initial_scale": 1.0,
            "noise_sampler_factory": None,
            "normalized": False,
        }
        return result if no_super else super().ng_params() | result

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_points = self.grid_xyz = None
        self.noise_samplers = None
        self.n_points = tuple(max(2, val) for val in self.n_points)

    def voronoi_reset(self, *args):
        self.z_curr = self.z_initial
        octave_range = tuple(
            range(self.octaves if self.octave_mode == "new_features" else 1),
        )
        if self.noise_sampler_factory is not None and self.noise_samplers is None:
            self.noise_samplers = tuple(
                self.noise_sampler_factory.make_noise_sampler(
                    torch.zeros(
                        self.batch,
                        self.channels,
                        self.n_points[octave % len(self.n_points)],
                        3,
                        device=self.gen_device,
                        dtype=self.dtype,
                    ),
                    cpu=self.cpu,
                    normalized=False,
                )
                for octave in octave_range
            )
        self.feature_points = tuple(
            (
                torch.rand(
                    self.batch,
                    self.channels,
                    self.n_points[octave % len(self.n_points)],
                    3,
                    device=self.gen_device,
                    dtype=self.dtype,
                )
                if self.noise_samplers is None
                else utils.normalize_to_scale(
                    self.noise_samplers[octave](*args),
                    target_min=0.0,
                    target_max=1.0,
                    dim=(-1, -2),
                )
            ).to(device=self.device)
            for octave in octave_range
        )
        if self.grid_xyz is not None:
            return
        y = torch.linspace(
            0,
            self.height - 1,
            self.height,
            device=self.device,
            dtype=self.dtype,
        )
        x = torch.linspace(
            0,
            self.width - 1,
            self.width,
            device=self.device,
            dtype=self.dtype,
        )
        self.grid_xyz = torch.stack(
            torch.meshgrid(y, x, indexing="ij"),
            dim=-1,
        ) / torch.tensor(
            (self.height, self.width),
            device=self.device,
        )

    def get_feature_points(self, octave: int) -> torch.Tensor:
        result = self.feature_points[octave % len(self.feature_points)]
        odd_octave = (octave % 2) == 1
        om = self.octave_mode
        if (om == "same_invert_odd" and odd_octave) or (
            om == "same_invert_even" and not odd_octave
        ):
            return 1.0 - result
        if octave > 0 and om in {"same_roll_chan_up", "same_roll_chan_down"}:
            return torch.roll(
                result,
                (-1 if om == "same_roll_chan_up" else 1) * (octave % 3),
                dims=(1,),
            )
        if octave > 0 and om in {"same_roll_dir_up", "same_roll_dir_down"}:
            return torch.roll(
                result,
                (-1 if om == "same_roll_dir_up" else 1) * (octave % 3),
                dims=(3,),
            )
        return result

    def get_distance_mode(self, octave: int) -> torch.Tensor:
        return self.distance_mode[octave % len(self.distance_mode)]

    def get_result_mode(self, octave: int) -> torch.Tensor:
        return self.result_mode[octave % len(self.result_mode)]

    def voronoi_call_mode(
        self,
        name: str,
        *,
        result: bool,
        args: list | tuple = (),
        kwargs: dict | None = None,
    ) -> torch.Tensor:
        name = name.strip().lower()
        modes = self.voronoi_result_modes if result else self.voronoi_distance_modes
        mode_label = "result" if result else "distance"
        if name not in modes:
            errstr = f"Bad Voronoi {mode_label} mode {name}"
            raise ValueError(errstr)
        kwargs = (
            {}
            if kwargs is None
            else {
                k[1:] if k.startswith("_") and len(k) > 1 else k: v
                for k, v in kwargs.items()
            }
        )
        return getattr(self, f"_voronoi_{mode_label}_{name}")(*args, **kwargs)

    @staticmethod
    def _voronoi_distance_euclidean(d: torch.Tensor, **_kwargs) -> torch.Tensor:
        return d.pow(2).sum(dim=-1).sqrt_()

    @staticmethod
    def _voronoi_distance_manhatten(d: torch.Tensor, **_kwargs) -> torch.Tensor:
        return d.pow(2).sum(dim=-1).sqrt_()

    @staticmethod
    def _voronoi_distance_chebyshev(d: torch.Tensor, **_kwargs) -> torch.Tensor:
        return d.abs().amax(dim=-1)

    @staticmethod
    def _voronoi_distance_minkowski(
        d: torch.Tensor,
        *,
        p: float | str = 3.0,
        **_kwargs,
    ) -> torch.Tensor:
        p = float(p)
        return d.abs().pow(p).sum(dim=-1).pow(1 / p)

    @staticmethod
    def _voronoi_distance_quadratic(d: torch.Tensor, **_kwargs) -> torch.Tensor:
        return d.pow(2).sum(dim=-1)

    @staticmethod
    def _voronoi_distance_angle(
        d: torch.Tensor,
        *,
        idx: int | str = 2,
        **_kwargs,
    ) -> torch.Tensor:
        return (
            torch.nn.functional.normalize(d, dim=-1)[..., int(idx)]
            .clamp_(-1.0, 1.0)
            .acos_()
        )

    @staticmethod
    def _voronoi_distance_angle_tanh(
        d: torch.Tensor,
        *,
        idx: int | str = 2,
        **_kwargs,
    ) -> torch.Tensor:
        return torch.nn.functional.normalize(d, dim=-1)[..., int(idx)].tanh_().acos_()

    @staticmethod
    def _voronoi_distance_angle_sigmoid(
        d: torch.Tensor,
        *,
        idx: int | str = 2,
        **_kwargs,
    ) -> torch.Tensor:
        return (
            torch.nn.functional.normalize(d, dim=-1)[..., int(idx)]
            .sigmoid_()
            .mul_(2)
            .sub_(1)
            .acos_()
        )

    def _voronoi_distance_weight(
        self,
        d: torch.Tensor,
        *args,
        name: str = "euclidean",
        h: float | str = 1.0,
        w: float | str = 1.0,
        z: float | str = 0.25,
        **kwargs,
    ) -> torch.Tensor:
        weights = d.new_tensor((float(h), float(w), float(z)))
        return self.voronoi_call_mode(
            name,
            result=False,
            args=(d * weights, *args),
            kwargs=kwargs,
        )

    def _voronoi_distance_fractal_norm(
        self,
        d: torch.Tensor,
        *args,
        name: str = "euclidean",
        mode: str = "sin",
        scale: str | float = 0.1,
        multiplier: str | float = 10.0,
        **kwargs,
    ) -> torch.Tensor:
        if mode == "sin":
            fun = torch.sin
        elif mode == "cos":
            fun = torch.cos
        else:
            raise ValueError(
                "Bad mode parameter for fractal_norm distance mode, must be one of: sin, cos",
            )
        adjustment = float(scale) * fun(d * float(multiplier))
        return self.voronoi_call_mode(
            name,
            result=False,
            args=(d + adjustment, *args),
            kwargs=kwargs,
        )

    def _voronoi_distance_fuzz(
        self,
        *args,
        name: str = "euclidean",
        fuzz: float | str = 0.25,
        **kwargs,
    ) -> torch.Tensor:
        fuzz = float(fuzz)
        result = self.voronoi_call_mode(name, result=False, args=args, kwargs=kwargs)
        rmin, rmax = result.aminmax()
        fuzz = max(abs(rmin.item()), abs(rmax.item())) * fuzz
        result += (
            torch.rand(result.shape, device=self.gen_device, dtype=result.dtype)
            .mul_(fuzz * 2)
            .sub_(fuzz)
            .to(device=result.device)
        )
        return utils.normalize_to_scale(result, rmin.item(), rmax.item(), dim=(-2, -1))

    @staticmethod
    def _voronoi_result_f(
        _d: torch.Tensor,
        *,
        get_sorted: Callable,
        idx: int | str = 0,
        **_kwargs,
    ) -> torch.Tensor:
        return get_sorted()[..., int(idx)]

    def _voronoi_result_f1(self, *args, **kwargs) -> torch.Tensor:
        return self._voronoi_result_f(*args, idx=0, **kwargs)

    def _voronoi_result_f2(self, *args, **kwargs) -> torch.Tensor:
        return self._voronoi_result_f(*args, idx=1, **kwargs)

    def _voronoi_result_f3(self, *args, **kwargs) -> torch.Tensor:
        return self._voronoi_result_f(*args, idx=2, **kwargs)

    def _voronoi_result_f4(self, *args, **kwargs) -> torch.Tensor:
        return self._voronoi_result_f(*args, idx=3, **kwargs)

    def _voronoi_result_inv_f(self, *args, eps=1e-06, **kwargs) -> torch.Tensor:
        return 1.0 / (self._voronoi_result_f(*args, **kwargs) + eps)

    def _voronoi_result_inv_f1(self, *args, **kwargs) -> torch.Tensor:
        return self._voronoi_result_inv_f(*args, idx=0, **kwargs)

    def _voronoi_result_inv_f2(self, *args, **kwargs) -> torch.Tensor:
        return self._voronoi_result_inv_f(*args, idx=1, **kwargs)

    def _voronoi_result_inv_f3(self, *args, **kwargs) -> torch.Tensor:
        return self._voronoi_result_inv_f(*args, idx=2, **kwargs)

    def _voronoi_result_inv_f4(self, *args, **kwargs) -> torch.Tensor:
        return self._voronoi_result_inv_f(*args, idx=3, **kwargs)

    def _voronoi_result_diff(
        self,
        *args,
        idx1: int | str = 0,
        idx2: int | str = 1,
        **kwargs,
    ) -> torch.Tensor:
        val1, val2 = (
            self._voronoi_result_f(*args, idx=i, **kwargs) for i in (idx1, idx2)
        )
        return val2 - val1

    def _voronoi_result_diff2(
        self,
        *args,
        idx1: int | str = 0,
        idx2: int | str = 1,
        **kwargs,
    ) -> torch.Tensor:
        val1, val2 = (
            self._voronoi_result_f(*args, idx=i, **kwargs) for i in (idx1, idx2)
        )
        return (val2 - val1) / (val2 + val1 + 1e-06)

    @staticmethod
    def _voronoi_result_cellid(d, *_args, **_kwargs) -> torch.Tensor:
        cellids = d.argmin(dim=-1).to(dtype=d.dtype)
        return (cellids / cellids.max()).add_(1.0)

    def _voronoi_result_ridge(
        self,
        *args,
        name: str = "diff",
        exp: float | str = -10.0,
        **kwargs,
    ) -> torch.Tensor:
        return 1.0 - (
            float(exp)
            * self.voronoi_call_mode(name, result=True, args=args, kwargs=kwargs)
        )

    @staticmethod
    def _voronoi_result_median_distance(
        *_args,
        get_sorted: Callable,
        **_kwargs,
    ) -> torch.Tensor:
        return get_sorted().median(dim=-1).values

    @staticmethod
    def _voronoi_result_softmin(
        d: torch.Tensor,
        *_args,
        temperature=50.0,
        use_sorted=None,
        d_orig: torch.Tensor,
        get_sorted: Callable,
        **_kwargs,
    ) -> torch.Tensor:
        d_norm = d_orig.norm(dim=-1)
        soft_weights = F.softmax(-d_norm * float(temperature), dim=-1)
        eff_d = get_sorted() if use_sorted is not None else d
        return (eff_d * soft_weights).sum(dim=-1)

    def _voronoi_result_gradient_magnitude(
        self,
        *args,
        name1: str = "f4",
        name2: str = "f4",
        pad_mode: str = "replicate",
        **kwargs,
    ) -> torch.Tensor:
        r1 = self.voronoi_call_mode(name1, result=True, args=args, kwargs=kwargs)
        r1_padded = F.pad(r1, (1, 1, 1, 1), mode=pad_mode)
        if name2 != name1:
            r2 = self.voronoi_call_mode(name2, result=True, args=args, kwargs=kwargs)
            r2_padded = F.pad(r2, (1, 1, 1, 1), mode=pad_mode)
        else:
            r2 = r1
            r2_padded = r1_padded
        dx = r1_padded[..., 1:-1, 2:] - r2_padded[..., 1:-1, :-2]
        dy = r1_padded[..., 2:, 1:-1] - r2_padded[..., :-2, 1:-1]
        return (dx**2 + dy**2).sqrt_()

    def _voronoi_result_fractal_norm(
        self,
        d: torch.Tensor,
        *args,
        name: str = "diff",
        mode: str = "sin",
        scale: str | float = 0.1,
        multiplier: str | float = 10.0,
        **kwargs,
    ) -> torch.Tensor:
        if mode == "sin":
            fun = torch.sin
        elif mode == "cos":
            fun = torch.cos
        else:
            raise ValueError(
                "Bad mode parameter for fractal_norm result mode, must be one of: sin, cos",
            )
        d_adjusted = float(scale) * fun(d * float(multiplier))
        my_d_sorted = None

        def my_get_sorted():
            nonlocal my_d_sorted
            if my_d_sorted is not None:
                return my_d_sorted
            my_d_sorted = d_adjusted.sort(dim=-1).values
            return my_d_sorted

        return self.voronoi_call_mode(
            name,
            result=True,
            args=(d_adjusted, *args),
            kwargs=kwargs | {"get_sorted": my_get_sorted},
        )

    def _voronoi_result_fuzz(
        self,
        *args,
        name: str = "f1",
        fuzz: float | str = 0.25,
        **kwargs,
    ) -> torch.Tensor:
        fuzz = float(fuzz)
        result = self.voronoi_call_mode(name, result=True, args=args, kwargs=kwargs)
        rmin, rmax = result.aminmax()
        fuzz = max(abs(rmin.item()), abs(rmax.item())) * fuzz
        result += (
            torch.rand(result.shape, device=self.gen_device, dtype=result.dtype)
            .mul_(fuzz * 2)
            .sub_(fuzz)
            .to(device=result.device)
        )
        return utils.normalize_to_scale(result, rmin.item(), rmax.item(), dim=(-2, -1))

    def voronoi_distance(self, d: torch.Tensor, octave: int) -> torch.Tensor:
        modes = self.get_distance_mode(octave).split("+")
        result_scale_base = 1.0 / len(modes)
        result = None
        for mode in modes:
            if ":" in mode:
                mode_name, *mode_rest = mode.split(":")
                mode_kwargs = dict(
                    tuple(val.strip() for val in di.split("=", 1)) for di in mode_rest
                )
                result_scale = result_scale_base * float(mode_kwargs.pop("dscale", 1.0))
            else:
                mode_name = mode
                mode_kwargs = {}
                result_scale = result_scale_base
            curr_result = self.voronoi_call_mode(
                mode_name,
                result=False,
                args=(d,),
                kwargs=mode_kwargs,
            ).mul_(result_scale)
            result = curr_result if result is None else result.add_(curr_result)
        return result

    def voronoi_result(
        self,
        d: torch.Tensor,
        d_orig: torch.Tensor,
        *,
        octave: int,
    ) -> torch.Tensor:
        modes = self.get_result_mode(octave).split("+")
        result_scale_base = 1.0 / len(modes)
        result = None
        d_sorted = None

        def get_sorted():
            nonlocal d_sorted
            if d_sorted is not None:
                return d_sorted
            d_sorted = d.sort(dim=-1).values
            return d_sorted

        base_kwargs = {
            "d_orig": d_orig,
            "get_sorted": get_sorted,
        }
        for mode in modes:
            if ":" in mode:
                mode_name, *mode_rest = mode.split(":")
                mode_kwargs = dict(
                    tuple(v.strip() for v in di.split("=", 1)) for di in mode_rest
                )
                result_scale = result_scale_base * float(mode_kwargs.pop("rscale", 1.0))
            else:
                result_scale = result_scale_base
                mode_name = mode
                mode_kwargs = {}
            curr_result = self.voronoi_call_mode(
                mode_name,
                result=True,
                args=(d,),
                kwargs=mode_kwargs | base_kwargs,
            ).mul_(result_scale)
            result = curr_result if result is None else result.add_(curr_result)
        return result

    def generate_octave(
        self,
        *,
        octave: int,
        grid: torch.Tensor,
        z_grid: torch.Tensor,
        scale: float = 1.0,
    ) -> torch.Tensor:
        # Full 3D grid (H, W, 3)
        grid_3d = torch.cat((grid, z_grid), dim=-1)[None, None, ...]  # (1, 1, H, W, 3)
        grid_3d = grid_3d.expand(self.batch, self.channels, -1, -1, -1)
        grid_3d = grid_3d.unsqueeze(-2)  # (B, C, H, W, 1, 3)
        grid_3d = (grid_3d * scale) % 1.0

        # Normalize feature points: already assumed in [0, 1)
        fp = self.get_feature_points(octave)  # (B, C, N, 3)
        fp = fp[:, :, None, None]  # (B, C, 1, 1, N, 3)
        fp = (fp * scale) % 1.0

        # Toroidal wrapped difference
        d_orig = d = (grid_3d - fp + 0.5) % 1.0 - 0.5  # Wrap to [-0.5, 0.5)
        d = self.voronoi_distance(d.clone(), octave=octave)
        return self.voronoi_result(d, d_orig, octave=octave)

    def generate(self, *args):
        if self.grid_xyz is None or self.feature_points is None or self.z_max == 0:
            self.voronoi_reset(*args)
        elif self.z_max != 0 and abs(self.z_initial - self.z_curr) > abs(self.z_max):
            if self.z_max_mode == "reset":
                self.voronoi_reset(*args)
            elif self.z_max_mode == "bounce":
                self.z_increment = -self.z_increment
                self.z_curr += self.z_increment
            else:
                self.curr_z = self.z_initial
        z_range = utils.fallback(self.z_range, max(self.height, self.width))
        z_norm = (self.z_curr % z_range) / z_range
        self.z_curr += self.z_increment
        grid = self.grid_xyz
        z_grid = grid.new_full((self.height, self.width, 1), z_norm)

        result = grid.new_zeros(self.shape)
        amplitude = self.initial_amplitude
        scale = self.initial_scale
        total_amplitude = 0.0

        for octave in range(self.octaves):
            result += self.generate_octave(
                octave=octave,
                grid=grid,
                z_grid=z_grid,
                scale=scale,
            ).mul_(amplitude)
            total_amplitude += abs(amplitude)
            amplitude *= self.gain
            scale *= self.lacunarity
        result /= total_amplitude if total_amplitude != 0 else 1.0
        return result
