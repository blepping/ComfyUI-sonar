# ruff: noqa: ANN002, ANN003
from __future__ import annotations

import itertools
import math

import torch

from .base import NoiseGenerator

F = torch.nn.functional


class SimulationNoiseGenerator(NoiseGenerator):
    name = "simulation"
    MIN_DIMS = 4
    MAX_DIMS = 4

    @classmethod
    def ng_params(cls, *, no_super: bool = False):
        result = {
            # multi_octave, power_law, band_pass
            "spectral_mode": "multi_octave",
            # curl, projection, basis
            "field_mode": "basis",
            "depth_mode": "reset",
            "channel_mode": "stacked",
            "band_shape": "log_gaussian",
            "dims": (),
            "base_k": 0.0,
            "power_law_beta": 1.0,
            "depth": 64,
            "initial_depth": 0,
            "max_depth": -1,
            # reset, wrap, bounce
            "octaves": 5,
            "lacunarity": 2.0,
            "gain": 0.5,
            # log_gaussian, raised_cosine
            "log_gaussian_sigma": 0.3,
            # (float, float, float)
            "band_pass_low": 0.00001,
            "band_pass_high": 1.0,
            "anisotropy": (),
            "normalized": False,
            "noise_sampler_factory_h": None,
            "noise_sampler_factory_w": None,
            "noise_sampler_factory_z": None,
        }
        return result if no_super else super().ng_params() | result

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_chunk = None
        cm = self.channel_mode
        self.depth_increment = 1
        if cm in {"over_depth", "over_depth_alt"}:
            self.depth_increment = math.ceil(self.channels / 3)
        elif cm.startswith("over_depth_"):
            self.depth_increment = self.channels
        else:
            self.depth_increment = 1
        if self.initial_depth < 0:
            self.initial_depth = self.depth + self.initial_depth
        if self.initial_depth < 0:
            raise ValueError("Initial depth out of range")
        self.initial_depth = min(self.depth - 1, self.initial_depth)
        if self.max_depth < 0:
            self.max_depth = self.depth + self.max_depth
        if self.max_depth < 0:
            raise ValueError("Max depth out of range")
        self.max_depth = min(self.depth - 1, self.max_depth)
        self.current_depth = self.initial_depth
        self.direction = 1
        self.cdtype = (
            torch.complex128 if self.dtype == torch.float64 else torch.complex64
        )
        self.eff_batch = (
            self.batch if cm != "stacked" else self.batch * math.ceil(self.channels / 3)
        )
        ns_shape = torch.Size((
            self.eff_batch,
            self.depth * self.depth_increment,
            self.height,
            self.width,
        ))

        def gaussian_noise_sampler(*_args: list) -> torch.Tensor:
            return torch.randn(ns_shape, dtype=self.cdtype, device=self.gen_device).to(
                device=self.device,
            )

        self.noise_samplers = tuple(
            factory.make_noise_sampler(
                torch.zeros(ns_shape, device=self.gen_device, dtype=self.dtype),
                cpu=self.cpu,
                normalized=False,
            )
            if factory is not None
            else gaussian_noise_sampler
            for factory in (
                self.noise_sampler_factory_z,
                self.noise_sampler_factory_h,
                self.noise_sampler_factory_w,
            )
        )

    def _k_grids(self, *, shape: tuple, dims: tuple = (-3, -2, -1)) -> tuple:
        """Creates k-space grids."""
        return torch.meshgrid(
            *(
                torch.fft.fftfreq(
                    shape[dim],
                    d=1.0,
                    device=self.device,
                    dtype=self.dtype,
                )
                for dim in dims
            ),
            indexing="ij",
        )

    @staticmethod
    def _radial_k(*ks: list) -> torch.Tensor:
        """Calculates the radial distance in k-space."""
        return sum(kt**2 for kt in ks).sqrt_()

    @staticmethod
    def _raised_cosine_band(
        k: torch.Tensor,
        k_lo: float,
        k_hi: float,
    ) -> torch.Tensor:
        """A raised cosine spectral band filter."""
        kc = 0.5 * (k_lo + k_hi)
        hw = 0.5 * (k_hi - k_lo) + 1e-12
        t = (k - kc) / hw
        return torch.where(
            t.abs() <= 1.0,
            0.5 * (1.0 + torch.cos(math.pi * t)),
            torch.zeros_like(k),
        )

    def _log_gaussian_band(
        self,
        k: torch.Tensor,
        k_lo: float,
        k_hi: float,
    ) -> torch.Tensor:
        """A log-Gaussian spectral band filter."""
        k_center = math.sqrt(k_lo * k_hi)
        log_k = torch.log(torch.clamp(k, min=1e-12))
        log_center = math.log(k_center)
        return torch.exp(-0.5 * ((log_k - log_center) / self.log_gaussian_sigma) ** 2)

    def _handle_band_shape(
        self,
        k_rad: torch.Tensor,
        k_low: float,
        k_high: float,
    ) -> torch.Tensor:
        if self.band_shape == "raised_cosine":
            return self._raised_cosine_band(k_rad, k_low, k_high)
        if self.band_shape == "log_gaussian":
            return self._log_gaussian_band(k_rad, k_low, k_high)
        errstr = f"Bad band shape mode {self.band_shape}"
        raise ValueError(errstr)

    def _make_wk(
        self,
        k_rad: torch.Tensor,
        sizes: tuple,
        *,
        eps: float = 1e-09,
    ) -> torch.Tensor:
        def wk_out(wk: torch.Tensor) -> torch.Tensor:
            wk[k_rad == 0] = 0.0
            return wk

        if self.spectral_mode == "power_law":
            return wk_out((k_rad + eps).pow_(-self.power_law_beta))
        if self.spectral_mode == "band_pass":
            if self.band_pass_low >= self.band_pass_high:
                raise ValueError(
                    "band_pass_high must be greater than band_pass_low in band_pass spectral mode.",
                )
            return wk_out(
                self._handle_band_shape(k_rad, self.band_pass_low, self.band_pass_high),
            )
        base_k = 2 * math.pi / max(1, min(sizes)) if self.base_k == 0 else self.base_k
        wk = torch.zeros_like(k_rad)
        for o in range(self.octaves):
            k_lo = base_k * (self.lacunarity**o)
            k_hi = base_k * (self.lacunarity ** (o + 1))
            band = self._handle_band_shape(k_rad, k_lo, k_hi)
            wk += (self.gain**o) * band

        return wk_out(wk)

    def _handle_field_projection(
        self,
        *,
        k_grids_orig: tuple,
        wk: torch.Tensor,
        ns_args: tuple | list,
        **_kwargs,
    ):
        f_fs = tuple(
            ns(*ns_args).to(device=self.device).mul_(wk) for ns in self.noise_samplers
        )
        # --- Perform the Helmholtz projection using the UN SCALED grids ---
        k_sq_proj = self._radial_k(*k_grids_orig) ** 2
        k_dot_f = sum(k_p * f_f for k_p, f_f in zip(k_grids_orig, f_fs))
        inv_k_sq = torch.where(k_sq_proj == 0, 0.0, 1.0 / k_sq_proj)

        k_grid_scale = k_dot_f.mul_(inv_k_sq)
        return tuple(
            f_f - k_grid * k_grid_scale for f_f, k_grid in zip(f_fs, k_grids_orig)
        )

    # def _handle_field_curl(
    #     self,
    #     *,
    #     k_rad: torch.Tensor,
    #     k_grids_orig: tuple,
    #     wk: torch.Tensor,
    #     ns_args: tuple | list,
    #     **_kwargs,
    # ):
    #     if len(k_grids_orig) != 3:
    #         raise ValueError("Can only handle 3 dimensions currently")
    #     # inv_k_rad = torch.where(k_rad == 0, 0.0, 1.0 / k_rad)
    #     # wk_potential = wk * inv_k_rad
    #     wk_potential = wk / k_rad
    #     f_fs = tuple(
    #         ns(*ns_args).to(device=self.device).mul_(wk_potential)
    #         for ns in self.noise_samplers
    #     )
    #     i_k_grids = tuple(
    #         (1j * k_grid).to(dtype=self.cdtype) for k_grid in k_grids_orig
    #     )
    #     fz, fy, fx = f_fs
    #     ikz, iky, ikx = i_k_grids
    #     return (
    #         ikx * fy - iky * fx,  # z
    #         ikz * fx - ikx * fz,  # y
    #         iky * fz - ikz * fy,  # x
    #     )

    def _handle_field_curl(
        self,
        *,
        k_rad: torch.Tensor,
        k_grids_orig: tuple,
        wk: torch.Tensor,
        ns_args: tuple | list,
        **_kwargs,
    ):
        n_dims = len(k_grids_orig)
        nd_fixup = int(self.field_mode != "curl_ndim")

        # The potential filter still uses the scaled k_rad for spectral shaping
        inv_k_rad = torch.where(k_rad == 0, 0.0, 1.0 / k_rad)
        wk_potential = wk * inv_k_rad

        # The curl operator (i*k) MUST use the original, un-scaled grids
        i_k_grids = tuple(
            (1j * k_grid).to(dtype=self.cdtype) for k_grid in k_grids_orig
        )

        n_samplers = len(self.noise_samplers)
        g_fs = tuple(
            self.noise_samplers[ns_idx % n_samplers](*ns_args)
            .to(
                device=self.device,
            )
            .mul_(wk_potential)
            for ns_idx in range(n_dims if n_dims != 2 else 1)
        )
        # --- Case 1: 2D Curl (Curl of a SCALAR potential) ---
        # This is the fundamental building block.
        if n_dims * nd_fixup == 2:
            # We only need one scalar potential field G.
            g_f = g_fs[0]
            ikx, iky = i_k_grids
            # F = (dG/dy, -dG/dx) -> F_f = (iky*G_f, -ikx*G_f)
            return (iky * g_f, -ikx * g_f)

        # --- Case 2: 3D Curl (The classic cross-product) ---
        # This is a special, unique case.
        if n_dims * nd_fixup == 3:
            gz_f, gy_f, gx_f = g_fs
            ikz, iky, ikx = i_k_grids
            # F_f = i*k x G_f
            return (
                ikx * gy_f - iky * gx_f,  # z component
                ikz * gx_f - ikx * gz_f,  # y component
                iky * gz_f - ikz * gy_f,  # x component
            )

        # --- Case 3: N-D Curl (Pragmatic construction) ---
        # We build the N-D field by summing 2D curls on orthogonal planes.

        # We need N potential fields, but we will use them in pairs.

        f_f_outputs = [torch.zeros_like(g_fs[0]) for _ in range(n_dims)]

        # Iterate over pairs of dimensions (0,1), (2,3), etc.
        for i in range(n_dims // 2):
            idx1 = i * 2
            idx2 = i * 2 + 1

            g1_f = g_fs[idx1]
            g2_f = g_fs[idx2]
            ik1 = i_k_grids[idx1]
            ik2 = i_k_grids[idx2]

            # Perform a 2D-like curl on the (G1, G2) plane
            # This is a bit abstract, but we are creating rotation in the 1-2 plane.
            f_f_outputs[idx1] = ik2 * g1_f - ik1 * g2_f
            f_f_outputs[idx2] = ik1 * g2_f - ik2 * g1_f

        return tuple(f_f_outputs)

    _handle_field_curl_ndim = _handle_field_curl

    def _handle_field_basis_orig(
        self,
        *,
        k_rad: torch.Tensor,
        k_grids_orig: tuple,
        wk: torch.Tensor,
        ns_args: tuple | list,
        **_kwargs,
    ) -> tuple:
        if len(k_grids_orig) != 3:
            raise ValueError("Can only handle 3 dimensions currently")
        k_norm_z, k_norm_y, k_norm_x = (
            torch.where(k_rad == 0, 0.0, k / k_rad) for k in k_grids_orig
        )

        # Pick a fixed vector `ez` to take a cross product with.
        # Handle the singularity where k is parallel to ez.
        ez = torch.tensor((0.0, 0.0, 1.0), device=self.device, dtype=self.dtype)
        is_parallel = (k_norm_x.abs() < 1e-6) & (k_norm_y.abs() < 1e-6)

        # First basis vector u = k x ez (or k x ey for the singularity)
        ux = torch.where(
            is_parallel,
            k_norm_y * 0 - k_norm_z * 1,
            k_norm_y * ez[2] - k_norm_z * ez[1],
        )
        uy = torch.where(
            is_parallel,
            k_norm_z * 0 - k_norm_x * 0,
            k_norm_z * ez[0] - k_norm_x * ez[2],
        )
        uz = torch.where(
            is_parallel,
            k_norm_x * 1 - k_norm_y * 0,
            k_norm_x * ez[1] - k_norm_y * ez[0],
        )

        u_mag = torch.sqrt(sum((ux**2, uy**2, uz**2)))
        inv_u_mag = torch.where(u_mag == 0, 0.0, 1.0 / u_mag)
        ux *= inv_u_mag
        uy *= inv_u_mag
        uz *= inv_u_mag
        ux[k_rad == 0], uy[k_rad == 0], uz[k_rad == 0] = 0, 0, 0

        # Second basis vector v = k x u
        vx = k_norm_y * uz - k_norm_z * uy
        vy = k_norm_z * ux - k_norm_x * uz
        vz = k_norm_x * uy - k_norm_y * ux

        # 3. Generate two independent random complex scalar fields
        a_f = self.noise_samplers[0](*ns_args).to(device=self.device)
        b_f = self.noise_samplers[1](*ns_args).to(device=self.device)

        # 4. Modulate the random fields by the spectral envelope
        a_f *= wk
        b_f *= wk

        # 5. Project the random fields onto the basis vectors to form the final field
        return (
            a_f * uz + b_f * vz,  # z
            a_f * uy + b_f * vy,  # y
            a_f * ux + b_f * vx,  # x
        )

    def _handle_field_basis(
        self,
        *,
        k_grids_orig: tuple,
        wk: torch.Tensor,
        ns_args: tuple | list,
        **_kwargs,
    ) -> tuple:
        n_dims = len(k_grids_orig)
        nd_fixup = int(self.field_mode != "basis_ndim")
        k_rad_orig = self._radial_k(*k_grids_orig)

        # Normalize the original k vector
        k_norm_components = tuple(
            torch.where(k_rad_orig == 0, 0.0, k / k_rad_orig) for k in k_grids_orig
        )

        # --- Case 1: 2D (simple and fast) ---
        if n_dims * nd_fixup == 2:
            if len(self.noise_samplers) < 1:
                raise ValueError("2D basis mode requires at least 1 noise sampler.")

            # The basis is a single vector perpendicular to k: u = (-ky, kx)
            kn_y, kn_x = k_norm_components
            basis_vectors = [
                (-kn_x, kn_y),
            ]  # A list containing one basis vector (a tuple)

            num_random_fields = 1

        # --- Case 2: 3D (fast cross-product method) ---
        elif n_dims * nd_fixup == 3:
            kn_z, kn_y, kn_x = k_norm_components
            ez = torch.tensor([0.0, 0.0, 1.0], device=self.device, dtype=self.dtype)
            is_parallel = (kn_x.abs() < 1e-6) & (kn_y.abs() < 1e-6)

            ux = torch.where(is_parallel, 0.0, kn_y * ez[2] - kn_z * ez[1])
            uy = torch.where(is_parallel, -kn_z, kn_z * ez[0] - kn_x * ez[2])
            uz = torch.where(is_parallel, kn_x, kn_x * ez[1] - kn_y * ez[0])

            u_mag = torch.sqrt(ux**2 + uy**2 + uz**2)
            inv_u_mag = torch.where(u_mag == 0, 0.0, 1.0 / u_mag)
            ux, uy, uz = ux * inv_u_mag, uy * inv_u_mag, uz * inv_u_mag

            u = (uz, uy, ux)

            vx = kn_y * u[2] - kn_z * u[1]
            vy = kn_z * u[0] - kn_x * u[2]
            vz = kn_x * u[1] - kn_y * u[0]

            v = (vz, vy, vx)

            basis_vectors = [u, v]
            num_random_fields = 2

        # --- Case 3: N-D (General Gram-Schmidt process) ---
        else:
            num_random_fields = n_dims - 1

            basis_vectors = []
            # Start with the standard basis vectors (e.g., [1,0,0], [0,1,0], [0,0,1])
            for i in range(n_dims):
                # Create a standard basis vector e_i
                e_i = [torch.zeros_like(k_rad_orig) for _ in range(n_dims)]
                e_i[i] = torch.ones_like(k_rad_orig)

                # Start with v = e_i and make it orthogonal to k
                v = list(e_i)
                dot_k = sum(
                    v_comp * k_comp for v_comp, k_comp in zip(v, k_norm_components)
                )
                v = [
                    v_comp - dot_k * k_comp
                    for v_comp, k_comp in zip(v, k_norm_components)
                ]

                # Make it orthogonal to all previously found basis vectors
                for b in basis_vectors:
                    dot_b = sum(v_comp * b_comp for v_comp, b_comp in zip(v, b))
                    v = [v_comp - dot_b * b_comp for v_comp, b_comp in zip(v, b)]

                # Normalize the new basis vector
                v_mag = torch.sqrt(sum(comp**2 for comp in v))
                # Only add the vector if it's not a zero vector
                if torch.any(v_mag > 1e-6):
                    inv_v_mag = torch.where(v_mag == 0, 0.0, 1.0 / v_mag)
                    v = [comp * inv_v_mag for comp in v]
                    basis_vectors.append(tuple(v))

                if len(basis_vectors) == num_random_fields:
                    break

        # --- Field Construction (works for all cases) ---

        # Generate N-1 independent random complex scalar fields
        n_samplers = len(self.noise_samplers)
        random_fields = tuple(
            self.noise_samplers[ns_idx % n_samplers](*ns_args)
            .to(device=self.device)
            .mul_(wk)
            for ns_idx in range(num_random_fields)
        )

        # Initialize the final field components to zero
        f_f_outputs = [torch.zeros_like(random_fields[0]) for _ in range(n_dims)]

        # Project each random field onto its corresponding basis vector and sum them up
        for i in range(num_random_fields):
            a_f = random_fields[i]
            basis_vec = basis_vectors[i]
            for j in range(n_dims):
                f_f_outputs[j] += a_f * basis_vec[j]

        return tuple(f_f_outputs)

    _handle_field_basis_ndim = _handle_field_basis

    def generate_octaves(
        self,
        batch: int,
        height: int,
        width: int,
        *,
        ns_args: tuple | list,
    ) -> torch.Tensor:
        depth = self.depth * self.depth_increment
        eff_shape = torch.Size((batch, 3, depth, height, width))

        # 1. Create the UN SCALED k-grids for the projection operator.
        k_grids_orig = k_grids = self._k_grids(shape=eff_shape)

        # 2. Create a separate set of k-grids for spectral shaping.
        # These can be scaled by the anisotropy factors.
        if self.anisotropy and not all(v in {0, 1} for v in self.anisotropy):
            n_anisotropy = len(self.anisotropy)
            anisotropy = tuple(
                1.0 if idx >= n_anisotropy else self.anisotropy[idx] for idx in range(3)
            )
            k_grids = tuple(
                k_p if a in {None, 0, 1} else k_p / a
                for k_p, a in itertools.zip_longest(k_grids, anisotropy)
            )

        # 3. Calculate radial k for the spectral envelope using the SCALED grids.
        k_rad = self._radial_k(*k_grids)

        # Build the multi-octave spectral envelope (Wk) using the anisotropic k_rad
        wk = self._make_wk(k_rad, sizes=(depth, height, width))

        field_handler = getattr(self, f"_handle_field_{self.field_mode}", None)
        if field_handler is None:
            errstr = f"Bad field mode {self.field_mode}"
            raise ValueError(errstr)
        f_f_outputs = field_handler(
            k_rad=k_rad,
            k_grids=k_grids,
            k_grids_orig=k_grids_orig,
            wk=wk,
            ns_args=ns_args,
        )

        # Inverse FFT to transform the field back to the spatial domain
        field = torch.stack(
            tuple(
                torch.fft.ifftn(f_proj, dim=(-3, -2, -1)).real
                for f_proj in reversed(f_f_outputs)
            ),
            dim=1,
        )

        rms = torch.sqrt(torch.mean(field**2))
        if rms > 1e-9:
            field /= rms

        return field

    def generate(self, *args) -> torch.Tensor:
        cm = self.channel_mode
        if self.noise_chunk is None:
            self.noise_chunk = self.generate_octaves(
                self.eff_batch,
                self.height,
                self.width,
                ns_args=args,
            ).to(dtype=self.dtype)
            self.current_depth = self.initial_depth
        depth_from = self.current_depth * self.depth_increment
        depth_to = depth_from + self.depth_increment
        if cm == "stacked":
            noise = self.noise_chunk[:, :, self.current_depth]
            noise = torch.cat(
                tuple(
                    noise[bidx * self.batch : bidx * self.batch + self.batch]
                    for bidx in range(noise.shape[0] // self.batch)
                ),
                dim=2,
            )
        elif cm in {"over_depth", "over_depth_alt"}:
            noise = self.noise_chunk[:, :, depth_from:depth_to]
            if cm == "over_depth":
                noise = noise.movedim(2, 1)
        elif cm == "over_depth_avg":
            noise = self.noise_chunk[:, :, depth_from:depth_to].mean(dim=1)
        elif cm.startswith("over_depth_"):
            channel_lookup = {"h": 0, "w": 1, "z": 2}
            mathop = cm[-5:-2]
            if mathop in {"add", "sub", "mul", "div"}:
                chan1, chan2 = channel_lookup[cm[-7]], channel_lookup[cm[-1]]
                noise1 = self.noise_chunk[:, chan1 : chan1 + 1, depth_from:depth_to]
                noise2 = self.noise_chunk[:, chan2 : chan2 + 1, depth_from:depth_to]
                if mathop == "sub":
                    noise = noise1 - noise2
                elif mathop == "add":
                    noise = noise1 + noise2
                elif mathop == "mul":
                    noise = noise1 * noise2
                elif mathop == "div":
                    noise = noise1 / (noise2 + 1e-07)
            else:
                chan = channel_lookup[cm[-1]]
                noise = self.noise_chunk[:, chan : chan + 1, depth_from:depth_to]
        else:
            raise ValueError("Bad channel mode")
        self.current_depth += 1 * self.direction
        if self.current_depth > self.max_depth or self.current_depth < 0:
            dm = self.depth_mode
            if dm == "reset":
                self.noise_chunk = None
            elif dm == "wrap":
                self.current_depth = self.initial_depth
            elif dm == "bounce":
                if self.depth < 2:
                    raise ValueError("Bounce depth mode requires depth of at least 2")
                self.direction = -self.direction
                self.current_depth += 2 * self.direction
        return (
            noise.reshape(self.batch, -1, self.height, self.width)[
                :,
                : self.channels,
            ]
            .clone()
            .contiguous()
        )
