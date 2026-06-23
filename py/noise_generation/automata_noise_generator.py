from __future__ import annotations

import math
from functools import partial
from typing import TYPE_CHECKING, Any

import torch
from comfy.model_management import throw_exception_if_processing_interrupted
from tqdm import trange

from .base import NoiseGenerator

if TYPE_CHECKING:
    from collections.abc import Callable

F = torch.nn.functional


# Analytic extension of the Collatz conjecture for floating point numbers.
def continuous_collatz(x: torch.Tensor) -> torch.Tensor:
    # f(x) = 1/4 * (2 + 7x - (2 + 5x)*cos(pi*x))
    cos_term = (x * torch.pi).cos_()
    return x.mul(7).add_(2).sub_(x.mul(5).add_(2).mul_(cos_term)).mul_(0.25)


def generate_spatial_collatz_noise(
    batch: int,
    channels: int,
    height: int,
    width: int,
    depth: int = None,  # Optional 3D depth
    steps: int = 20,
    num_seeds: int = 15,
    device: str = "cpu",
):
    is_3d = depth is not None

    # 1. Initialize grid
    if is_3d:
        grid = torch.zeros((batch, channels, depth, height, width), device=device)
        norm_dims = [2, 3, 4]
    else:
        grid = torch.zeros((batch, channels, height, width), device=device)
        norm_dims = [2, 3]

    # 2. Plant float/negative "seeds"
    for b in range(batch):
        for c in range(channels):
            seed_y = torch.randint(0, height, (num_seeds,))
            seed_x = torch.randint(0, width, (num_seeds,))

            # Using random floats from -1000 to 1000
            seed_vals = (
                torch.rand((num_seeds,), dtype=torch.float32, device=device) * 2000.0
            ) - 1000.0

            if is_3d:
                seed_z = torch.randint(0, depth, (num_seeds,))
                grid[b, c, seed_z, seed_y, seed_x] = seed_vals
            else:
                grid[b, c, seed_y, seed_x] = seed_vals

    # 3. Create spatial diffusion kernel
    if is_3d:
        # Create a 3x3x3 blurring kernel using outer products
        k1d = torch.tensor([1.0, 2.0, 1.0], device=device)
        kernel = (k1d.view(3, 1, 1) * k1d.view(1, 3, 1) * k1d.view(1, 1, 3)) / 64.0
        kernel = kernel.view(1, 1, 3, 3, 3).repeat(channels, 1, 1, 1, 1)
        conv_fn = F.conv3d
    else:
        # Create a 3x3 blurring kernel
        k1d = torch.tensor([1.0, 2.0, 1.0], device=device)
        kernel = (k1d.view(3, 1) * k1d.view(1, 3)) / 16.0
        kernel = kernel.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        conv_fn = F.conv2d

    # 4. Evolve the grid
    for _ in trange(steps, desc="Automata", miniter=25):
        # A. Spatial diffusion (spread values into neighboring dimensions)
        grid = conv_fn(grid, kernel, padding=1, groups=channels)

        # B. Apply Collatz activation
        grid = continuous_collatz(grid)

        # C. Reset rule: inject new seeds if elements get trapped in low magnitude cycles
        trapped_mask = grid.abs() <= 1.5
        if trapped_mask.any():
            new_seeds = (torch.rand_like(grid) * 200.0) - 100.0
            grid = torch.where(trapped_mask, new_seeds, grid)

        # D. Internal Instance Normalization to tame the math
        mean = grid.mean(dim=norm_dims, keepdim=True)
        std = grid.std(dim=norm_dims, keepdim=True) + 1e-5
        grid = (grid - mean) / std

    # 5. Final Output Normalization
    mean = grid.mean(dim=norm_dims, keepdim=True)
    std = grid.std(dim=norm_dims, keepdim=True) + 1e-5

    return (grid - mean) / std


class AutomataNoiseGenerator(NoiseGenerator):
    name = "automata"

    blend_function: Callable | None = None

    @classmethod
    def ng_params(cls):
        return super().ng_params() | {
            # Evolution mode
            # collatz, logistic, sawtooth, lenia, roll
            "evolution_mode": "collatz",
            # blur, laplacian, crystal
            "spread_mode": "blur",
            "steps": 20,
            "spread_substeps": 3,
            "num_seeds": 10,
            "depth": 10,
            "trapped_threshold": 1.5,
            "trapped_interval": 1,
            # Controls behavior for trapped elements.
            # new - new seed, reset - original seed, mean - replace with mean
            "trapped_mode": "reset",
            "range_negative": -100.0,
            "range_positive": 100.0,
            # Absolute value.
            "seed_minimum": 1.5,
            "noise_sampler_factory": None,
        }

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        if not (self.height and self.width):
            raise ValueError("Unsupported shape")
        self.grid = self.grid_orig = None
        self.noise_chunk = None
        self.current_depth = 0

    def create_grid(self) -> None:
        batch, channels = self.batch, self.channels
        height, width, depth = self.height, self.width, self.depth
        num_seeds = self.num_seeds
        is_3d = depth > 0
        device, dtype = self.gen_device, self.dtype
        total_seeds = batch * channels * num_seeds

        # Create flat arrays of coordinates for every single seed
        batch_idx = (
            torch.arange(batch, device=device)
            .view(-1, 1, 1)
            .expand(batch, channels, num_seeds)
            .flatten()
        )
        chan_idx = (
            torch.arange(channels, device=device)
            .view(1, -1, 1)
            .expand(batch, channels, num_seeds)
            .flatten()
        )
        y_idx = torch.randint(
            0,
            height,
            (total_seeds,),
            device=device,
            generator=self.generator,
        )
        x_idx = torch.randint(
            0,
            width,
            (total_seeds,),
            device=device,
            generator=self.generator,
        )

        if is_3d:
            grid = torch.zeros(
                (batch, channels, depth, height, width),
                device=device,
                dtype=dtype,
            )
        else:
            grid = torch.zeros(
                (batch, channels, height, width),
                device=device,
                dtype=dtype,
            )

        seed_vals = (
            torch.rand(
                (total_seeds,), dtype=dtype, device=device, generator=self.generator
            )
            * 2000.0
        ) - 1000.0
        seed_vals = seed_vals.abs().clamp_min(self.seed_minimum).copysign(seed_vals)

        if is_3d:
            z_idx = torch.randint(0, depth, (total_seeds,), device=device)
            grid[batch_idx, chan_idx, z_idx, y_idx, x_idx] = seed_vals
        else:
            grid[batch_idx, chan_idx, y_idx, x_idx] = seed_vals
        self.grid = grid
        self.initial_grid = grid.clone()

    def evolve_step(self, grid: torch.Tensor) -> torch.Tensor:
        depth = 0 if grid.ndim < 5 else grid.shape[-3]
        # k1d = torch.tensor([1.0, 2.0, 1.0], device=device)
        k1d = torch.tensor(
            [0.1, 1.0, 0.1],
            device=grid.device,
            dtype=grid.dtype,
        )
        if depth > 0:
            # Create a 3x3x3 blurring kernel using outer products
            kernel = k1d.view(3, 1, 1) * k1d.view(1, 3, 1) * k1d.view(1, 1, 3)
            kernel /= kernel.sum()
            kernel = kernel.view(1, 1, 3, 3, 3).repeat(self.channels, 1, 1, 1, 1)
        else:
            # Create a 3x3 blurring kernel
            kernel = k1d.view(3, 1) * k1d.view(1, 3)
            kernel /= kernel.sum()
            kernel = kernel.view(1, 1, 3, 3).repeat(self.channels, 1, 1, 1)
        op = partial(
            F.conv3d if depth > 0 else F.conv2d,
            weight=kernel,
            padding=1,
            groups=self.channels,
        )
        # op = partial(F.max_pool3d if depth > 0 else F.max_pool2d, kernel_size=3, stride=1, padding=1)
        for _ in range(self.spread_substeps):
            grid = grid.lerp(op(grid), 1.0)
            # grid = F.max_pool3d(grid, 3, stride=1, padding=1)
            # grid = conv_fn(grid, kernel, padding=1, groups=self.channels)
        grid = continuous_collatz(grid)
        return grid
        # return continuous_collatz(grid)

    def handle_trapped(
        self,
        *,
        grid: torch.Tensor,
        orig_grid: torch.Tensor,
        grid_prev: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.trapped_threshold == 0:
            return grid
        mask = grid.abs() < self.trapped_threshold
        if grid_prev is not None:
            mask &= grid_prev.abs() >= self.trapped_threshold
        if not torch.any(mask):
            return grid

        new_seeds = (torch.rand_like(grid) * 2000.0) - 1000.0
        return torch.where(mask, new_seeds, grid)
        # return torch.where(mask, orig_grid, grid) if torch.any(mask) else grid

    def handle_norm(self, grid: torch.Tensor) -> torch.Tensor:
        return grid.clamp(-10000.0, 10000.0)
        dims = tuple(range(2, grid.ndim))
        gn = grid.clone()
        gn /= gn.std(dim=dims, keepdim=True).clamp_min_(1e-06)
        return grid.lerp(gn, grid.abs().div_(10000.0).clamp_max_(1.0))
        # mask = grid.abs() > 100.0
        # return torch.where(mask, grid.lerp(gn, 0.5), grid)
        # gn = grid - grid.mean(dim=dims, keepdim=True)

    def handle_norm_(self, grid: torch.Tensor) -> torch.Tensor:
        # return (grid.abs() % 1000000.0).copysign_(grid)
        # mask = grid.abs() > 40000.0
        # return torch.where(
        #     mask,
        #     (grid.cos() * 1000.0).abs().clamp_min(1.5).copysign(grid),
        #     grid,
        # )
        # return torch.where(mask, (grid.abs() % 2000.0).copysign(grid), grid)
        # new_seeds = (torch.rand_like(grid) * 2000.0) - 1000.0
        # return torch.where(mask, new_seeds, grid)
        # return grid * (~mask).to(grid)
        mask = grid.abs() > 100000000.0
        grid = (grid.abs() % 100000000.0).copysign(grid)
        return grid
        dims = tuple(range(2, grid.ndim))
        std = grid.std(dim=dims, keepdim=True)
        std = std.abs().clamp_min_(1e-08).copysign(std)
        grid_adj = grid / std
        grid_adj -= grid_adj.mean(dim=dims, keepdim=True)
        grid = torch.where(mask, grid_adj, grid)
        return grid

    def evolve(self):
        if self.grid is None:
            self.create_grid()
        grid = self.grid
        for i in trange(self.steps, miniters=10, desc="Automata step"):
            if i > 1 and (i % 5) == 0:
                throw_exception_if_processing_interrupted()
            grid_prev = grid
            grid = self.evolve_step(grid)
            grid = self.handle_trapped(
                grid=grid,
                orig_grid=self.initial_grid,
                grid_prev=grid_prev,
            )
            grid = self.handle_norm(grid)
        self.grid = grid

    def reset_grid(self):
        self.grid = self.initial_grid = None
        self.current_depth = 0

    def generate(self, *args) -> torch.Tensor:
        if self.grid is None:
            self.create_grid()
            self.current_depth = 0
            self.evolve()
        if self.grid.ndim < 5:
            return self.grid.clone()
            grid = self.grid
            self.reset_grid()
            return grid
        result = self.grid[:, :, self.current_depth, ...].clone()
        self.current_depth += 1
        if self.current_depth >= self.grid.shape[-3]:
            self.reset_grid()
        return result
