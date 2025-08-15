# ruff: noqa: ANN002, ANN003
from __future__ import annotations

import torch

from ..utils import quantile_normalize
from .base import NoiseGenerator


class DistroNoiseGenerator(NoiseGenerator):
    name = "distro"

    simple_distros = frozenset((
        "cauchy",
        "exponential",
        "geometric",
        "log_normal",
        "normal",
    ))

    def __init__(self, x, *args, **kwargs):
        super().__init__(x, *args, **kwargs)
        if self.distro not in self.distro_params():
            raise ValueError("Bad distro")

    _distro_params = None

    @classmethod
    def distro_params(cls):
        if cls._distro_params is not None:
            return cls._distro_params
        td = torch.distributions
        tt = torch.Tensor
        cls._distro_params = {
            # Simple
            "exponential": (
                tt.exponential_,
                {
                    "lambd": {
                        "default": 1.0,
                    },
                },
            ),
            "cauchy": (
                tt.cauchy_,
                {
                    "median": {
                        "default": "0.0",
                    },
                    "sigma": {
                        "default": 1.0,
                        "min": 0.0,
                    },
                },
            ),
            "geometric": (
                tt.geometric_,
                {
                    "p": {
                        "default": 0.25,
                    },
                },
            ),
            "log_normal": (
                tt.log_normal_,
                {
                    "mean": {
                        "default": 1.0,
                    },
                    "std": {
                        "default": 2.0,
                    },
                },
            ),
            "normal": (
                tt.normal_,
                {
                    "mean": {
                        "default": 0.0,
                    },
                    "std": {
                        "default": 1.0,
                    },
                },
            ),
            # Complex distros
            "beta": (
                td.Beta,
                {
                    "concentration0": {
                        "default": "0.5",
                    },
                    "concentration1": {
                        "default": "0.5",
                    },
                },
            ),
            "continuous_bernoulli": (
                td.ContinuousBernoulli,
                {
                    "probs": {
                        "default": "0.5",
                    },
                },
            ),
            "dirichlet": (
                td.Dirichlet,
                {
                    "concentration": {
                        "default": "0.5 0.5",
                    },
                },
            ),
            "fisher_snedecor": (
                td.FisherSnedecor,
                {
                    "df1": {
                        "default": "1.0",
                    },
                    "df2": {
                        "default": "2.0",
                    },
                },
            ),
            "gamma": (
                td.Gamma,
                {
                    "concentration": {
                        "default": "1.0",
                    },
                    "rate": {
                        "default": "1.0",
                    },
                },
            ),
            "gumbel": (
                td.Gumbel,
                {
                    "loc": {
                        "default": "1.0",
                    },
                    "scale": {
                        "default": "2.0",
                    },
                },
            ),
            "inverse_gamma": (
                td.InverseGamma,
                {
                    "concentration": {
                        "default": "1.0",
                    },
                    "rate": {
                        "default": "1.0",
                    },
                },
            ),
            "kumaraswamy": (
                td.Kumaraswamy,
                {
                    "concentration0": {
                        "default": "1.0",
                    },
                    "concentration1": {
                        "default": "1.0",
                    },
                },
            ),
            "laplacian": (
                td.Laplace,
                {
                    "loc": {
                        "default": "0.0",
                    },
                    "scale": {
                        "default": "1.0",
                    },
                },
            ),
            "lkjcholesky": (
                td.LKJCholesky,
                {
                    "dim": {
                        "_ty": "INT",
                        "default": 3,
                    },
                    "concentration": {
                        "default": "1.0",
                    },
                },
            ),
            "lrmvariate_normal": (
                lambda loc, cov_factor, cov_diag: td.LowRankMultivariateNormal(
                    loc=loc,
                    cov_factor=cov_factor.reshape(loc.numel(), -1),
                    cov_diag=cov_diag,
                ),
                {
                    "loc": {
                        "default": "0.0 0.0",
                    },
                    "cov_factor": {
                        "default": "1.0 0.0",
                    },
                    "cov_diag": {
                        "default": "1.0 1.0",
                    },
                },
            ),
            "mvariate_normal": (
                lambda loc, cov_multiplier=1.0: td.MultivariateNormal(
                    loc=loc,
                    covariance_matrix=torch.eye(
                        loc.numel(),
                        dtype=loc.dtype,
                        device=loc.device,
                    ).mul_(cov_multiplier),
                ),
                {
                    "loc": {
                        "default": "0.0 0.0",
                    },
                    "cov_multiplier": {
                        "default": 1.0,
                    },
                },
            ),
            "pareto": (
                td.Pareto,
                {
                    "scale": {
                        "default": "1.0",
                    },
                    "alpha": {
                        "default": "1.0",
                    },
                },
            ),
            "poisson": (
                td.Poisson,
                {
                    "rate": {
                        "default": "1.5",
                    },
                },
            ),
            "relaxed_bernoulli": (
                td.RelaxedBernoulli,
                {
                    "temperature": {
                        "default": 0.75,
                    },
                    "probs": {
                        "default": "0.66",
                    },
                },
            ),
            "relaxed_onehotcategorical": (
                td.RelaxedOneHotCategorical,
                {
                    "temperature": {
                        "default": 1.5,
                    },
                    "probs": {
                        "default": "0.33 0.66",
                    },
                },
            ),
            "studentt": (
                td.StudentT,
                {
                    "loc": {
                        "default": "0.0",
                    },
                    "scale": {
                        "default": "1.0",
                    },
                    "df": {
                        "default": "1.0",
                    },
                },
            ),
            "uniform": (
                td.Uniform,
                {
                    "low": {
                        "default": 0.0,
                    },
                    "high": {
                        "default": 1.0,
                    },
                },
            ),
            "vonmises": (
                td.VonMises,
                {
                    "loc": {
                        "default": "1.0",
                    },
                    "concentration": {
                        "default": "1.0",
                    },
                },
            ),
            "weibull": (
                td.Weibull,
                {
                    "scale": {
                        "default": "1.0",
                    },
                    "concentration": {
                        "default": "1.0",
                    },
                },
            ),
            "wishart": (
                lambda df, cov_size=2, cov_multiplier=1.0: td.Wishart(
                    df=df,
                    covariance_matrix=torch.eye(
                        int(cov_size),
                        dtype=df.dtype,
                        device=df.device,
                    ).mul_(cov_multiplier),
                ),
                {
                    "df": {
                        "default": "2.0",
                    },
                    "cov_size": {
                        "_ty": "INT",
                        "default": 2,
                    },
                    "cov_multiplier": {
                        "default": 1.0,
                    },
                },
            ),
        }
        return cls._distro_params

    _build_params = None

    @classmethod
    def build_params(cls):
        if cls._build_params is not None:
            return cls._build_params
        cls._build_params = {
            f"{tykey}_{pkey}": pval
            for tykey, tyval in cls.distro_params().items()
            for pkey, pval in tyval[1].items()
            if not pkey.startswith("_")
        }
        return cls._build_params

    _ng_params = None

    @classmethod
    def ng_params(cls):
        if cls._ng_params is not None:
            return cls._ng_params
        dparams = {
            k: v["default"]
            for k, v in cls.build_params().items()
            if not k.startswith("_")
        }
        cls._ng_params = (
            super().ng_params()
            | {
                "distro": "normal",
                "quantile_norm": 0.85,
                "quantile_norm_flatten": True,
                "quantile_norm_dim": 1,
                "quantile_norm_pow": 0.5,
                "quantile_norm_fac": 1.0,
                "result_index": "-1",
            }
            | dparams
        )
        return cls._ng_params

    def norm_output(self, noise):
        if noise.ndim > len(self.shape):
            if noise.shape[: len(self.shape)] != self.shape:
                errstr = f"Unexpected shape when normalizing distro({self.distro}) noise! Output shape={self.shape}, noise shape={noise.shape}, generator dump: {self}"
                raise RuntimeError(errstr)
            selfdims = len(self.shape)
            result_index = self.result_index
            if not isinstance(result_index, (tuple, list)):
                result_index = (result_index,)
            ri_len = len(result_index)
            if ri_len == 0:
                raise ValueError("When result_index is a list, it must not be empty")
            trim_count = 0
            while noise.ndim > selfdims:
                idx = result_index[trim_count % ri_len]
                if idx < 0:
                    idx = noise.shape[-1] + idx
                noise = noise[..., max(0, min(noise.shape[-1] - 1, idx))]
                trim_count += 1
        return (
            quantile_normalize(
                noise,
                quantile=self.quantile_norm,
                dim=self.quantile_norm_dim,
                flatten=self.quantile_norm_flatten,
                nq_fac=self.quantile_norm_fac,
                pow_fac=self.quantile_norm_pow,
            )
            .reshape(self.shape)
            .contiguous()
        )

    def distro_param(self, val, *, simple_fun=None):
        if isinstance(val, torch.Tensor):
            return simple_fun(val) if simple_fun is not None else val
        if isinstance(val, str):
            val = tuple(float(v) for v in val.split(None))
        if simple_fun is not None:
            if isinstance(val, (float, int)):
                return simple_fun(val)
            if len(val) > 1:
                raise ValueError("Couldn't return result as float")
            return simple_fun(val[0])
        if not isinstance(val, (tuple, list)):
            val = (val,)
        return torch.tensor(
            val,
            dtype=self.dtype,
            device=self.gen_device,
        )

    def get_distro_kwargs(self, distro, ddef, *, simple=False):
        return {
            k: self.distro_param(
                getattr(self, f"{distro}_{k}"),
                simple_fun=None
                if not simple and k != "dim"
                else (int if k == "dim" else float),
            )
            for k in ddef
        }

    def generate(self, *_args):
        distro = self.distro
        dfun, ddef = self.distro_params()[distro]
        is_simple = distro in self.simple_distros
        dkwargs = self.get_distro_kwargs(distro, ddef, simple=is_simple)
        if is_simple:
            noise = torch.empty(
                *self.shape,
                device=self.gen_device,
                dtype=self.dtype,
                layout=self.layout,
            )
            noise = dfun(noise, **dkwargs)
        else:
            dobj = dfun(**dkwargs)
            noise = (
                dobj.rsample if getattr(dobj, "has_rsample", False) else dobj.sample
            )(self.shape)
        return self.norm_output(noise)
