# ruff: noqa: A002
from __future__ import annotations

from copy import deepcopy
from functools import partial
from typing import Callable, TypeVar


class InputCollection:
    _DELEGATE_KEYS = frozenset((
        "bool",
        "boolean",
        "clip",
        "conditioning",
        "field",
        "float",
        "image",
        "int",
        "latent",
        "model",
        "sampler",
        "seed",
        "sigmas",
        "string",
        "vae",
    ))

    def __init__(self, **kwargs: dict):
        self.fields = kwargs

    def __getattr__(self, key: str):
        splitkey = key.split("_", 1)
        if len(splitkey) == 1 or splitkey[0] not in self._DELEGATE_KEYS:
            errstr = f"Unknown attribute {key} for InputCollection"
            raise AttributeError(errstr)
        meth = getattr(self, splitkey[0])
        return partial(meth, splitkey[1]) if len(splitkey) == 2 else meth

    def to_dict(self):
        return deepcopy(self.fields)

    def clone(self):
        return InputCollection(**self.to_dict())

    def __len__(self) -> int:
        return len(self.fields)

    def __contains__(self, key: str) -> bool:
        return key in self.fields

    def field(
        self,
        name: str,
        type: str | tuple,
        *,
        _skip: bool = False,
        **kwargs: dict,
    ) -> InputCollection:
        if not _skip:
            self.fields[name] = (type,) if not kwargs else (type, kwargs)
        return self

    def string(
        self,
        name: str,
        **kwargs: dict,
    ) -> InputCollection:
        return self.field(name, "STRING", **kwargs)

    def float(
        self,
        name: str,
        *,
        step: float = 0.001,
        min: float = -10000.0,
        max: float = 10000.0,
        round: bool = False,
        **kwargs: dict,
    ) -> InputCollection:
        return self.field(
            name,
            "FLOAT",
            step=step,
            min=min,
            max=max,
            round=round,
            **kwargs,
        )

    def int(
        self,
        name: str,
        *,
        min: float = -10000,
        max: float = 10000,
        **kwargs: dict,
    ) -> InputCollection:
        return self.field(
            name,
            "INT",
            min=min,
            max=max,
            **kwargs,
        )

    def bool(
        self,
        name: str,
        default: bool = False,
        **kwargs: dict,
    ) -> InputCollection:
        return self.field(name, "BOOLEAN", default=default, **kwargs)

    boolean = bool

    def seed(
        self,
        name: str = "seed",
        *,
        default: int = 0,
        min: int = 0,
        max: int = 0xFFFFFFFFFFFFFFFF,
        tooltip="Seed to use for generated noise",
        **kwargs: dict,
    ) -> InputCollection:
        return self.int(
            name,
            default=default,
            min=min,
            max=max,
            tooltip=tooltip,
            **kwargs,
        )

    def image(self, name: str = "image", **kwargs: dict) -> InputCollection:
        return self.field(name, "IMAGE", **kwargs)

    def latent(self, name: str = "latent", **kwargs: dict) -> InputCollection:
        return self.field(name, "LATENT", **kwargs)

    def conditioning(
        self,
        name: str = "conditioning",
        **kwargs: dict,
    ) -> InputCollection:
        return self.field(name, "CONDITIONING", **kwargs)

    def model(self, name: str = "model", **kwargs: dict) -> InputCollection:
        return self.field(name, "MODEL", **kwargs)

    def sigmas(self, name: str = "sigmas", **kwargs: dict) -> InputCollection:
        return self.field(name, "SIGMAS", **kwargs)

    def sampler(self, name: str = "sampler", **kwargs: dict) -> InputCollection:
        return self.field(name, "SAMPLER", **kwargs)

    def clip(self, name: str = "clip", **kwargs: dict) -> InputCollection:
        return self.field(name, "CLIP", **kwargs)

    def vae(self, name: str = "vae", **kwargs: dict) -> InputCollection:
        return self.field(name, "VAE", **kwargs)


class InputTypes:
    C = TypeVar("C", bound=type)

    def __init__(
        self,
        *,
        parent=None,
        parent_field: str | None = "INPUT_TYPES",
        parent_args=(),
        parent_kwargs=None,
        required: dict | C | None = None,
        optional: dict | C | None = None,
        collection_class: C = InputCollection,
    ):
        if parent is not None and parent_field is not None:
            parent = getattr(parent, parent_field)
        if isinstance(parent, LazyInputTypes):
            parent = parent.get_input_types(
                *parent_args,
                **({} if parent_kwargs is None else parent_kwargs),
            )
        if isinstance(parent, LazyInputTypes):
            raise TypeError("Unexpected multi-level LazyInputTypes parent!")
        if required is None:
            required = {}
        elif isinstance(required, collection_class):
            required = required.to_dict()
        elif not isinstance(required, dict):
            raise TypeError("Bad type for 'required' parameter.")
        if optional is None:
            optional = {}
        elif isinstance(optional, collection_class):
            optional = optional.to_dict()
        elif not isinstance(optional, dict):
            raise TypeError("Bad type for 'optional' parameter.")
        if parent is not None:
            required = parent.required.to_dict() | required
            optional = parent.optional.to_dict() | optional
        self.required = collection_class(**required)
        self.optional = collection_class(**optional)

    def __len__(self) -> int:
        return len(self.required) + len(self.optional)

    def clone(self) -> InputTypes:
        return InputTypes(required=self.required, optional=self.optional)

    def to_dict(self) -> dict:
        return {
            "required": self.required.to_dict(),
            "optional": self.optional.to_dict(),
        }

    def __call__(self) -> dict:
        return self.to_dict()

    def __getattr__(self, key: str):
        if key.startswith("req_"):
            meth = getattr(self.required, key[4:])
        elif key.startswith("opt_"):
            meth = getattr(self.optional, key[4:])
        else:
            errstr = f"Unknown attribute {key} for InputTypes"
            raise AttributeError(errstr)

        def wrapper(*args: list, **kwargs: dict):
            meth(*args, **kwargs)
            return self

        return wrapper


class LazyInputTypes:
    def __init__(self, builder: Callable, initializers=()):
        self._input_types_params = {}
        self._input_types = None
        self.builder = builder
        self.initializers = initializers

    def get_input_types(self, *args: list, **kwargs: dict):
        if args or kwargs:
            args = tuple(args)
            cache_key = (args, tuple(kwargs.items()))
            cached = self._input_types_params.get(cache_key)
        else:
            cache_key = None
            cached = self._input_types
        if cached:
            return cached
        for fun in self.initializers:
            fun()
        result = self.builder(*args, **kwargs)
        if not cache_key:
            self._input_types = result
        else:
            self._input_types_params[cache_key] = result
        return result

    def __call__(self, *args: list, **kwargs: dict) -> dict:
        return self.get_input_types(*args, **kwargs)()
