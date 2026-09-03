# ruff: noqa: A002
from __future__ import annotations

from copy import deepcopy
from functools import partial
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

bi_int = int
bi_bool = bool
bi_float = float


class InputCollection:
    _DELEGATE_KEYS = frozenset(
        (
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
        ),
    )

    def __init__(self, **kwargs: Any):
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

    def __len__(self) -> bi_int:
        return len(self.fields)

    def __contains__(self, key: str) -> bi_bool:
        return key in self.fields

    def field(
        self,
        name: str,
        type: str | tuple,
        *,
        _skip: bi_bool = False,
        **kwargs: Any,
    ) -> InputCollection:
        if not _skip:
            self.fields[name] = (type,) if not kwargs else (type, kwargs)
        return self

    def string(
        self,
        name: str,
        **kwargs: Any,
    ) -> InputCollection:
        return self.field(name, "STRING", **kwargs)

    def float(
        self,
        name: str,
        *,
        step: bi_float = 0.001,
        min: bi_float = -10000.0,
        max: bi_float = 10000.0,
        round: bi_bool = False,
        **kwargs: Any,
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
        min: bi_float = -10000,
        max: bi_float = 10000,
        **kwargs: Any,
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
        default: bi_bool = False,
        **kwargs: Any,
    ) -> InputCollection:
        return self.field(name, "BOOLEAN", default=default, **kwargs)

    boolean = bool  # noqa: A003

    def seed(
        self,
        name: str = "seed",
        *,
        default: bi_int = 0,
        min: bi_int = 0,
        max: bi_int = 0xFFFFFFFFFFFFFFFF,
        tooltip="Seed to use for generated noise",
        **kwargs: Any,
    ) -> InputCollection:
        return self.int(
            name,
            default=default,
            min=min,
            max=max,
            tooltip=tooltip,
            **kwargs,
        )

    def image(self, name: str = "image", **kwargs: Any) -> InputCollection:
        return self.field(name, "IMAGE", **kwargs)

    def latent(self, name: str = "latent", **kwargs: Any) -> InputCollection:
        return self.field(name, "LATENT", **kwargs)

    def conditioning(
        self,
        name: str = "conditioning",
        **kwargs: Any,
    ) -> InputCollection:
        return self.field(name, "CONDITIONING", **kwargs)

    def model(self, name: str = "model", **kwargs: Any) -> InputCollection:
        return self.field(name, "MODEL", **kwargs)

    def sigmas(self, name: str = "sigmas", **kwargs: Any) -> InputCollection:
        return self.field(name, "SIGMAS", **kwargs)

    def sampler(self, name: str = "sampler", **kwargs: Any) -> InputCollection:
        return self.field(name, "SAMPLER", **kwargs)

    def clip(self, name: str = "clip", **kwargs: Any) -> InputCollection:
        return self.field(name, "CLIP", **kwargs)

    def vae(self, name: str = "vae", **kwargs: Any) -> InputCollection:
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

        def wrapper(*args: Any, **kwargs: Any):
            meth(*args, **kwargs)
            return self

        return wrapper


class LazyInputTypes:
    def __init__(self, builder: Callable, initializers=()):
        self._input_types_params = {}
        self._input_types = None
        self.builder = builder
        self.initializers = initializers

    def get_input_types(self, *args: Any, **kwargs: Any):
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

    def __call__(self, *args: Any, **kwargs: Any) -> dict:
        return self.get_input_types(*args, **kwargs)()
