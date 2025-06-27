from __future__ import annotations

import contextlib
import importlib
import sys
from functools import partial
from typing import TYPE_CHECKING, Callable, NamedTuple

if TYPE_CHECKING:
    from types import ModuleType


class Integrations:
    class Integration(NamedTuple):
        key: str
        module_name: str
        handler: Callable | None = None

    def __init__(self):
        self.initialized = False
        self.modules = {}
        self.init_handlers = []
        self.handlers = []

    def __getitem__(self, key):
        return self.modules[key]

    def __contains__(self, key):
        return key in self.modules

    def __getattr__(self, key):
        return self.modules.get(key)

    @staticmethod
    def get_custom_node(module_name: str, key: str) -> ModuleType | None:
        bi_module = sys.modules.get("_blepping_integrations", {}).get(key)
        if bi_module is not None:
            return bi_module
        module_key = f"custom_nodes.{module_name}"
        with contextlib.suppress(StopIteration):
            spec = importlib.util.find_spec(module_key)
            if spec is None:
                return None
            return next(
                v
                for v in sys.modules.copy().values()
                if hasattr(v, "__spec__")
                and v.__spec__ is not None
                and v.__spec__.origin == spec.origin
            )
        return None

    def register_init_handler(self, handler):
        self.init_handlers.append(handler)

    def register_integration(self, key: str, module_name: str, handler=None) -> None:
        if self.initialized:
            raise ValueError(
                "Internal error: Cannot register integration after initialization",
            )
        if any(item[0] == key or item[1] == module_name for item in self.handlers):
            errstr = (
                f"Module {module_name} ({key}) already in integration handlers list!"
            )
            raise ValueError(errstr)
        self.handlers.append(self.Integration(key, module_name, handler))

    def initialize(self) -> None:
        if self.initialized:
            return
        self.initialized = True
        for ih in self.handlers:
            module = self.get_custom_node(ih.module_name, ih.key)
            if module is None:
                continue
            if ih.handler is not None:
                module = ih.handler(module)
            if module is not None:
                self.modules[ih.key] = module

        for init_handler in self.init_handlers:
            init_handler(self)


class SonarIntegrations(Integrations):
    def __init__(self, *args: list, **kwargs: dict):
        super().__init__(*args, **kwargs)
        self.register_integration("bleh", "ComfyUI-bleh", self.bleh_integration)
        self.register_integration(
            "restart",
            "ComfyUI_restart_sampling",
            self.restart_integration,
        )

    @classmethod
    def bleh_integration(cls, module: ModuleType) -> ModuleType | None:
        bleh_version = getattr(module, "BLEH_VERSION", -1)
        if bleh_version < 1:
            return None
        return module

    @classmethod
    def restart_integration(cls, module: ModuleType) -> ModuleType | None:
        if hasattr(module, "restart_sampling") and hasattr(
            module.restart_sampling,
            "DEFAULT_SEGMENTS",
        ):
            return module
        return None


MODULES = SonarIntegrations()


class IntegratedNode(type):
    @staticmethod
    def wrap_INPUT_TYPES(orig_method: Callable, *args: list, **kwargs: dict) -> dict:
        MODULES.initialize()
        return orig_method(*args, **kwargs)

    def __new__(cls: type, name: str, bases: tuple, attrs: dict) -> object:
        obj = type.__new__(cls, name, bases, attrs)
        if hasattr(obj, "INPUT_TYPES"):
            obj.INPUT_TYPES = partial(cls.wrap_INPUT_TYPES, obj.INPUT_TYPES)
        return obj


__all__ = ("MODULES",)
