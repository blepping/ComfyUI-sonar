import contextlib
import importlib

MODULES = {}

with contextlib.suppress(ImportError, NotImplementedError):
    bleh = importlib.import_module("custom_nodes.ComfyUI-bleh")
    bleh_version = getattr(bleh, "BLEH_VERSION", -1)
    if bleh_version < 1:
        raise NotImplementedError
    MODULES["bleh"] = bleh

with contextlib.suppress(ImportError, NotImplementedError):
    import custom_nodes.ComfyUI_restart_sampling as rs

    if not hasattr(rs.restart_sampling, "DEFAULT_SEGMENTS"):
        # Dumb test but this should only exist in restart sampling versions that
        # support plugging in custom noise.
        raise NotImplementedError
    MODULES["restart"] = rs

__all__ = ("MODULES",)
