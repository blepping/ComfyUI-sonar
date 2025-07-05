import sys

from . import py  # noqa: F401
from .py import freeu_extreme, nodes, powernoise, sonar


def blep_init():
    bi = sys.modules.get("_blepping_integrations", {})
    if "sonar" in bi:
        return
    bi["sonar"] = sys.modules[__name__]
    sys.modules["_blepping_integrations"] = bi


sonar.add_samplers()
blep_init()

NODE_CLASS_MAPPINGS = (
    nodes.NODE_CLASS_MAPPINGS
    | powernoise.NODE_CLASS_MAPPINGS
    | freeu_extreme.NODE_CLASS_MAPPINGS
)
NODE_DISPLAY_NAME_MAPPINGS = nodes.NODE_DISPLAY_NAME_MAPPINGS
NODE_DISPLAY_NAME_MAPPINGS = (
    getattr(nodes, "NODE_DISPLAY_NAME_MAPPINGS", {})
    | getattr(powernoise, "NODE_DISPLAY_NAME_MAPPINGS", {})
    | getattr(freeu_extreme, "NODE_DISPLAY_NAME_MAPPINGS", {})
)


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
