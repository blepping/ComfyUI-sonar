# ruff: noqa: TID252
from __future__ import annotations

import abc
from typing import Any

from .. import noise
from ..external import IntegratedNode

try:
    from comfy_execution import validation as comfy_validation

    if not hasattr(comfy_validation, "validate_node_input"):
        raise NotImplementedError  # noqa: TRY301
    HAVE_COMFY_UNION_TYPE = comfy_validation.validate_node_input("B", "A,B")
except (ImportError, NotImplementedError):
    HAVE_COMFY_UNION_TYPE = False
except Exception as exc:  # noqa: BLE001
    HAVE_COMFY_UNION_TYPE = False
    print(
        f"** ComfyUI-sonar: Warning, caught unexpected exception trying to detect ComfyUI union type support. Disabling. Exception: {exc}",
    )

NOISE_INPUT_TYPES = frozenset(("SONAR_CUSTOM_NOISE", "OCS_NOISE"))

if not HAVE_COMFY_UNION_TYPE:

    class Wildcard(str):  # noqa: FURB189
        __slots__ = ("whitelist",)

        @classmethod
        def __new__(cls, s, *args: list, whitelist=None, **kwargs: dict):
            result = super().__new__(s, *args, **kwargs)
            result.whitelist = whitelist
            return result

        def __ne__(self, other):
            return False if self.whitelist is None else other not in self.whitelist

    WILDCARD_NOISE = Wildcard("*", whitelist=NOISE_INPUT_TYPES)
else:
    WILDCARD_NOISE = ",".join(NOISE_INPUT_TYPES)


NOISE_INPUT_TYPES_HINT = (
    f"The following input types are supported: {', '.join(NOISE_INPUT_TYPES)}"
)


class SonarCustomNoiseNodeBase(metaclass=IntegratedNode):
    DESCRIPTION = "A custom noise item."
    RETURN_TYPES = ("SONAR_CUSTOM_NOISE",)
    OUTPUT_TOOLTIPS = ("A custom noise chain.",)
    CATEGORY = "advanced/noise"
    FUNCTION = "go"

    @abc.abstractmethod
    def get_item_class(self):
        raise NotImplementedError

    @classmethod
    def INPUT_TYPES(cls, *, include_rescale=True, include_chain=True):
        result = {
            "required": {
                "factor": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -10000.0,
                        "max": 10000.0,
                        "step": 0.001,
                        "round": False,
                        "tooltip": "Scaling factor for the generated noise of this type.",
                    },
                ),
            },
            "optional": {},
        }
        if include_rescale:
            result["required"] |= {
                "rescale": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 10000.0,
                        "step": 0.001,
                        "round": False,
                        "tooltip": "When non-zero, this custom noise item and other custom noise items items connected to it will have their factor scaled to add up to the specified rescale value. When set to 0, rescaling is disabled.",
                    },
                ),
            }
        if include_chain:
            result["optional"] |= {
                "sonar_custom_noise_opt": (
                    WILDCARD_NOISE,
                    {
                        "tooltip": f"Optional input for more custom noise items.\n{NOISE_INPUT_TYPES_HINT}",
                    },
                ),
            }
        return result

    def go(
        self,
        factor=1.0,
        rescale=0.0,
        sonar_custom_noise_opt=None,
        **kwargs: dict[str, Any],
    ):
        nis = (
            sonar_custom_noise_opt.clone()
            if sonar_custom_noise_opt
            else noise.CustomNoiseChain()
        )
        if factor != 0:
            nis.add(self.get_item_class()(factor, **kwargs))
        return (nis if rescale == 0 else nis.rescaled(rescale),)


class SonarCustomNoiseNode(SonarCustomNoiseNodeBase):
    @classmethod
    def INPUT_TYPES(cls):
        result = super().INPUT_TYPES()
        result["required"] |= {
            "noise_type": (
                tuple(noise.NoiseType.get_names()),
                {
                    "tooltip": "Sets the type of noise to generate.",
                },
            ),
        }
        return result

    @classmethod
    def get_item_class(cls):
        return noise.CustomNoiseItem


class SonarCustomNoiseAdvNode(SonarCustomNoiseNode):
    DESCRIPTION = "A custom noise item allowing advanced YAML parameter input."

    @classmethod
    def INPUT_TYPES(cls):
        result = super().INPUT_TYPES()
        result["optional"] |= {
            "yaml_parameters": (
                "STRING",
                {
                    "tooltip": "Allows specifying custom parameters via YAML. Note: When specifying paramaters this way, there is no error checking.",
                    "placeholder": "# YAML or JSON here",
                    "dynamicPrompts": False,
                    "multiline": True,
                },
            ),
        }
        return result


class SonarNormalizeNoiseNodeMixin:
    @staticmethod
    def get_normalize(val: str) -> bool | None:
        return None if val == "default" else val == "forced"
