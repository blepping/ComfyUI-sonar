from __future__ import annotations

import abc
from typing import Any

from .. import noise, utils
from ..external import MODULES, IntegratedNode
from .base_inputtypes import InputCollection, InputTypes, LazyInputTypes

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


class SonarInputCollection(InputCollection):
    def __init__(self, *args: list, **kwargs: dict):
        super().__init__(*args, **kwargs)
        self._DELEGATE_KEYS = self._DELEGATE_KEYS | frozenset((  # noqa: PLR6104
            "customnoise",
            "floatpct",
            "normalizetristate",
            "selectblend",
            "selectnoise",
            "selectscalemode",
            "yaml",
        ))

    def yaml(
        self,
        name: str = "yaml_parameters",
        *,
        tooltip="Allows specifying custom parameters via YAML. Note: When specifying paramaters this way, there is generally not much error checking.",
        placeholder="# YAML or JSON here",
        dynamicPrompts=False,  # noqa: N803
        multiline=True,
        **kwargs: dict,
    ):
        return self.field(
            name,
            "STRING",
            tooltip=tooltip,
            placeholder=placeholder,
            dynamicPrompts=dynamicPrompts,
            multiline=multiline,
            **kwargs,
        )

    def selectblend(
        self,
        name: str = "blend_mode",
        *,
        default="lerp",
        insert_modes=(),
        tooltip="Mode used for blending. If you have ComfyUI-bleh then you will have access to many more blend modes.",
        **kwargs: dict,
    ) -> InputCollection:
        if not MODULES.initialized:
            raise RuntimeError(
                "Attempt to get blending modes before integrations were initialized",
            )
        return self.field(
            name,
            (*insert_modes, *utils.BLENDING_MODES.keys()),
            default=default,
            tooltip=tooltip,
            **kwargs,
        )

    def selectscalemode(
        self,
        name: str,
        *,
        default="nearest-exact",
        insert_modes=(),
        tooltip="Mode used for scaling. If you have ComfyUI-bleh then you will have access to many more scale modes.",
        **kwargs: dict,
    ) -> InputCollection:
        if not MODULES.initialized:
            raise RuntimeError(
                "Attempt to get scale modes before integrations were initialized",
            )
        return self.field(
            name,
            (*insert_modes, *utils.UPSCALE_METHODS),
            default=default,
            tooltip=tooltip,
            **kwargs,
        )

    def selectnoise(
        self,
        name: str,
        *,
        default="gaussian",
        insert_types=(),
        tooltip="Sets the type of noise.",
        **kwargs: dict,
    ) -> InputCollection:
        return self.field(
            name,
            (*insert_types, *noise.NoiseType.get_names()),
            default=default,
            tooltip=tooltip,
            **kwargs,
        )

    def customnoise(
        self,
        name: str,
        add_hint: bool = True,  # noqa: FBT001
        tooltip="Allows connecting a custom noise chain.",
        **kwargs: dict,
    ) -> InputCollection:
        if add_hint:
            tooltip = f"{tooltip}\n{NOISE_INPUT_TYPES_HINT}"
        return self.field(name, WILDCARD_NOISE, tooltip=tooltip, **kwargs)

    def normalizetristate(
        self,
        name: str,
        *,
        default="default",
        tooltip="Controls whether noise is normalized to 1.0 strength.",
        **kwargs: dict,
    ):
        return self.field(
            name,
            ("default", "forced", "disabled"),
            default=default,
            tooltip=tooltip,
            **kwargs,
        )

    def floatpct(self, name: str, *, min=0.0, max=1.0, **kwargs: dict):  # noqa: A002
        return self.float(name=name, min=min, max=max, **kwargs)


class SonarInputTypes(InputTypes):
    _NO_REPLACE = True

    def __init__(self, *args: list, **kwargs: dict):
        super().__init__(
            *args,
            collection_class=SonarInputCollection,
            **kwargs,
        )


class SonarLazyInputTypes(LazyInputTypes):
    _NO_REPLACE = True

    def __init__(self, *args: list, initializers=(MODULES.initialize,), **kwargs: dict):
        super().__init__(
            *args,
            initializers=initializers,
            **kwargs,
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

    INPUT_TYPES = SonarLazyInputTypes(
        lambda *, include_rescale=True, include_chain=True: SonarInputTypes()
        .req_float_factor(
            default=1.0,
            tooltip="Scaling factor for the generated noise of this type.",
        )
        .req_float_rescale(
            _skip=not include_rescale,
            default=0.0,
            min=0.0,
            tooltip="When non-zero, this custom noise item and other custom noise items items connected to it will have their factor scaled to add up to the specified rescale value. When set to 0, rescaling is disabled.",
        )
        .opt_customnoise_sonar_custom_noise_opt(
            _skip=not include_chain,
            tooltip="Optional input for more custom noise items.",
        ),
        initializers=(),
    )

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


class NoiseChainInputTypes(SonarInputTypes):
    def __init__(self, *, parent=SonarCustomNoiseNodeBase, **kwargs: dict):
        super().__init__(parent=parent, **kwargs)


class NoiseNoChainInputTypes(SonarInputTypes):
    def __init__(
        self,
        *,
        parent=SonarCustomNoiseNodeBase,
        parent_args=(),
        parent_kwargs=None,
        **kwargs: dict,
    ):
        super().__init__(
            parent=parent,
            parent_args=parent_args,
            parent_kwargs={"include_chain": False, "include_rescale": False}
            | (parent_kwargs if parent_kwargs is not None else {}),
            **kwargs,
        )


class SonarCustomNoiseNode(SonarCustomNoiseNodeBase):
    INPUT_TYPES = SonarLazyInputTypes(
        lambda: NoiseChainInputTypes().req_selectnoise_noise_type(
            tooltip="Sets the type of noise to generate.",
        ),
    )

    @classmethod
    def get_item_class(cls):
        return noise.CustomNoiseItem


class SonarCustomNoiseAdvNode(SonarCustomNoiseNode):
    DESCRIPTION = "A custom noise item allowing advanced YAML parameter input."

    INPUT_TYPES = SonarLazyInputTypes(
        lambda: NoiseChainInputTypes(parent=SonarCustomNoiseNode).opt_yaml(
            tooltip="Allows specifying custom parameters via YAML. Note: When specifying paramaters this way, there is generally little to no error checking.",
        ),
    )


class SonarNormalizeNoiseNodeMixin:
    @staticmethod
    def get_normalize(val: str) -> bool | None:
        return None if val == "default" else val == "forced"
