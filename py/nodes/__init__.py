from . import (
    base,
    freeu_extreme,
    integrations,
    latent_operations,
    misc,
    momentum_samplers,
    noise_filters,
    noise_types,
    powernoise,
)

NODE_CLASS_MAPPINGS = {
    "SonarCustomNoise": base.SonarCustomNoiseNode,
    "SonarCustomNoiseAdv": base.SonarCustomNoiseAdvNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {}

for nm in (
    freeu_extreme,
    integrations,
    latent_operations,
    misc,
    momentum_samplers,
    noise_filters,
    noise_types,
    powernoise,
):
    NODE_CLASS_MAPPINGS |= getattr(nm, "NODE_CLASS_MAPPINGS", {})
    NODE_DISPLAY_NAME_MAPPINGS |= getattr(nm, "NODE_DISPLAY_NAME_MAPPINGS", {})
