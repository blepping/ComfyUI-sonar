from . import (
    base,
    integrations,
    latent_operations,
    misc,
    momentum_samplers,
    noise_filters,
    noise_types,
)

NODE_CLASS_MAPPINGS = {
    "SonarCustomNoise": base.SonarCustomNoiseNode,
    "SonarCustomNoiseAdv": base.SonarCustomNoiseAdvNode,
} | (
    integrations.NODE_CLASS_MAPPINGS
    | latent_operations.NODE_CLASS_MAPPINGS
    | misc.NODE_CLASS_MAPPINGS
    | momentum_samplers.NODE_CLASS_MAPPINGS
    | noise_filters.NODE_CLASS_MAPPINGS
    | noise_types.NODE_CLASS_MAPPINGS
)


NODE_DISPLAY_NAME_MAPPINGS = (
    getattr(integrations, "NODE_DISPLAY_NAME_MAPPINGS", {})
    | getattr(latent_operations, "NODE_DISPLAY_NAME_MAPPINGS", {})
    | getattr(misc, "NODE_DISPLAY_NAME_MAPPINGS", {})
    | getattr(momentum_samplers, "NODE_DISPLAY_NAME_MAPPINGS", {})
    | getattr(noise_filters, "NODE_DISPLAY_NAME_MAPPINGS", {})
    | getattr(noise_types, "NODE_DISPLAY_NAME_MAPPINGS", {})
)
