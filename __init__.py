from .py import nodes, powernoise, sonar

sonar.add_samplers()

NODE_CLASS_MAPPINGS = nodes.NODE_CLASS_MAPPINGS | {
    "SonarPowerNoise": powernoise.SonarPowerNoiseNode,
}
NODE_DISPLAY_NAME_MAPPINGS = nodes.NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
