from .py import freeu_extreme, nodes, powernoise, sonar

sonar.add_samplers()

NODE_CLASS_MAPPINGS = nodes.NODE_CLASS_MAPPINGS | {
    "SonarPowerNoise": powernoise.SonarPowerNoiseNode,
    "SonarPowerFilterNoise": powernoise.SonarPowerFilterNoiseNode,
    "SonarPowerFilter": powernoise.SonarPowerFilterNode,
    "SonarPreviewFilter": powernoise.SonarPreviewFilterNode,
    "FreeUExtremeConfig": freeu_extreme.FreeUExtremeConfigNode,
    "FreeUExtreme": freeu_extreme.FreeUExtremeNode,
}
NODE_DISPLAY_NAME_MAPPINGS = nodes.NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
