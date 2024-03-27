from .py import nodes, powernoise, sonar

sonar.add_samplers()

NODE_CLASS_MAPPINGS = {
    "SamplerSonarEuler": nodes.SamplerNodeSonarEuler,
    "SamplerSonarEulerA": nodes.SamplerNodeSonarEulerAncestral,
    "SamplerSonarDPMPPSDE": nodes.SamplerNodeSonarDPMPPSDE,
    "SamplerConfigOverride": nodes.SamplerNodeConfigOverride,
    "NoisyLatentLike": nodes.NoisyLatentLikeNode,
    "SonarCustomNoise": nodes.SonarCustomNoiseNode,
    "SonarPowerNoise": powernoise.SonarPowerNoiseNode,
    "SonarGuidanceConfig": nodes.GuidanceConfigNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {}

if hasattr(nodes, "KRestartSamplerCustomNoise"):
    NODE_CLASS_MAPPINGS["KRestartSamplerCustomNoise"] = nodes.KRestartSamplerCustomNoise

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
