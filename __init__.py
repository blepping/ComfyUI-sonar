from .py import nodes, sonar

sonar.add_samplers()

NODE_CLASS_MAPPINGS = {
    "SamplerSonarEuler": nodes.SamplerNodeSonarEuler,
    "SamplerSonarEulerA": nodes.SamplerNodeSonarEulerAncestral,
    "SamplerSonarNaive": nodes.SamplerNodeSonarNaive,
    "SonarGuidanceConfig": nodes.GuidanceConfigNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
