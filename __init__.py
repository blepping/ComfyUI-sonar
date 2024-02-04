from .py import sonar

sonar.add_samplers()

NODE_CLASS_MAPPINGS = {
    "SamplerSonarEuler": sonar.SamplerNodeSonarEuler,
    "SamplerSonarEulerA": sonar.SamplerNodeSonarEulerAncestral,
    "SamplerSonarNaive": sonar.SamplerNodeSonarNaive,
}

NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
