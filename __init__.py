from .py import sonar

NODE_CLASS_MAPPINGS = {
    "SamplerSonarEuler": sonar.SamplerNodeSonarEuler,
    "SamplerSonarEulerA": sonar.SamplerNodeSonarEulerAncestral,
}

NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
