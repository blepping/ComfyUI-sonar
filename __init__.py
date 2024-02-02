from .py import sonar

NODE_CLASS_MAPPINGS = {
    "SamplerSonarEuler": sonar.SamplerSonarEuler,
    "SamplerSonarEulerA": sonar.SamplerSonarEulerAncestral,
}

NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
