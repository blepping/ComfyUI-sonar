# ComfyUI-sonar

Extremely WIP and untested implementation of Sonar sampling. Currently it may not be even close to working properly.

Only supports Euler and Euler Ancestral sampling.

## Description

See https://github.com/Kahsolt/stable-diffusion-webui-sonar for a more in-depth explanation.

The `direction` parameter should (unless I screwed it up) work like setting sign to positive or negative: `1.0` is positive, `-1.0` is negative. You can also potentially play with fractional values.

Like the original documentation says, you normally would not want to set `momentum` to a value below `0.85`. The default values are considered reasonable, doing stuff like using a negative direction may not produce good results.

## Credits

Original implementation: https://github.com/Kahsolt/stable-diffusion-webui-sonar

My version basically just rips off this implementation for Diffusers: https://github.com/alexblattner/modified-euler-samplers-for-sonar-diffusers/
