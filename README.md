# ComfyUI-sonar

Extremely WIP and untested implementation of Sonar sampling for [ComfyUI](https://github.com/comfyanonymous/ComfyUI). Currently it may not be even close to working _properly_ but it does produce pretty reasonable results.

Only supports Euler and Euler Ancestral sampling.

## Description

See https://github.com/Kahsolt/stable-diffusion-webui-sonar for a more in-depth explanation.

The `direction` parameter should (unless I screwed it up) work like setting sign to positive or negative: `1.0` is positive, `-1.0` is negative. You can also potentially play with fractional values.

Like the original documentation says, you normally would not want to set `momentum` to a value below `0.85`. The default values are considered reasonable, doing stuff like using a negative direction may not produce good results.

## Usage

The most flexible way to use this is with a custom sampler:

![Usage](assets/example_images/custom_sampler_usage.png)

You can also just choose `sonar_euler` or `sonar_euler_ancestral` from the normal samplers list (will use the default settings). I personally recommend using the custom sampler approach and the ancestral version.

## Parameters

Very abbreviated section. The init type can make a big difference. If you use `RANDOM` you can get away with setting `direction` to high values (like up to `2.25` or so) and absurdly low values (like `-30.0`). It's also possible to set `momentum` and `momentum_hist` to negative values, although whether it's a good idea...

## Guidance

You can try the `SamplerSonarNaive` sampler which has an optional latent input. The guidance _probably_ isn't working correctly and the implementation definitely isn't exactly the same as the original A1111 version but it still might be fun to play with. The `linear` guidance type is a lot more sensitive to the `guidance_factor` than the `euler` type. For `euler`, reasonable values are around `0.01` to `0.1`, for `linear` reasonable values are more like `0.001` to `0.02`. It is also possible to set guidance factor to a negative value, I've found this results in high contrast and very vivid colors.

It is possible to set the start and end steps guidance is activate. Rather than setting a low guidance and using it for the whole generation, it's also possible to set high guidance and end it after a relatively low number of steps.

Without guidance it should basically work the same as the ancestral Euler version. There are some example images in the examples section below.

**Note**: The reference latent needs to be the same size as the one being sampled. Also note that step numbers in the step range are 1-based and inclusive, so 1 is the first step.

## Noise

I basically just copied a bunch of noise functions without really knowing what they do. The main thing I can say is they produce a semi-reasonable result and it's different from the other noise samplers. See credits below.

1. `gaussian`: This is the default noise type.
2. `uniform`: Might enhance background details?
3. `brownian`: This is the noise type SDE samplers use.
4. `perlin`
5. `studentt`: There's a comment that says it may enhance subject details. It seemed to produce a fairly dark result.
6. `studentt_test`: An experiment that may be removed, it doesn't seem to be adding enough noise. You can possibly compensate by increasing `s_noise`.
7. `pink`
8. `highres_pyramid`: Not extensively tested, but it is slower than the other noise types. I would guess it does something like enhance details.

You can scroll down to the the examples section near the bottom to see some example generations with different noise types.

## Credits

Original Sonar Sampler implementation (for A1111): https://github.com/Kahsolt/stable-diffusion-webui-sonar

My version basically just rips off this Sonar sampler implementation for Diffusers: https://github.com/alexblattner/modified-euler-samplers-for-sonar-diffusers/

Noise generation functions copied from https://github.com/Clybius/ComfyUI-Extra-Samplers with only minor modifications. I may have broken some of them in the process _or_ they may not have been suitable for use and I took them anyway. If they don't work it is not a reflection on the original source.

## Examples

### Guidance

<details>
<summary>Expand guidance example images</summary>

#### Positive

Using the `linear` guidance type and `guidance_factor=0.02`. The reference image was a red and blue checkboard pattern.

![Positive](assets/example_images/guidance/guidance_linear_pos.png)

#### Negative

Using the `linear` guidance type and `guidance_factor=-0.015`. The reference image was a red and blue checkboard pattern.

![Positive](assets/example_images/guidance/guidance_linear_neg.png)

</details>


### Noise Types (img2img)

These were generated with `s_noise=1.05` to make the noise effect more pronounced, 30 steps at `0.66` denoise, sonar settings increased slightly to enhance the effect (`momentum=0.9, momentum_hist=0.85, direction=1.0, momentum_init=ZERO`). It is probably easier to compare using these as the image _mostly_ stays the same as the sonar sampler settings change.

<details>
<summary>Expand renoise example images</summary>

#### Base

Base image - no Sonar Sampler steps.

![Base](assets/example_images/noise/renoise_base.png)

#### Euler A

Normal (non-sonar) Eular A. Not really a comparison with noise (think it would use gaussian) but with the difference in effect from momentum.

![Euler A](assets/example_images/noise/renoise_eulera.png)


#### Gaussian

![Gaussian](assets/example_images/noise/renoise_gaussian.png)

#### Brownian

![Brownian](assets/example_images/noise/renoise_brownian.png)

#### Perlin

![Perlin](assets/example_images/noise/renoise_perlin.png)

#### Uniform

![Uniform](assets/example_images/noise/renoise_uniform.png)

#### Highres Pyramid

![Highres_pyramid](assets/example_images/noise/renoise_highres_pyramid.png)

#### Pink

![Pink](assets/example_images/noise/renoise_pink.png)

#### StudentT

![StudentT](assets/example_images/noise/renoise_studentt.png)


#### StudentT_test

![StudentT_test](assets/example_images/noise/renoise_studentt_test.png)

</details>

### Noise Types (Initial Generations)

These were generated with `s_noise=1.1` to make the noise effect more pronounced, default sonar settings (`momentum=0.95, momentum_hist=0.75, direction=1.0, momentum_init=ZERO`). It may be harder to see the noise effects since the composition can change a lot in initial generations.

<details>
<summary>Expand initial generation example images</summary>

#### Gaussian

![Gaussian](assets/example_images/noise/noise_gaussian.png)

#### Brownian

![Brownian](assets/example_images/noise/noise_brownian.png)

#### Perlin

![Perlin](assets/example_images/noise/noise_perlin.png)

#### Uniform

![Uniform](assets/example_images/noise/noise_uniform.png)

#### Highres Pyramid

![Highres_pyramid](assets/example_images/noise/noise_highres_pyramid.png)

#### Pink

![Pink](assets/example_images/noise/noise_pink.png)

#### StudentT

![StudentT](assets/example_images/noise/noise_studentt.png)


#### StudentT_test

![StudentT_test](assets/example_images/noise/noise_studentt_test.png)

</details>
