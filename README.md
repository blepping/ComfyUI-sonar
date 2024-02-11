# ComfyUI-sonar

A janky implementation of Sonar sampling (momentum-based sampling) for [ComfyUI](https://github.com/comfyanonymous/ComfyUI). It may or may not be working _properly_ but it does produce pretty reasonable results. I am using it personally. At this point, I would say it's suitable for general use with the caveat that it's very likely stuff like implementation and inputs to nodes will still be changing fairly frequently. In other words, don't depend on reproduceable generations with this unless you're willing to keep track of the git revision something was generated with.

Currently supports Euler, Euler Ancestral, and DPM++ SDE sampling.

## Description

See https://github.com/Kahsolt/stable-diffusion-webui-sonar for a more in-depth explanation.

The `direction` parameter should (unless I screwed it up) work like setting sign to positive or negative: `1.0` is positive, `-1.0` is negative. You can also potentially play with fractional values.

Like the original documentation says, you normally would not want to set `momentum` to a value below `0.85`. The default values are considered reasonable, doing stuff like using a negative direction may not produce good results.

## Usage

The most flexible way to use this is with a custom sampler:

![Usage](assets/example_images/custom_sampler_usage.png)

You can also just choose `sonar_euler`, `sonar_euler_ancestral` or `sonar_dpmpp_sde` from the normal samplers list (will use the default settings). I personally recommend using the custom sampler approach and the ancestral version.

## Nodes

1. `SamplerSonarEuler` — Custom sampler node that combines Euler sampling and momentum and optionally guidance. A bit boring compared to the ancestral version but it has predictability going for it. You can possibly try setting init type to `RAND` and using different noise types, however this sampler seems _very_ sensitive to that init type. You may want to set direction to a very low value like `0.05` or `-0.15` when using the `RAND` init type.
2. `SamplerSonarEulerAncestral` — Ancestral version of the above. Same features, just with ancestral Euler.
4. `SonarGuidanceConfig` — You can optionally plug this into the Sonar sampler nodes. See the [Guidance](#guidance) section below.
5. `NoisyLatentLike` — If you give it a latent (or latent batch) it'll return a noisy latent of the same shape. Allows specifying all the custom noise types except `brownian` which has some special requirements. Provided just because the noise generation functions are conveniently available. You can also use this as a reference latent with `SonarGuidanceConfig` node and depending on the strength it can act like variation seed (you'd change the seed in the `NoisyLatentLike` node). *Note*: The seed stuff may or may not work correctly.
6. `SamplerSonarDPMPPSDE` — This one is extra experimental but it is an attempt to add moment and guidance to the DPM++ SDE sampler. It may not work correctly but you can sample stuff with it and get interesting results. I actually really like this one, and you can get away with more extreme stuff like `green_test` noise and still produce reasonable results. You may want to use the `BlehDiscardPenultimateSigma` node from my [ComfyUI-bleh](https://github.com/blepping/ComfyUI-bleh) collection if you find the result seems a bit washed out and b lurry.

## Parameters

Very abbreviated section. The init type can make a big difference. If you use `RANDOM` you can get away with setting `direction` to high values (like up to `2.25` or so) and absurdly low values (like `-30.0`). It's also possible to set `momentum` and `momentum_hist` to negative values, although whether it's a good idea...

## Guidance

You can try the `SamplerSonarNaive` sampler which has an optional latent input. The guidance _probably_ isn't working correctly and the implementation definitely isn't exactly the same as the original A1111 version but it still might be fun to play with. The `linear` guidance type is a lot more sensitive to the `guidance_factor` than the `euler` type. For `euler`, reasonable values are around `0.01` to `0.1`, for `linear` reasonable values are more like `0.001` to `0.02`. It is also possible to set guidance factor to a negative value, I've found this results in high contrast and very vivid colors.

It is possible to set the start and end steps guidance is activate. Rather than setting a low guidance and using it for the whole generation, it's also possible to set high guidance and end it after a relatively low number of steps.

Without guidance it should basically work the same as the ancestral Euler version. There are some example images in the [Examples](#examples) section below.

**Note**: The reference latent needs to be the same size as the one being sampled. Also note that step numbers in the step range are 1-based and inclusive, so 1 is the first step.

## Noise

I basically just copied a bunch of noise functions without really knowing what they do. The main thing I can say is they produce a semi-reasonable result and it's different from the other noise samplers. See [Credits](#credits) below.

1. `gaussian`: This is the default noise type.
2. `uniform`: Might enhance background details?
3. `brownian`: This is the noise type SDE samplers use.
4. `perlin`
5. `studentt`: There's a comment that says it may enhance subject details. It seemed to produce a fairly dark result.
6. `studentt_test`: An experiment that may be removed, it doesn't seem to be adding enough noise. You can possibly compensate by increasing `s_noise`.
7. `pink`
8. `highres_pyramid`: Not extensively tested, but it is slower than the other noise types. I would guess it does something like enhance details.
9. `laplacian`
10. `power`
11. `rainbow_mild` and `rainbow_intense`: A combination of green (-ish, the implementation may be broken) noise plus perlin noise. Very colorful results.
12. `green_test`: Even more rainbow-y than the rainbow noise types. It _probably_ isn't working correctly, but the results are very interesting and colorful. Depending on the model, it may not work well for an initial generation but may be worth trying with img2img type workflows.

You can scroll down to the the [Examples](#examples) section near the bottom to see some example generations with different noise types.

The sampler and `NoisyLatentLike` nodes now take an optional `SonarCustomNoise` input. You can chain `SonarCustomNoise` nodes together to mix different types of noise, similar to how some of the built in ones. It shouldn't matter what order the noise types are chained. If `rescale` is set to `0.0` no rescaling will occur. `factor` is the proportion of that type of noise you want. If you want to use `rescale` it should be on the node that you are plugging into a sampler. Just for example if you had two `SonarCustomNoise` nodes both with `factor=0.7` and `rescale=1.0` on the last one, it would be effectively the same as if you'd used `factor=0.5` and `rescale=1.0` doesn't actually do anything. You can also rescale to values above `1.0` — the result is more noise, similar to increasing `s_noise` above `1.0` on a sampler. The simple explanation is `rescale` means you don't have to make sure the `factor`s add up to the scale you want (which normally would be `1.0`).

**Note**: If you connect the optional `SonarCustomNoise` node to a Sonar sampler or the `NoisyLatentLike` node it will override the noise type selected in the node.

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

#### Laplacian

![Laplacian](assets/example_images/noise/renoise_laplacian.png)

#### Power

![Power](assets/example_images/noise/renoise_power.png)

#### Rainbow Mild

![Rainbow Mild](assets/example_images/noise/renoise_rainbow_mild.png)

#### Rainbow Intense

![Rainbow Intense](assets/example_images/noise/renoise_rainbow_intense.png)

#### Green_test_

![Green_test](assets/example_images/noise/renoise_green_test.png)

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

#### Laplacian

![Laplacian](assets/example_images/noise/noise_laplacian.png)

#### Power

![Power](assets/example_images/noise/noise_power.png)

#### Rainbow Mild

![Rainbow Mild](assets/example_images/noise/noise_rainbow_mild.png)

#### Rainbow Intense

![Rainbow Intense](assets/example_images/noise/noise_rainbow_intense.png)

#### Green_test

This might seem too crazy for actual use, but you can actually get decent results using the DPMPP Sonar sampler and a relatively high step count.

![Green_test](assets/example_images/noise/noise_green_test.png)

</details>
