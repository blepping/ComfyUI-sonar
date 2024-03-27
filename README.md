# ComfyUI-sonar

A janky implementation of Sonar sampling (momentum-based sampling) for [ComfyUI](https://github.com/comfyanonymous/ComfyUI). It may or may not be working _properly_ but it does produce pretty reasonable results. I am using it personally. At this point, I would say it's suitable for general use with the caveat that it's very likely stuff like implementation and inputs to nodes will still be changing fairly frequently. In other words, don't depend on reproduceable generations with this unless you're willing to keep track of the git revision something was generated with.

Currently supports Euler, Euler Ancestral, and DPM++ SDE sampling.

See the [ChangeLog](changelog.md) for recent user-visible changes.

## Description

See https://github.com/Kahsolt/stable-diffusion-webui-sonar for a more in-depth explanation.

The `direction` parameter should (unless I screwed it up) work like setting sign to positive or negative: `1.0` is positive, `-1.0` is negative. You can also potentially play with fractional values.

Like the original documentation says, you normally would not want to set `momentum` to a value below `0.85`. The default values are considered reasonable, doing stuff like using a negative direction may not produce good results.

## Usage

The most flexible way to use this is with a custom sampler:

![Usage](assets/example_images/custom_sampler_usage.png)

You can also just choose `sonar_euler`, `sonar_euler_ancestral` or `sonar_dpmpp_sde` from the normal samplers list (will use the default settings). I personally recommend using the custom sampler approach and the ancestral version.

## Nodes

### `SamplerSonarEuler`

Custom sampler node that combines Euler sampling and momentum and optionally guidance. A bit boring compared to the ancestral version but it has predictability going for it. You can possibly try setting init type to `RAND` and using different noise types, however this sampler seems _very_ sensitive to that init type. You may want to set direction to a very low value like `0.05` or `-0.15` when using the `RAND` init type. Setting `momentum=1` is the same as disabling momentum, so this sampler with `momentum=1` is basically the same as the basic `euler` sampler.

### `SamplerSonarEulerAncestral`

Ancestral version of the above. Same features, just with ancestral Euler.

### `SamplerSonarDPMPPSDE`

Attempt to add momentum and guidance to the DPM++ SDE sampler. It may not work correctly but you can sample stuff with it and get interesting results. I actually really like this one, and you can get away with more extreme stuff like `green_test` noise and still produce reasonable results. You may want to use the `BlehDiscardPenultimateSigma` node from my [ComfyUI-bleh](https://github.com/blepping/ComfyUI-bleh) collection if you find the result seems a bit washed out and blurry.

### `SonarGuidanceConfig`

You can optionally plug this into the Sonar sampler nodes. See the [Guidance](#guidance) section below.

### `NoisyLatentLike`

This node takes a reference latent and generates noise of the same shape. The one required input is `latent`.

You can connect a `SonarCustomNoise` or `SonerPowerNoise` node to the `custom_noise_opt` input: if that is attached, the built in noise type selector is ignored. The generated noise will be multiplied by the `multiplier` value. Note that you cannot use `brownian` noise whether specified directly or via custom noise nodes.

The node has two main modes: simply generate and scale the noise by the multiplier and return or add it to the input latent. In this mode, you don't connect anything to the `mul_by_sigmas_opt` or `model_opt` inputs and you would use other nodes to calculate the correct strength.

In the second mode you must connect sigmas (for example from a `BasicScheduler` node) to the `mul_by_sigmas_opt` input and connect a model to the `model_opt` input. It will calculate the strength based on the first item in the list of sigmas (so you could use something like a `SplitSigmas` node to slice them as needed). Note that `multiplier` still applies: the calculated strength will be scaled by it. This second mode is generally this is the most convenient way to use the node since the two main uses cases are: making a latent with initial noise or adding noise to a latent (for img2img type stuff).

If you want to create noise for initial sampling, connect model and sigmas to the node, connect an empty latent (or one of the appropriate size) to it and that is basically all you need to do (aside from configuring the noise types). For img2img (upscaling, etc), either slice the sigmas at the appropriate or set a denoise in something like the `BasicScheduler` node. **Note**: You also need to turn on the `add_to_latent` toggle. Turning this on doesn't matter for initial noise since an empty latent is all zeros.


### `SamplerConfigOverride`

can be used to override configuration settings for other samplers, including the noise type. For example, you could force `euler_ancestral` to use a different noise type. It's also possible to override other settings like `s_noise`, etc. *Note*: The wrapper inspects the sampling function's arguments to see what it supports, so you should connect the sampler directly to this rather than having other nodes (like a different sampler wrapper) in between.

### `SonarCustomNoise`

See the [Noise](#noise) section below for information on noise types.

### `SonarPowerNoise`

This node generates [fractional Brownian motion (fBm) noise](https://en.wikipedia.org/wiki/Fractional_Brownian_motion#Frequency-domain_interpretation). It offers versatility in producing various types of noise including gaussian, pink, 2D brownian noise, and all intermediates.

By default, the node generates normal gaussian noise.

<details>

<summary>Expand detailed explanation</summary>


Here's an overview of its parameters:

- `factor` and `rescale` operate similarly to `SonarCustomNoise`, enabling the addition of multiple sources of noises.
- `time_brownian` introduces correlation across sampler timesteps for SDE solvers.
- `alpha` is the main parameter. `alpha > 0` amplifies low frequencies; `alpha = 1` yields pink noise, and `alpha = 2` produces brownian noise. Conversely, for `alpha < 0`, it amplifies high frequencies.
- `min_freq` and `max_freq` determine the range of frequencies allowed through. Setting `max_freq = `$\sqrt{1/2} \simeq 0.7071$ enables the passage of the highest frequencies. In cases where `alpha < 0`, setting `max_freq = 0.5` is advisable to diminish the power of diagonally oriented frequencies.
- `stretch`, `rotate`, and `pnorm` alter the filter's shape by stretching, rotating, or cushioning the band-pass region.
- Lowering `mix` moderates the filter's effect by blending back unfiltered gaussian noise from the same sample.
- `common_mode` is an attempt to desaturate the latent by injecting the average across channels into every latent channel. However, this may result in a specific color due to the encoding of the unit vector by the latent space. Note that this is done _after_ the `mix`ing of unfiltered gaussian noise.
- Enabling `preview` provides a visual representation of the filter. `no_mix` sets `mix = 1` for the preview. The preview includes, from left to right:
  - Fourier domain visualization: Low frequencies at the center, with black indicating filtered-out frequencies.
  - Spatial visualization of the 2D kernel: The filtering can be interpreted as convolution with the displayed kernel.
  - Sample: Gaussian sample with shaped frequency spectrum. A single latent channel will look like this.

**Frequency-domain Interpretation**: The Fourier transform decomposes a 2D latent into sinusoids covering all spatial orientations and frequencies. For an independent and identically distributed gaussian sample, energy is evenly distributed across all frequencies and orientations. Scaling the power spectrum by $1 / f^\alpha$, where $\alpha>0$, boosts low frequencies, introducing spatial correlations.

**Spatial Domain Interpretation**: A gaussian latent sample comprises independently sampled pixels, exhibiting no spatial correlations. Conversely, a requirement that each pixel value differs from its neighbors by a $\epsilon \sim \mathcal{N}(0, 1)$ results in 2D brownian noise ($\alpha=2$).

**Seed Considerations**: While the node defaults to outputting gaussian noise, a given seed produce a different sample than the one produced by other gaussian noise sources. This stems from sampling the noise directly in the frequency domain to avoid the cost of a FFT. When `time_brownian = true`, noise sampling occurs in the spatial domain, ensuring that default parameters yield output equivalent to `SonarCustomNoise` set to `brownian`.

</details>

From a usage perspective, using positive alpha will tend to create a colorful effect, using negative alpha will create line/streak like artifacts sort of like an oil painting canvas. Start with small values at first (`-0.1`, `0.1`) and adjust as necessary. `time_brownian` makes the effect of power noise (and alpha) stronger - also note that it can only be used when sampling and not for `NoisyLatentLike`. Setting `common_mode` also generally seems to intensify these effects. Different types of models (normal EPS models, v-prediction models, SDXL) generally react differently to these exotic noise types so my advice is to experiment! Lowering `mix` uses normal gaussian noise for part of the generated noise. For example, `mix=1.0` means 100% power noise, `mix=0.5` means 50/50 power noise and normal gaussian noise. This also is about the same as setting factor to `0.5` and plugging in a `SonarCustomNoise` node with factor at `0.5` also and the type set to `guassian`.

Noise from the `SonarCustomNoise` node and `SonarPowerNoise` can be freely mixed.

### `KRestartSamplerCustomNoise`

If you have a recent enough version of [ComfyUI_restart_sampling](https://github.com/ssitu/ComfyUI_restart_sampling/)
installed, you'll also get the `KRestartSamplerCustomNoise` node which is exactly the same as `KRestartSamplerCustom`
except for adding an optional custom noise input.
See the restart sampling repo for more information: https://github.com/ssitu/ComfyUI_restart_sampling

## Sonar Sampler Parameters

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
6. `pink`
7. `highres_pyramid`: Not extensively tested, but it is slower than the other noise types. I would guess it does something like enhance details.
8. `laplacian`
9. `power`
10. `rainbow_mild` and `rainbow_intense`: A combination of green (-ish, the implementation may be broken) noise plus perlin noise. Very colorful results.
11. `green_test`: Even more rainbow-y than the rainbow noise types. It _probably_ isn't working correctly, but the results are very interesting and colorful. Depending on the model, it may not work well for an initial generation but may be worth trying with img2img type workflows.

You can scroll down to the the [Examples](#examples) section near the bottom to see some example generations with different noise types.

The sampler and `NoisyLatentLike` nodes now take an optional `SonarCustomNoise` input. You can chain `SonarCustomNoise` nodes together to mix different types of noise, similar to how some of the built in ones. It shouldn't matter what order the noise types are chained. If `rescale` is set to `0.0` no rescaling will occur. `factor` is the proportion of that type of noise you want. If you want to use `rescale` it should be on the node that you are plugging into a sampler. Just for example if you had two `SonarCustomNoise` nodes both with `factor=0.7` and `rescale=1.0` on the last one, it would be effectively the same as if you'd used `factor=0.5` and `rescale=1.0` doesn't actually do anything. You can also rescale to values above `1.0` — the result is more noise, similar to increasing `s_noise` above `1.0` on a sampler. The simple explanation is `rescale` means you don't have to make sure the `factor`s add up to the scale you want (which normally would be `1.0`).

**Note**: If you connect the optional `SonarCustomNoise` node to a Sonar sampler, the `NoisyLatentLike` node or the `SamplerConfigOverride` node, it will override the noise type selected in the node.



## Related

I also have some other ComfyUI nodes here: https://github.com/blepping/ComfyUI-bleh/

## Credits

Original Sonar Sampler implementation (for A1111): https://github.com/Kahsolt/stable-diffusion-webui-sonar

My version was initially based on this Sonar sampler implementation for Diffusers: https://github.com/alexblattner/modified-euler-samplers-for-sonar-diffusers/

Many noise generation functions copied from https://github.com/Clybius/ComfyUI-Extra-Samplers with only minor modifications. I may have broken some of them in the process _or_ they may not have been suitable for use and I took them anyway. If they don't work it is not a reflection on the original source.

`SonarPowerNoise` contributed by [elias-gaeros](https://github.com/elias-gaeros/). Thanks!

## Examples

Unfortunately, right now these examples are somewhat incomplete and out of date. I hope to update them when I get the time.

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

**outdated**

![StudentT](assets/example_images/noise/renoise_studentt.png)


#### StudentT_test

**outdated**

![StudentT_test](assets/example_images/noise/renoise_studentt_test.png)

#### Laplacian

![Laplacian](assets/example_images/noise/renoise_laplacian.png)

#### Power

![Power](assets/example_images/noise/renoise_power.png)

#### Rainbow Mild

![Rainbow Mild](assets/example_images/noise/renoise_rainbow_mild.png)

#### Rainbow Intense

![Rainbow Intense](assets/example_images/noise/renoise_rainbow_intense.png)

#### Green_test

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

**outdated**

![StudentT](assets/example_images/noise/noise_studentt.png)

#### StudentT_test

**outdated**

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
