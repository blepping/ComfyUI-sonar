# Advanced Power Noise

## `SonarPowerNoise`

This node generates [fractional Brownian motion (fBm) noise](https://en.wikipedia.org/wiki/Fractional_Brownian_motion#Frequency-domain_interpretation). It offers versatility in producing various types of noise including gaussian, pink, 2D brownian noise, and all intermediates.

By default, the node generates normal gaussian noise.

<details>

<summary>⭐⭐ Expand advanced parameter explanation ⭐⭐</summary>

Here's an overview of its parameters:

- `factor` and `rescale` operate similarly to `SonarCustomNoise`, enabling the addition of multiple sources of noises.
- `time_brownian` introduces correlation across sampler timesteps for SDE solvers.
- `alpha` is the main parameter. `alpha > 0` amplifies low frequencies; `alpha = 1` yields pink noise, and `alpha = 2` produces brownian noise. Conversely, for `alpha < 0`, it amplifies high frequencies.
- `min_freq` and `max_freq` determine the range of frequencies allowed through. Setting `max_freq = `$\sqrt{1/2} \simeq 0.7071$ enables the passage of the highest frequencies. In cases where `alpha < 0`, setting `max_freq = 0.5` is advisable to diminish the power of diagonally oriented frequencies.
- `stretch`, `rotate`, and `pnorm` alter the filter's shape by stretching, rotating, or cushioning the band-pass region.
- Lowering `mix` moderates the filter's effect by blending back unfiltered gaussian noise from the same sample.
- `common_mode` is an attempt to desaturate the latent by injecting the average across channels into every latent channel. **FIXME: it's not all channels anymore** However, this may result in a specific color due to the encoding of the unit vector by the latent space. Note that this is done _after_ the `mix`ing of unfiltered gaussian noise.
- `channel_correlation` **FIXME**: TBD
- Enabling `preview` provides a visual representation of the filter. `no_mix` sets `mix = 1` for the preview. The preview includes, from left to right:
  - Fourier domain visualization: Low frequencies at the center, with black indicating filtered-out frequencies.
  - Spatial visualization of the 2D kernel: The filtering can be interpreted as convolution with the displayed kernel.
  - Sample: Gaussian sample with shaped frequency spectrum. A single latent channel will look like this.

**Frequency-domain Interpretation**: The Fourier transform decomposes a 2D latent into sinusoids covering all spatial orientations and frequencies. For an independent and identically distributed gaussian sample, energy is evenly distributed across all frequencies and orientations. Scaling the power spectrum by $1 / f^\alpha$, where $\alpha>0$, boosts low frequencies, introducing spatial correlations.

**Spatial Domain Interpretation**: A gaussian latent sample comprises independently sampled pixels, exhibiting no spatial correlations. Conversely, a requirement that each pixel value differs from its neighbors by a $\epsilon \sim \mathcal{N}(0, 1)$ results in 2D brownian noise ($\alpha=2$).

**Seed Considerations**: While the node defaults to outputting gaussian noise, a given seed produce a different sample than the one produced by other gaussian noise sources. This stems from sampling the noise directly in the frequency domain to avoid the cost of a FFT. When `time_brownian = true`, noise sampling occurs in the spatial domain, ensuring that default parameters yield output equivalent to `SonarCustomNoise` set to `brownian`.

</details>

<br/>

From a usage perspective, using positive alpha will tend to create a colorful effect, using negative alpha will create line/streak like artifacts sort of like an oil painting canvas. Start with small values at first (`-0.1`, `0.1`) and adjust as necessary. `time_brownian` makes the effect of power noise (and alpha) stronger - also note that it can only be used when sampling and not for `NoisyLatentLike`. Setting `common_mode` also generally seems to intensify these effects. Different types of models (normal EPS models, v-prediction models, SDXL) generally react differently to these exotic noise types so my advice is to experiment! Lowering `mix` uses normal gaussian noise for part of the generated noise. For example, `mix=1.0` means 100% power noise, `mix=0.5` means 50/50 power noise and normal gaussian noise. This also is about the same as setting factor to `0.5` and plugging in a `SonarCustomNoise` node with factor at `0.5` also and the type set to `guassian`.

Noise from the `SonarCustomNoise` node and `SonarPowerNoise` can be freely mixed.

## `SonarPowerFilterNoise`

This node lets you connect a filter (see below) and a custom noise chain. It basically lets you run any type of noise through the power noise filter.

New parameters:

* `filter_norm_factor` controls how much normalization is applied to the filter. `1.0` means fully normalized, `0.0` means no normalization.
* You may set the preview type to `custom` to see a color preview of the filtered noise. Note that this uses whatever preview type you have configured in ComfyUI (for example, TAESD). The preview is based on SD 1.5's interpretation of the noise.

## `SonarPowerFilter`

Most of the parameters here are similar to the `SonarPowerNoise` node. New parameters:

* `scale` allows you to scale the filter (you could consider this to be set to `1.0` in the `SonarPowerNoise` node).
* `compose_mode` allows you to compose multiple filters. Note that composition occurs like `current_filter OPERATION connected_filter`. So if you set `compose_mode` to `sub`, you will get `current_filter - connected_filter`. Scaling occurs before composition.

## `SonarPreviewFilter`

Allows you to preview a filter. It does not modify the input filter.

***

## Examples

The example images are all workflow-included. Generated using `dpmpp_2s_ancestral`, Karras scheduler and
starting out with gaussian noise then switching to power noise at the 35% mark. `filter_norm_factor` is set to
1.0 in these examples.

### Node Defaults

This should be the same as normal gaussian noise.

![PowernoiseDefault](../assets/example_images/noise_base_types/noise_powernoise_default.png)

### Positive Alpha

Positive alpha generally produces a colorful effect. Start with relatively low values and increase
until you achieve the desired result. Note that these examples use _relatively_ extreme settings.

With alpha 0.25:

![PowernoiseAlpha_0.25](../assets/example_images/noise_base_types/noise_powernoise_alpha_0.25.png)

With alpha 0.25, common mode 0.25:

![PowernoiseAlpha_0.25_common_0.25](../assets/example_images/noise_base_types/noise_powernoise_alpha_0.25_common_0.25.png)

With alpha 0.35:

![PowernoiseAlpha_0.35](../assets/example_images/noise_base_types/noise_powernoise_alpha_0.35.png)

With alpha 0.35, common mode 0.35:

![PowernoiseAlpha_0.35_common_0.35](../assets/example_images/noise_base_types/noise_powernoise_alpha_0.35_common_0.35.png)

With alpha 0.5:

![PowernoiseAlpha_0.5](../assets/example_images/noise_base_types/noise_powernoise_alpha_0.5.png)

With alpha 0.5, common mode 0.5:

![PowernoiseAlpha_0.5_common_0.5](../assets/example_images/noise_base_types/noise_powernoise_alpha_0.5_common_0.5.png)

### Negative Alpha

With alpha -0.5:

![PowernoiseAlpha-0.5](../assets/example_images/noise_base_types/noise_powernoise_alpha_-0.5.png)

With alpha -1.5:

![PowernoiseAlpha-1.5](../assets/example_images/noise_base_types/noise_powernoise_alpha_-1.5.png)

### Time Brownian Mode

![PowernoiseTb](../assets/example_images/noise_base_types/noise_powernoise_tb.png)

With alpha 0.5:

![PowernoiseTbAlpha_0.5](../assets/example_images/noise_base_types/noise_powernoise_tb_alpha_0.5.png)

With alpha -0.5:

![PowernoiseTbAlpha-0.5](../assets/example_images/noise_base_types/noise_powernoise_tb_alpha_-0.5.png)


