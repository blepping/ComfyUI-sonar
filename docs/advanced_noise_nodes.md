# Advanced Nodes

## Normalization

Normalization essentially rebalances the noise (or mixture of noise) to 1.0 strength and then scales based
on the factor of the node. Most nodes will allow you to set three values:

* `default`: By default, noise will be normalized only just before it's used. So you could consider this setting to be false except for where it is connected to an actual noise consumer (i.e. a `SamplerConfigOverride` node).
* `forced`: Will always normalize.
* `disabled`: Will never normalize.

## `SONAR_CUSTOM_NOISE`

This node output type actually constitutes a chain of noise items. For most nodes, when you use it as input,
they will add an item to the chain. There are some exceptions that treat the `SONAR_CUSTOM_NOISE` input as a list:

* `SonarRepeatedNoise`
* `SonarRandomNoise`

There are also some exceptions that will consume the list rather than adding an item to it:

* `SonarModulatedNoise`
* `SonarCompositeNoise`
* `SonarScheduledNoise`
* `SonarGuidedNoise`

The distinction is mainly only important when setting `rescale`. Visual example:

![Chain example](../assets/example_images/noise_adv/noise_chain_example.png)

It may be counter intuitive that there are actually two separate chains here.

## Examples

Note on the examples included for some of these nodes:

The example images included for some of these nodes all have metadata and can be loaded in ComfyUI.
Generated using `dpmpp_2s_ancestral`, Karras scheduler and starting out with gaussian noise then switching
to the custom noise type at the 35% mark.

***

### `SonarCustomNoise`

You can chain `SonarCustomNoise` nodes together to mix different types of noise. The order of `SonarCustomNoise` nodes is not important.

Parameters:

- `factor` controls the strength of the noise.
- `rescale` controls rebalancing `factor` for nodes in the chain. When `rescale` is set to `0.0`, no rebalancing will occur. Otherwise the current node as well as the nodes connect to it will have their `factor` adjusted to add up to the rescale value. For example, if you have three nodes with `factor` 1.0 and the last with `rescale` 1.0, then the `factor` value will be adjusted to `1/3 = 0.3333...`. *Note*: Rescaling uses the `factor` absolute value.
- `noise_type` allows you to select the built-in noise type.

***

### `NoisyLatentLike`

This node takes a reference latent and generates noise of the same shape. The one required input is `latent`.

You can connect a `SonarCustomNoise` or `SonerPowerNoise` node to the `custom_noise_opt` input: if that is attached, the built in noise type selector is ignored. The generated noise will be multiplied by the `multiplier` value. **Note**: If you select `brownian` noise (either through the dropdown or by connecting custom noise nodes) you must connect a model and sigmas.

The node has two main modes: simply generate and scale the noise by the multiplier and return or add it to the input latent. In this mode, you don't connect anything to the `mul_by_sigmas_opt` or `model_opt` inputs and you would use other nodes to calculate the correct strength.

In the second mode you must connect sigmas (for example from a `BasicScheduler` node) to the `mul_by_sigmas_opt` input and connect a model to the `model_opt` input. It will calculate the strength based on the first item in the list of sigmas (so you could use something like a `SplitSigmas` node to slice them as needed). Note that `multiplier` still applies: the calculated strength will be scaled by it. This second mode is generally this is the most convenient way to use the node since the two main uses cases are: making a latent with initial noise or adding noise to a latent (for img2img type stuff).

If you want to create noise for initial sampling, connect model and sigmas to the node, connect an empty latent (or one of the appropriate size) to it and that is basically all you need to do (aside from configuring the noise types). For img2img (upscaling, etc), either slice the sigmas at the appropriate or set a denoise in something like the `BasicScheduler` node. *Note*: For img2img, you also need to turn on the `add_to_latent` toggle. Turning this on doesn't matter for initial noise since an empty latent is all zeros.

**Note**: This node does not currently respect the latent noise mask.

***

### `SamplerConfigOverride`

This node can be used to override configuration settings for other samplers, including the noise type. For example, you could force `euler_ancestral` to use a different noise type. It's also possible to override other settings like `s_noise`, etc. *Note*: The wrapper inspects the sampling function's arguments to see what it supports, so you should connect the sampler directly to this rather than having other nodes (like a different sampler wrapper) in between.

***

### `SONAR_CUSTOM_NOISE to NOISE`

This node can be used to convert Sonar custom noise to the `NOISE` type used by the builtin `SamplerCustomAdvanced` (and any other nodes that take a `NOISE` input).

***

### `SonarAdvancedPyramidNoise`

Allows setting some parameters for the pyramid noise variants (`pyramid`, `highres_pyramid` and `pyramid_old`). `discount` further from zero generally results in a more extreme colorful effect (can also be set to negative values). Higher `iterations` also tends to make the effect more extreme - zero iterations will just return normal Gaussian noise. You can also experiment with the `upscale_mode` for different effects.

***

### `SonarModulatedNoise`

Experimental noise modulation based on code stolen from
[ComfyUI-Extra-Samplers](https://github.com/Clybius/ComfyUI-Extra-Samplers). `intensity` and `frequency` modulation
types _probably_ do not work correctly for normal sampling — I expect the modulation will be based on the tensor
where the noise sampler was created rather than each step. However it may be useful for something like restart sampling
noise (see `KRestartSamplerCustomNoise` below). You can also pass it a reference latent to modulate based on
instead (only used for `intensity` and `frequency` modulation types).

*Note*: It's likely this node will be changed in the future.

<details>

<summary>⭐ Expand Example Images ⭐</summary>

<br/>

These examples all use the `spectral_signum` modulation type as it doesn't depend on a reference.

#### Positive Strength

Dims 3:

![Dims 3](../assets/example_images/noise_adv/noise_modulated_ss_dims3.png)

Dims 3 (with studentt noise):

![Dims 3](../assets/example_images/noise_adv/noise_modulated_ss_dims3_studentt.png)

Dims 2:

![Dims 2](../assets/example_images/noise_adv/noise_modulated_ss_dims2.png)

Dims 1:

![Dims 1](../assets/example_images/noise_adv/noise_modulated_ss_dims1.png)

#### Negative Strength

Dims 3:

![Dims 3 Negative](../assets/example_images/noise_adv/noise_modulated_ss_neg_dims3.png)

Dims 3 (with studentt noise):

![Dims 3](../assets/example_images/noise_adv/noise_modulated_ss_neg_dims3_studentt.png)

Dims 2:

![Dims 2 Negative](../assets/example_images/noise_adv/noise_modulated_ss_neg_dims2.png)

Dims 1:

![Dims 1 Negative](../assets/example_images/noise_adv/noise_modulated_ss_neg_dims1.png)

</details>

***

### `SonarRepeatedNoise`

Experimental node to cache noise sampler results. Why would you want to do this? Some noise samplers are
relatively slow (`pyramid` for example) or it may be slow to generate noise if you are mixing many types
of noise. When `permute` is enabled, a random effect like flipping the noise or rolling it in some dimension
will be chosen each time the noise sampler is called. I recommend leaving `permute` on. Note that repeated
noise (especially with `permute` disabled) can be stronger than normal noise, so you may need to rescale to
a value lower than `1.0` or decrease `s_noise` for the sampler. You may also set the maximum number of
times noise is reused by setting `max_recycle`.

<details>

<summary>⭐ Expand Example Images ⭐</summary>

<br/>

Repeated noise is very strong (especially when permute is disabled). You generally won't get good
results using 1.0 strength:

![Normal](../assets/example_images/noise_adv/noise_repeated_normal.png)

I recommend considerably decreasing the strength (example here is using 0.75 which is still a bit too much):

![Adjusted](../assets/example_images/noise_adv/noise_repeated_adjusted.png)

</details>

***

### `SonarCompositeNoise`

Allows compositing noise types based on a mask. Noise is mixed based on the strength of the mask at a location.
For example, where the mask is 1.0 (max strength) you will get 100% `noise_src` and 0% `noise_dst`. Where the
mask is 0.75 you will get 75% `noise_src` and 25% `noise_dst`.

<details>

<summary>⭐ Expand Example Images ⭐</summary>

<br/>

These examples use a base noise type of gaussian and composite in an area with a different type
near middle. The custom noise is also set to a higher strength than normal to highlight the effect.

**No Composite (for comparison)**

![No Composite](../assets/example_images/noise_base_types/noise_gaussian.png)

**Brownian**

![Brownian](../assets/example_images/noise_adv/noise_composite_brownian.png)

**Pyramid**

![Pyramid](../assets/example_images/noise_adv/noise_composite_pyramid.png)

**Pyramid negative factor**

![Pyramid negative](../assets/example_images/noise_adv/noise_composite_pyramid_neg.png)

</details>

***

### `SonarScheduledNoise`

Allows switching between noise types based on percentage of sampling (note: not percentage of steps).

**Note**: You don't have to connect the fallback noise type but the default is to generate _no_ noise, which
is most likely not what you want. The majority of the time, it is recommend to connect something like gaussian
noise at 1.0 strength.

All the example images here use the `SonarScheduledNoise` node so you can pick any one of them to see it
in action!

***

### `SonarGuidedNoise`

Works similarly as described in the [Guidance](../README.md#guidance) section of the main README, however the guidance is applied
to the raw noise. You can use `SonarScheduledNoise` to only apply guidance at certain times. Using `euler`
mode seems considerably stronger than `linear`. The default value should be reasonable for `euler`, may need to be
increased somewhat for `linear`.

<details>

<summary>⭐ Expand Example Images ⭐</summary>

<br/>

#### Pattern

These examples use a half circle pattern as the reference: ![pattern](../assets/example_images/noise_adv/noise_guided_ref_pattern.png)


##### Euler

Positive strength:

![Positive](../assets/example_images/noise_adv/noise_guided_pattern_euler.png)

Negative strength:

![Negative](../assets/example_images/noise_adv/noise_guided_pattern_euler_neg.png)

***

##### Linear

Normal positive strength:

![Positive](../assets/example_images/noise_adv/noise_guided_pattern_linear.png)

Normal negative strength:

![Negative](../assets/example_images/noise_adv/noise_guided_pattern_linear_neg.png)

Strong positive strength:

![Strong Positive](../assets/example_images/noise_adv/noise_guided_pattern_linear_strong.png)

Strong negative strength:

![Strong Negative](../assets/example_images/noise_adv/noise_guided_pattern_linear_strong_neg.png)


***

#### Gradient

These examples use a vertical gradient as the reference: ![pattern](../assets/example_images/noise_adv/noise_guided_ref_gradient.png)

That is dark to light. Light to dark examples just flip the gradient vertically.

##### Euler

Dark to light:

![Dark to light](../assets/example_images/noise_adv/noise_guided_dtol_euler.png)

Light to dark:

![Light to dark](../assets/example_images/noise_adv/noise_guided_ltod_euler.png)

Dark to light (negative strength):

![Dark to light](../assets/example_images/noise_adv/noise_guided_dtol_euler_neg.png)

Light to dark (negative strength):

![Light to dark](../assets/example_images/noise_adv/noise_guided_ltod_euler_neg.png)

***

##### Linear

Dark to light:

![Dark to light](../assets/example_images/noise_adv/noise_guided_dtol_linear.png)

Light to dark:

![Light to dark](../assets/example_images/noise_adv/noise_guided_ltod_linear.png)

Dark to light (negative strength):

![Dark to light](../assets/example_images/noise_adv/noise_guided_dtol_linear_neg.png)

Light to dark (negative strength):

![Light to dark](../assets/example_images/noise_adv/noise_guided_ltod_linear_neg.png)

</details>

***

### `SonarRandomNoise`

Randomly chooses between the noise types in the chain connected to it each time the noise sampler is called.
You generally do not want to use `rescale` here. You can also set `mix_count` to choose and combine multiple
types.
