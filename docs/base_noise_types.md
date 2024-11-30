# Base Noise Examples

The example images are all workflow-included. Generated using `dpmpp_2s_ancestral`, Karras scheduler and
starting out with gaussian noise then switching to the custom noise type at the 35% mark.

Some of these noise types are too extreme to be used for initial generations or even with pure
noise of that type. However you can either schedule the noise type to kick in at a certain percentage
(as in these examples) and/or mix it with something a bit more run of the mill. See
[advanced_noise_nodes](advanced_noise_nodes.md).

## Documentation TBD

* `grey`
* `onef_greenish_mix` (50/50 mix of positive/negative noise.)
* `onef_greenish`
* `onef_pinkish_mix` (50/50 mix of positive/negative noise.)
* `onef_pinkish`
* `onef_pinkishgreenish` (50/50 mix of `onef_pinkish` and `onef_greenish`.)
* `velvet`
* `violet`
* `white`

## Brownian

This is the default noise type for SDE samplers.

![Brownian](../assets/example_images/noise_base_types/noise_brownian.png)

***
## Gaussian

This is the default noise type for non-SDE samplers.

![Gaussian](../assets/example_images/noise_base_types/noise_gaussian.png)

***

## Green Test

This is _probably_ not actually green noise. It produces a very colorful effect, however
it's very strong and not really suitable for initial generation.

![GreenTest](../assets/example_images/noise_base_types/noise_green_test.png)

You can also use a negative multiplier to achieve a different effect:

![GreenTestNeg](../assets/example_images/noise_base_types/noise_green_test_neg.png)

***

## Highres Pyramid

![HighresPyramid](../assets/example_images/noise_base_types/noise_highres_pyramid.png)

Variation using area scaling:

![HighresPyramidArea](../assets/example_images/noise_base_types/noise_highres_pyramid_area.png)

Variation using bislerp scaling:

![HighresPyramidBislerp](../assets/example_images/noise_base_types/noise_highres_pyramid_bislerp.png)

***

## Laplacian

![Laplacian](../assets/example_images/noise_base_types/noise_laplacian.png)

***

## Perlin

![Perlin](../assets/example_images/noise_base_types/noise_perlin.png)

***

## Pink Old

Previously known as `pink`. The implementation isn't correct, though in terms of results it's fine.

![Pink](../assets/example_images/noise_base_types/noise_pink.png)

***

## Power Old

Previously known as `power`. The implementation isn't correct, though in terms of results it's fine.

![PowerBuiltin](../assets/example_images/noise_base_types/noise_power_builtin.png)

Also see the [Advanced Power Noise](advanced_power_noise.md) examples.

***

## Pyramid

![Pyramid](../assets/example_images/noise_base_types/noise_pyramid.png)

You can also use a negative multiplier to achieve a different effect:

![PyramidNeg](../assets/example_images/noise_base_types/noise_pyramid_neg.png)

Variation using area scaling:

![PyramidArea](../assets/example_images/noise_base_types/noise_pyramid_area.png)

Variation using bislerp scaling:

![PyramidBislerp](../assets/example_images/noise_base_types/noise_pyramid_bislerp.png)

***

## Pyramid Discount5

Pyramid noise, generated with a discount of 0.5. (Generally less extreme effect.)

![PyramidDiscount5](../assets/example_images/noise_base_types/noise_pyramid_discount5.png)

***

## Pyramid Mix

Pyramid mix is a combination of positive and negative pyramid noise. The effect on
the generation is mild compared to raw pyramid noise.

![PyramidMix](../assets/example_images/noise_base_types/noise_pyramid_mix.png)

You can also use a negative multiplier to achieve a different effect:

![PyramidMixNeg](../assets/example_images/noise_base_types/noise_pyramid_mix_neg.png)

Variation using area scaling:

![PyramidMixArea](../assets/example_images/noise_base_types/noise_pyramid_mix_area.png)

You can also use a negative multiplier to achieve a different effect:

![PyramidMixAreaNeg](../assets/example_images/noise_base_types/noise_pyramid_mix_area_neg.png)

Variation using bislerp scaling:

![PyramidMixBislerp](../assets/example_images/noise_base_types/noise_pyramid_mix_bislerp.png)

You can also use a negative multiplier to achieve a different effect:

![PyramidMixBislerpNeg](../assets/example_images/noise_base_types/noise_pyramid_mix_bislerp_neg.png)

***

## Pyramid Old

This may not actually be pyramid noise at all. Also note that it is quite slow to generate as it
effectively generates noise ~60x the latent size.

![PyramidOld](../assets/example_images/noise_base_types/noise_pyramid_old.png)

Variation using area scaling:

![PyramidOldArea](../assets/example_images/noise_base_types/noise_pyramid_old_area.png)

Variation using bislerp scaling:

![PyramidOldBislerp](../assets/example_images/noise_base_types/noise_pyramid_old_bislerp.png)

***

## Rainbow

Rainbow is a mix of Perlin and Green noise types.

The "mild" variation uses a relatively low proportion of green noise:

![RainbowMild](../assets/example_images/noise_base_types/noise_rainbow_mild.png)

The "intense" variation uses a higher proportion of green noise for a more extreme effect.

![RainbowIntense](../assets/example_images/noise_base_types/noise_rainbow_intense.png)

***

## Studentt

![Studentt](../assets/example_images/noise_base_types/noise_studentt.png)

***

## Uniform

![Uniform](../assets/example_images/noise_base_types/noise_uniform.png)
