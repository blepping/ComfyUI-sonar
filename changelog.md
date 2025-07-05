# Changes

Note, only relatively significant changes to user-visible functionality will be included here. Most recent changes at the top.

## 20250705

This is a large set of changes. Please let me know anything doesn't seem to be working properly.

* Reorganized the node structure. This is an internal change and shouldn't affect users but please let me know if you notice anything weird.
* Added a `SonarLatentOperationAdvanced` node which allows more control over when individual latent operations are active and their effects get blended.
* Added a `SonarSplitNoiseChain` node. Can be useful if you want to have an item in the chain be a blended.
* Added a `SonarLatentOperationNoise` node that can be used to inject noise. You can also use the guided noise node to turn a reference into "noise".
* Expanded the functionality of the `SonarWaveletFilteredNoise` node. You can now attach two custom noise inputs to use for the high/low frequency parts of the wavelet as well as blend the wavelets.
* Added a `SonarNormalizeNoiseToScale` node that lets you normalize noise to specific value ranges.
* Added a `SonarPerDimNoise` node that lets you do stuff like call a noise sampler once per batch item (can be useful for 3D Perlin noise).
* Fixed an issue where the normalization parameter wasn't respected. This may change seeds.
* Added a `SonarLatentOperationFilteredNoise` node that allows you to run noise through a `LATENT_OPERATION`.
* Added a `SonarLatentOperationSetSeed` node that can be used to set the seed (mainly useful for running latent operations that add noise outside of sampling).
* Added a `SonarScatternetFilteredNoise` node that uses a scatternet to filter noise. Similar to wavelet filtering. Note: Very experimental, way not work properly.
* Fixed an issue with pattern break noise, this may change seeds for workflows using that noise type.

## 20250627

* Added `SonarRippleFilteredNoise` node.
* Added `SonarApplyLatentOperationCFG` node, similar to the built-in `ApplyLatentOperationCFG` node with scheduling and a lot of different application modes.
* Added a `SonarLatentOperationQuantileFilter` node that can be used to apply the quantile normalization functioen to the latent during sampling.
* A bunch more quantile normalization modes.
* Fixed broken quantile normalization dimension handling. Unfortunately this will likely change seeds.

## 20250612

* Reimplemented Collatz noise with many new features. Unfortunately this breaks existing workflows. If anyone misses the old version, let me know and I can add it back in (might do that anyway).
* Added actual wavelet noise based on https://en.wikipedia.org/wiki/Wavelet_noise .
* Added `reverse_zero`, `scale_down`, `tanh`, `tanh_outliers`, `sigmoid` and `sigmoid_outliers` quantile normalization limit modes.

## 20250602

* Fixed broken calculation for Collatz noise.
* Added `SonarPatternBreakNoise` node that allows breaking patterns in the noise.
* Added `SonarShuffledNoise` node that allows shuffling elements along user-specified dimensions.
* Added a strategy option to the `SonarQuantileFilteredNoise` node.
* Added variants to Collatz noise. Variant one is maybe similar to the original iteration.
* Added `SonarNoiseImage` node that allows generating noisy images or adding noise to existing images.

## 20250528

* Added `override_sigma`, `override_sigma_next`, `override_sigma_min` and `override_sigma_max` options that can be set in the `SonarCustomNoiseAdv` node YAML options. This enables using noise generators that require a sigma in stuff like initial noise (for example, Brownian). You will need to manually find and set the correct values yourself.
* Added Collatz noise based on the Collatz conjecture. Very experimental, very slow, likely to change and quite possibly just plain bad. But you can try it.

## 20250505

* Added `SonarQuantileFilteredNoise` node.
* Better compatibility with older Python versions.

## 20250227

* Add 5D latent (video models) support for most custom noise types.

## 20250130

*Note*: May change seeds.

This set of changes includes some pretty major internal refactoring. Definitely possible that I broke something, so please create an issue if you run into problems.

* Noise generation should now respect whether generating on CPU vs GPU is selected. Previously it likely was defaulting to generating on GPU. This may change seeds.
* Refactored momentum samplers, this may change seeds especially if you were using weird parameters like negative direction.
* Added some new parameters for momentum samplers.
* Removed the `s_noise` and churn parameters from the normal Sonar Euler sampler. May break workflows. (Churn was the predecessor to ancestral samplers and is basically obsolete.)
* Added `wavelet` and `distro` noise types.
* Added `SonarCustomNoiseAdv` node that allows passing parameters via YAML.
* Added `SonarResizedNoise` node that allows you to generate noise at a fixed size and then crop/resize it to match the generation.
* Added `SonarAdvancedDistroNoise` node that allows generating noise with basically all the distributions PyTorch supports.
* Added `SonarWaveletFilteredNoise` node that lets you filter another noise generator using wavelets.

## 20241129

*Note*: Contains some potentially workflow-breaking changes.

* `pink` noise type renamed to `pink_old` - the implementation was incorrect.
* `power` noise type renamed to `power_old` - the implementation was incorrect.
* Added `onef_pinkish` (higher frequencye) and `onef_greenish` (lower frequency) noise types.
* Added `SonarAdvanced1fNoise` node and `onef_pinkish`, `onef_greenish`, `onef_pinkish_mix`, `onef_greenish_mix`, and `onef_pinkishgreenish` noise types.
* Added `SonarAdvancedPowerLawNoise` node and `grey`, `white`, `violet` and `velvet` noise types.
* The `SonarAdvancedPyramidNoise` node can now use upscale methods from my [ComfyUI-bleh](https://github.com/blepping/ComfyUI-bleh) node pack if it is available.
* Added the `SonarChannelNoise` and `SonarBlendedNoise` nodes.
* Added the `SonarBlehOpsNoise` node.
* Added advanced parameter input to the SampleConfigOverride node, you can now pass options directly to the wrapped sampler function.
* Custom noise inputs now are semi-wildcard and will accept `OCS_NOISE` or `SONAR_CUSTOM_NOISE` interchangeably.

## 20240823

* Added descriptions and tooltips for most nodes.
* Added `repeat_batch` parameter to `NoisyLatentLike` node.
* Added a `SONAR_CUSTOM_NOISE to NOISE` node to allow converting from Sonar's custom noise type to the built in ComfyUI `NOISE` (used by `SamplerCustomAdvanced` and possibly other nodes).
* Added a `SonarAdvancedPyramidNoise` node that allows setting parameters for the pyramid noise variants.

## 20240521

Mega update! Many new features, documentation reorganized.

* Add `SonarScheduledNoise`, `SonarCompositeNoise`, `SonarGuidedNoise`, `SonarRandomNoise` nodes. See [Advanced Noise Nodes](docs/advanced_noise_nodes.md).
* Add `SonarPowerFilterNoise`, `SonarPowerFilter`, `SonarPreviewFilter` nodes. See [Advanced Power Noise](docs/advanced_power_noise.md).
* Add `FreeUExtreme`, `FreeUExtremeConfig` nodes. See [FreeU Extreme](docs/frux.md).
* Replace `pyramid` noise type with a (hopefully) more correct implementation. You can use `pyramid_old` for the previous behavior.
* Add more noise types and variations.
* The `NoisyLatentLike` node now allows using brownian noise if you connect a model and sigmas.

## 20240506

* Add `SonarModulatedNoise` and `SonarRepeatedNoise` nodes.

## 20240327

* Fixed issue when using Sonar samplers in normal sampling nodes/via stuff like `KSamplerSelect`.
* Add `pyramid` (non-high-res) noise type.
* Allow selecting `brownian` noise in custom noise nodes (but it won't work with `NoisyLatentLike`).
* Use `brownian` as the default noise type for `SamplerSonarDPMPP`.
* Make overriding the selected noise type in Sonar samplers a warning instead of a hard error.
* Improve noise scaling (may change seeds).
* Add `KRestartSamplerCustomNoise` if the user has a recent enough version of ComfyUI_restart_sampling installed.

## 20240320

* `NoisyLatentLike` node improved to allow calculating strength with sigmas and injecting noise itself.

## 20240314

* `SonarPowerNoise` node added.

## 20240227

* Refactored noise generation functions (will break seeds).
* Added `SamplerOverride` node.
* `studentt` noise type replaced with `studentt_test` (the more correct version).

## 20240210

* Added `SonarCustomNoise` node.
* Changed existing sampler nodes and `NoisyLatentLike` nodes to take an optional `SonarCustomNoise`.
