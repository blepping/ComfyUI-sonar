# Changes

Note, only relatively significant changes to user-visible functionality will be included here. Most recent changes at the top.

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
