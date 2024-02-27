# Changes

Note, only relatively significant changes to user-visible functionality will be included here. Most recent changes at the top.

## 20240227

* Refactored noise generation functions (will break seeds).
* Added `SamplerOverride` node.
* `studentt` noise type replaced with `studentt_test` (the more correct version).

## 20240210

* Added `SonarCustomNoise` node.
* Changed existing sampler nodes and `NoisyLatentLike` nodes to take an optional `SonarCustomNoise`.
