# Wavelet CFG

A CFG function that lets you use different CFG values for different frequencies.

Node: `SonarWaveletCFG`

## Requirements

You will need to have the `pytorch_wavelets` package installed in your Python environment to use this.

Link: https://github.com/fbcotter/pytorch_wavelets

## YAML crash course

You can skip past this if you already know YAML. Since wavelet CFG definitions are defined with YAML rules,
I am putting this section near the beginning.

First, JSON is valid YAML, so if you know JSON you can use that if you prefer. Since JSON is valid YAML, this also means the structure of YAML documents is the same as a JSON document.

YAML looks like this:

```yaml
# Comments start with the hash symbol.
# YAML will guess the type if you don't do stuff like quote strings, so the item below
# will be "value".
key1_name: value
# The value of a key can also be a set of keys (usually called an object)
# YAML uses indentation to control block grouping.
key2_name:
    # A comment
    subkey1_name: 123
    # A list of three items, two integers and a string.
    some_list: [1, 2, "hello"]
    # You can also specify lists like this:
    some_other_list:
        - 1
        - 2
        - hello
    # It's legal to specify the same key multiple times. This just overwrites
    # whatever the previous value was.
    subkey1_name: 345
```

YAML item types:

* String: `"hello there"` or `hello there` (you may need to quote when there are special characters).
* Integer: `123`
* Floating point value: `1.23`
* Object, may be specified in-line like JSON: `{ key: value, key: value }`. Be careful to separate the value from the colon after the key or it may be interpreted incorrectly. It may be a good idea to quote string values if you're using this syntax (and it's necessary if they have special characters or spaces).
* List, may be specified in-line like JSON: `[1, 2, "hello there"]`
* Null: `null`. If you want the string "null" then you'd need to quote it.
* Boolean: `true` and `false`.

#### Advanced YAML

YAML also has a number of advanced features like references. You can use this to avoid repeating the same information multiple times. For example:

```yaml
single_value: &single_ref_name [1, 2, 3]
# This is the same as other_value: [1, 2, 3]
other_value: *single_ref_name

# You can also do this with objects.
reference_block: &ref_name
    key: value
    other_key: other_value
whatever:
    # This sets the "whatever" object to be the same as "reference_block"
    # Note that you can't just do "whatever: *ref_name" to get that effect here.
    <<: *ref_name
    # And you can just overwrite the keys you want to change:
    key: 123
    # At this point, "whatever" is { key: 123, other_key: other_value }
```

## Usage

Unfortunately, this isn't very user-friendly and needs to be configured with YAML. This is a relatively basic description of
usage. To see all possible options, look at the default configuration definition in the node.

General information on wavelets from the library I'm using to do wavelet transforms: https://pytorch-wavelets.readthedocs.io/en/latest/index.html

Trimmed down, the default config looks like this:

```yaml
# This block is used to set the CFG scales.
diff:
    # Scale for the low-frequency band.
    yl_scale: 5.0

    # Scale for the high-frequency bands.
    yh_scales: 3.0

# Sets the wavelet type. DB4 is a good general-purpose wavelet to use.
wave: db4

# Sets the wavelet level.
level: 5

# Set to true if you want to get detailed information dumped to your console.
verbose: false
```

Wavelets decompose the value into a low frequency value and a set of high-frequency bands. The number of high-frequency
will be equal to the wavelet level, so in this example you will have one low-frequency band and five high-frequency
bands to work with. The high-frequency bands are further decomposed into three parts which can also be targeted
individually: horizontal, vertical, diagonal.

The `diff` (or `difference`, whichever you prefer) block is where you set the CFG scales. It's called `difference`
because CFG is defined as `uncond + (cond - uncond) * cfg_scale` (`cond` is the positive prompt, `uncond` is
negative). So CFG is just the difference between `cond` and `uncond`, multiplied by the CFG scale.

The example configuration here is using CFG 5 for the low frequency band and CFG 3 for the high frequency bands. It is
possible to get even more specific that that. Since we're using `level: 5` here, that means there are five frequency
bands that can all be set individually. The high-frequency bands are ordered from fine to coarse detail levels. Example:

```yaml
diff:
    yl_scale: 5.0
    # Can also be written: yh_scales: [5.0, 3.0, fill]
    yh_scales:
        # Highest/finest band.
        - 5.0
        # Decreasing order of detail/frequency.
        - 3.0
        - fill
```

The special value `fill` will just repeat the value before it to fill the rest of the bands. Note that if
you don't specify the bands or fill then the bands you don't set will use `1.0`. For example with five
bands, `[5.0, 3.0, fill]` is the same as `[5.0, 3.0, 3.0, 3.0, 3.0]` while `[5.0, 3.0]` is the same
as `[5.0, 3.0, 1.0, 1.0, 1.0]`. You can only use one `fill` per `yh_scales` definition.

As mentioned, it's also possible to target horizontal, vertical and diagonal bands. You can do this
by using a list instead of numeric value for a band definition. **Note**: You need to specify all
three bands, `fill` isn't valid here. Example:

```yaml
diff:
    yl_scale: 5.0
    # Can also be written: yh_scales: [5.0, 3.0, fill]
    yh_scales:
        - [3.0, 3.0, 5.0]
        - 3.0
        - fill
```

This example uses CFG 3.0 for horizontal and vertical in finest high-frequency band and CFG 5.0 for
diagonal. The remaining high-frequency bands use CFG 3.0.

## Scheduling CFG

It's also possible to transition from one set of CFG scales to another over time. The wavelet scales
block has an alternative definition format:

```yaml
diff:
    # One of linear, logarithmic, exponential, half_cosine, sine
    # Sine mode will hit the peak scales_after values in the middle of the range.
    schedule: linear

    # One of: sampling, enabled_sampling, sigmas, enabled_sigmas, step, enabled_steps
    schedule_mode: enabled_sampling

    # When enabled, flips the schedule percentage. This happens before the schedule is applied
    # or any offset/multiplier stuff. If you want to flip the final result you can do something like
    # schedule_offset_after: -1.0 and schedule_multiplier_after: -1.0
    reverse_schedule: false

    scales_start:
        yl_scale: 5.0
        yh_scales: 3.0
    scales_end:
        yl_scale: 2.0
        yh_scales: 5.0
```

The way interpolating scales works is we determine a value between 0.0 and 1.0 based on `schedule` and
`schedule_mode` and then do linear interpolation (LERP) between the values in `scales_start` and `scales_end`.
LERP is just `value_1 * (1.0 - ratio) + value_2 * ratio` so when `ratio` is 1 you get 100% `value_2`,
when it's 0.5 you get half of each and when it's 0 you get 100% of `value_1`.

**Note**: If you have both `scales_start` and toplevel `yl_scale`/`yh_scales` definitions, the
scales in `scales_start` will take precedence.

#### `schedule`

Current schedule types: `linear`, `logarithmic`, `exponential`, `half_cosine`, `sine`

Linear just changes by the same amount over time, with the exception
of `sine`, the other possible values are similar except the change forms a curve (where it may be slow at first and then
accelerate or vice versa). Experiment with them to see what you prefer. `sine` has a somewhat different effect, the
percentage of `scales_end` will increase and peak in the middle of the range, then decrease.

#### `schedule_mode`

Current modes: `sampling`, `enabled_sampling`, `sigmas`, `enabled_sigmas`, `step`, `enabled_steps`

* `sampling`: Sampling is a percentage that starts at 0.0 and ends at 1.0 (assuming you're doing txt2img). This isn't really related
to the schedule or steps.
* `sigmas`: This calculates the difference between the starting sigma and ending sigma as a percentage.
* `step`: Not very well tested and may not work (especially with multi-step samplers). The percentage in this case is the percentage
    steps

The `_enabled` variants calculate the percentage based on the range that wavelet CFG is enabled for (in other words,
the range betwmeen `start_sigma` and `end_sigma`). I'd suggest not using them as it's a lot easier to predict what values
will be used when the schedule start/end points aren't also changing.

***

In addition to the values described here, there are number of other advanced configuration options that can be used
to add/subtract and offset to the calculated percentage value, multiply it, etc. These advanced parameters can be
used to do stuff like speed up the transition between config values, keep them within a certain range with minimum/maximum
thresholds, etc. See the default YAML config definition in the node to see what is possible.


## Scheduling rules

Rules may be scheduled using this syntax:

```yaml
rules:
    - start_sigma: -1.0
      end_sigma: 5.0
      diff:
        yl_scale: 5.0
        yh_scales: 3.0
    - start_sigma: -1.0
      end_sigma: 0.0
      diff:
        yl_scale: 2.0
        yh_scales: 5.0
```

Values from the top-level are valid within a rule. The definitions from the node are added as the first rule,
so if you want to only configure stuff in a `rules` block you can just set the start sigma in the node to `0.0` (
which will never match). Rules are checked in order and the first matching one is used.

An alternative method of scheduling rules is to just chain multiple `SonarWaveletCFG` nodes. If you set the fallback
mode to `existing` it is also possible to blend the current result with the next matching one (or normal CFG as the
case may be).
