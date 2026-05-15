# CUDA Sort

`algo::cuda::sort` provides in-place CUDA radix sort APIs for unsigned integer
keys and fixed-width multi-word keys. Key-only sorting and sorting value arrays
by key are supported.

```cpp
#include <algo/cuda/sort.cuh>
```

## API

```cpp
algo::cuda::sort::sort_keys<Config, Key>(
    d_keys, count, d_workspace, workspace_size, stream);

algo::cuda::sort::sort_by_key<Config, Key>(
    d_keys,
    algo::cuda::sort::value_arrays(d_values0, d_values1),
    count,
    d_workspace,
    workspace_size,
    stream);

algo::cuda::sort::required_workspace_size<Config, Key>(count);
algo::cuda::sort::required_sort_by_key_workspace_size<
    Config, Key, Value0, Value1>(count);
algo::cuda::sort::required_sort_by_key_workspace_size<Key>(
    count, algo::cuda::sort::value_arrays(d_values0, d_values1));
```

`sort_keys`, `sort_by_key`, and `required_workspace_size` also have overloads
without `Config`; those use the default configuration. For `sort_by_key`, the
default workspace query takes the value-array payload object so the value types
can be inferred.

## Default Configuration

The default is histogram radix sort:

```cpp
using DefaultConfig =
    algo::cuda::sort::RadixSort<
        algo::cuda::sort::HistogramPass<
            algo::cuda::sort::SharedAtomicHistogram,
            algo::cuda::sort::WarpLevelMultiSplitWarpRank,
            4>,
        512,
        4>;
```

This is equivalent to:

```cpp
using DefaultConfig =
    algo::cuda::sort::RadixSort<
        algo::cuda::sort::HistogramPass<
            algo::cuda::sort::SharedAtomicHistogram,
            algo::cuda::sort::WarpLevelMultiSplitWarpRank,
            4>,
        512,
        4,
        32>;
```

With `BlockSize = 512`, `RadixBits = 4`, `KeyBits = 32`, and
`ItemsPerThread = 4`, it runs 8 radix passes.

## Configuration

```cpp
RadixSort<PassPolicy, BlockSize, RadixBits, KeyBits>
```

| Parameter | Default | Notes |
| --- | --- | --- |
| `PassPolicy` | none | Implemented for `HistogramPass<...>`, `FlagPrefixSumPass<...>`, and `OneSweepPass<...>`. |
| `BlockSize` | `256` | Must be positive; warp-rank based policies require a multiple of 32. |
| `RadixBits` | `4` | Bits processed per pass. Max is `8` for histogram sort and `5` for flag-prefix-sum sort. |
| `KeyBits` | `32` | Number of low-order key bits to sort. |

`RadixBits` must be positive and less than 32, `KeyBits` must be positive,
must not exceed the selected key type's bit width, and
`KeyBits % RadixBits == 0`.
`OneSweepPass` currently supports `RadixBits <= 8`.

Available pass policies:

```cpp
algo::cuda::sort::HistogramPass<
    HistogramPolicy,
    WarpRankPolicy,
    ItemsPerThread>

algo::cuda::sort::FlagPrefixSumPass<ScanPolicy>

algo::cuda::sort::OneSweepPass<
    BlockHistogramPolicy,
    LocalRankPolicy,
    GlobalOffsetsBlockHistogramPolicy,
    ItemsPerThread>
```

The default sort uses
`HistogramPass<SharedAtomicHistogram, WarpLevelMultiSplitWarpRank, 4>` with a
block size of `512`.
`FlagPrefixSumPass<>` uses `BucketWiseScan`; it can also be configured with
`FlattenedScan`. `OneSweepPass` has no default template parameters; specify
`WarpLevelMultiSplitWarpRank` and `SharedAtomicGlobalOffsetsBlockHistogram`
explicitly. Its global histogram accumulation is fixed to `atomicAdd`.

Implemented policies:

| Role | Policies | Notes |
| --- | --- | --- |
| Pass policy | `HistogramPass<HistogramPolicy, WarpRankPolicy, ItemsPerThread>` | Default pass policy. Builds block histograms, scans them, then scatters by local rank. |
| Pass policy | `FlagPrefixSumPass<ScanPolicy>` | Alternate pass policy. Builds per-bucket flags, scans them, then scatters by scanned positions. |
| Pass policy | `OneSweepPass<BlockHistogramPolicy, LocalRankPolicy, GlobalOffsetsBlockHistogramPolicy, ItemsPerThread>` | Builds global digit offsets up front, then uses per-bucket decoupled lookback during each partition pass. |
| Histogram policy | `SharedAtomicHistogram` | Builds each block-local histogram in shared memory with atomics. |
| Histogram policy | `WarpBallotHistogram` | Builds each block-local histogram from warp ballot bucket counts. |
| Histogram policy | `WarpLevelMultiSplitHistogram` | Builds each block-local histogram while computing warp-level multi-split ranks. |
| OneSweep block histogram policy | `WarpBallotBlockHistogram` | Builds each block-local histogram from warp ballot bucket counts while computing local ranks. |
| OneSweep global offsets block histogram policy | `SharedAtomicGlobalOffsetsBlockHistogram` | Builds each block's all-pass digit histogram in shared memory; global accumulation after this local step is fixed to `atomicAdd`. |
| Warp rank policy | `BucketBallotWarpRank` | Computes per-block bucket-local ranks using per-bucket warp ballots. |
| Warp rank policy | `WarpLevelMultiSplitWarpRank` | Computes per-block bucket-local ranks with warp-level multi-split masks. |
| Scan policy | `BucketWiseScan` | Scans each bucket range separately; default for `FlagPrefixSumPass<>`. |
| Scan policy | `FlattenedScan` | Scans the full flattened flag array at once. |

## Workspace

Always query workspace with the same `Config` that will be passed to
`sort_keys` or `sort_by_key`; pass policies use different workspace layouts.

```cpp
const std::size_t workspace_size =
    algo::cuda::sort::required_workspace_size<std::uint32_t>(count);
```

```cpp
const std::size_t workspace_size =
    algo::cuda::sort::required_sort_by_key_workspace_size<
        Config,
        std::uint32_t,
        std::uint32_t,
        float>(count);
```

```cpp
const auto values = algo::cuda::sort::value_arrays(d_ids, d_weights);
const std::size_t workspace_size =
    algo::cuda::sort::required_sort_by_key_workspace_size<std::uint32_t>(
        count, values);
```

For `count <= 1`, no workspace is required. If the workspace pointer is null or
too small when workspace is required, `sort_keys` and `sort_by_key` return
`cudaErrorInvalidValue`.

Workspace shape by pass policy:

| Policy | Key-only workspace | Value sort addition |
| --- | --- | --- |
| `HistogramPass<...>` | `temp_keys`, `histograms`, `scan_workspace` | one `temp_values` buffer per value array |
| `FlagPrefixSumPass<BucketWiseScan>` | `temp_keys`, `flags`, `bucket_offsets`, `scan_workspace` | one `temp_values` buffer per value array |
| `FlagPrefixSumPass<FlattenedScan>` | `temp_keys`, `flags`, `scan_workspace` | one `temp_values` buffer per value array |
| `OneSweepPass<...>` | `temp_keys`, `global_offsets`, `tile_counter`, `lookback_records` | one `temp_values` buffer per value array |

## Supported Types

| API | Supported types |
| --- | --- |
| `sort_keys` | `Key` must be an unsigned integral type or `algo::cuda::sort::UIntKey<Words>`. |
| `sort_by_key` | `Key` must be an unsigned integral type or `algo::cuda::sort::UIntKey<Words>`; every value array element type must be trivially copyable. |

`UIntKey<Words>` stores `Words` little-endian 32-bit words: `words[0]` is the
least significant word. If `KeyBits` is less than the key width, ordering is
defined by the low-order `KeyBits` bits rather than by the full key value.

## Examples

Default key sort:

```cpp
const std::size_t workspace_size =
    algo::cuda::sort::required_workspace_size<std::uint32_t>(count);

auto status = algo::cuda::sort::sort_keys<std::uint32_t>(
    d_keys, count, d_workspace, workspace_size, stream);
```

Value arrays sorted by key:

```cpp
const std::size_t workspace_size =
    algo::cuda::sort::required_sort_by_key_workspace_size<
        Config,
        std::uint32_t,
        std::uint32_t,
        float>(count);

auto status = algo::cuda::sort::sort_by_key<Config>(
    d_keys,
    algo::cuda::sort::value_arrays(d_ids, d_weights),
    count,
    d_workspace,
    workspace_size,
    stream);
```

Explicit flag-prefix-sum sort:

```cpp
using Config = algo::cuda::sort::RadixSort<
    algo::cuda::sort::FlagPrefixSumPass<
        algo::cuda::sort::BucketWiseScan>,
    256,
    4,
    32>;
```

Explicit OneSweep sort:

```cpp
using Config = algo::cuda::sort::RadixSort<
    algo::cuda::sort::OneSweepPass<
        algo::cuda::sort::WarpBallotBlockHistogram,
        algo::cuda::sort::WarpLevelMultiSplitWarpRank,
        algo::cuda::sort::SharedAtomicGlobalOffsetsBlockHistogram,
        1>,
    256,
    4,
    32>;
```

Sort only 16 low-order bits:

```cpp
using Config = algo::cuda::sort::RadixSort<
    algo::cuda::sort::HistogramPass<
        algo::cuda::sort::SharedAtomicHistogram,
        algo::cuda::sort::WarpLevelMultiSplitWarpRank,
        4>,
    256,
    4,
    16>;
```

Fixed-width 96-bit key sort:

```cpp
using Key = algo::cuda::sort::UIntKey<3>;
using Config = algo::cuda::sort::RadixSort<
    algo::cuda::sort::HistogramPass<
        algo::cuda::sort::SharedAtomicHistogram,
        algo::cuda::sort::WarpLevelMultiSplitWarpRank,
        4>,
    256,
    4,
    96>;

auto status = algo::cuda::sort::sort_keys<Config, Key>(
    d_keys, count, d_workspace, workspace_size, stream);
```

## Selection Guide

| Goal | Configuration to try |
| --- | --- |
| General use | Default histogram sort with `WarpLevelMultiSplitWarpRank`. |
| Lower workspace for large inputs | Tune `ItemsPerThread` in `HistogramPass<SharedAtomicHistogram, WarpLevelMultiSplitWarpRank, ItemsPerThread>`. |
| Compare flag scan strategies | `FlagPrefixSumPass<BucketWiseScan>` vs. `FlagPrefixSumPass<FlattenedScan>`. |
| Fewer passes | Increase `RadixBits` within the pass policy limit. |
| Smaller key domain | Lower `KeyBits`. |
