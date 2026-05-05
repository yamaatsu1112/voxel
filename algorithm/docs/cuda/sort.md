# CUDA Sort

`algo::cuda::sort` provides in-place CUDA radix sort APIs for `std::uint32_t`
keys. Key-only sorting and key-value pair sorting are supported.

```cpp
#include <algo/cuda/sort.cuh>
```

## API

```cpp
algo::cuda::sort::sort_keys<Config, Key>(
    d_keys, count, d_workspace, workspace_size, stream);

algo::cuda::sort::sort_pairs<Config, Key, Value>(
    d_keys, d_values, count, d_workspace, workspace_size, stream);

algo::cuda::sort::required_workspace_size<Config, Key>(count);
algo::cuda::sort::required_pairs_workspace_size<Config, Key, Value>(count);
```

All four APIs also have overloads without `Config`; those use the default
configuration.

## Default Configuration

The default is histogram radix sort:

```cpp
using DefaultConfig =
    algo::cuda::sort::RadixSort<
        algo::cuda::sort::HistogramPass<
            algo::cuda::sort::FlattenedHistogram,
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
            algo::cuda::sort::FlattenedHistogram,
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

`RadixBits` must be positive and less than 32, `KeyBits` must be in `(0, 32]`,
and `KeyBits % RadixBits == 0`.
`OneSweepPass` currently supports `RadixBits <= 8`.

Available pass policies:

```cpp
algo::cuda::sort::HistogramPass<
    HistogramLayoutPolicy,
    HistogramPolicy,
    LocalRankPolicy,
    ItemsPerThread>

algo::cuda::sort::FlagPrefixSumPass<ScanPolicy>

algo::cuda::sort::OneSweepPass<
    BlockHistogramPolicy,
    LocalRankPolicy,
    GlobalOffsetsBlockHistogramPolicy,
    ItemsPerThread>
```

The default sort uses `HistogramPass<FlattenedHistogram,
SharedAtomicHistogram, WarpLevelMultiSplitWarpRank, 4>` with a block size of `512`.
`FlagPrefixSumPass<>` uses `BucketWiseScan`; it can also be configured with
`FlattenedScan`. OneSweep should specify `WarpLevelMultiSplitWarpRank` explicitly and uses
`SharedAtomicGlobalOffsetsBlockHistogram` by default; global histogram
accumulation is fixed to `atomicAdd`.

Implemented policies:

| Role | Policies | Notes |
| --- | --- | --- |
| Pass policy | `HistogramPass<HistogramLayoutPolicy, HistogramPolicy, LocalRankPolicy, ItemsPerThread>` | Default pass policy. Builds block histograms, scans them, then scatters by local rank. |
| Pass policy | `FlagPrefixSumPass<ScanPolicy>` | Alternate pass policy. Builds per-bucket flags, scans them, then scatters by scanned positions. |
| Pass policy | `OneSweepPass<BlockHistogramPolicy, LocalRankPolicy, GlobalOffsetsBlockHistogramPolicy, ItemsPerThread>` | Builds global digit offsets up front, then uses per-bucket decoupled lookback during each partition pass. |
| Histogram layout policy | `FlattenedHistogram` | Stores block histograms as `histograms[bucket * num_blocks + block]`. |
| Histogram policy | `SharedAtomicHistogram` | Builds each block-local histogram in shared memory with atomics. |
| OneSweep block histogram policy | `WarpBallotBlockHistogram` | Builds each block-local histogram from warp ballot bucket counts while computing local ranks. |
| OneSweep global offsets block histogram policy | `SharedAtomicGlobalOffsetsBlockHistogram` | Builds each block's all-pass digit histogram in shared memory; global accumulation after this local step is fixed to `atomicAdd`. |
| Warp rank policy | `BucketBallotWarpRank` | Computes per-block bucket-local ranks using per-bucket warp ballots. |
| Warp rank policy | `WarpLevelMultiSplitWarpRank` | Computes per-block bucket-local ranks with warp-level multi-split masks. |
| Scan policy | `BucketWiseScan` | Scans each bucket range separately; default for `FlagPrefixSumPass<>`. |
| Scan policy | `FlattenedScan` | Scans the full flattened flag array at once. |

## Workspace

Always query workspace with the same `Config` that will be passed to
`sort_keys` or `sort_pairs`; pass policies use different workspace layouts.

```cpp
const std::size_t workspace_size =
    algo::cuda::sort::required_workspace_size<std::uint32_t>(count);
```

```cpp
const std::size_t workspace_size =
    algo::cuda::sort::required_pairs_workspace_size<
        std::uint32_t,
        MyValue>(count);
```

For `count <= 1`, no workspace is required. If the workspace pointer is null or
too small when workspace is required, `sort_keys` and `sort_pairs` return
`cudaErrorInvalidValue`.

Workspace shape by pass policy:

| Policy | Key-only workspace | Pair sort addition |
| --- | --- | --- |
| `HistogramPass<FlattenedHistogram, ...>` | `temp_keys`, `histograms`, `scan_workspace` | `temp_values` |
| `FlagPrefixSumPass<BucketWiseScan>` | `temp_keys`, `flags`, `bucket_offsets`, `scan_workspace` | `temp_values` |
| `FlagPrefixSumPass<FlattenedScan>` | `temp_keys`, `flags`, `scan_workspace` | `temp_values` |
| `OneSweepPass<...>` | `temp_keys`, `global_offsets`, `tile_counter`, `lookback_records` | `temp_values` |

## Supported Types

| API | Supported types |
| --- | --- |
| `sort_keys` | `Key` must be `std::uint32_t`. |
| `sort_pairs` | `Key` must be `std::uint32_t`; `Value` must be trivially copyable. |

If `KeyBits < 32`, ordering is defined by the low-order `KeyBits` bits rather
than by the full key value.

## Examples

Default key sort:

```cpp
const std::size_t workspace_size =
    algo::cuda::sort::required_workspace_size<std::uint32_t>(count);

auto status = algo::cuda::sort::sort_keys<std::uint32_t>(
    d_keys, count, d_workspace, workspace_size, stream);
```

Default key-value pair sort:

```cpp
const std::size_t workspace_size =
    algo::cuda::sort::required_pairs_workspace_size<
        std::uint32_t,
        MyValue>(count);

auto status = algo::cuda::sort::sort_pairs<std::uint32_t, MyValue>(
    d_keys, d_values, count, d_workspace, workspace_size, stream);
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
        algo::cuda::sort::FlattenedHistogram,
        algo::cuda::sort::SharedAtomicHistogram,
        algo::cuda::sort::WarpLevelMultiSplitWarpRank,
        4>,
    256,
    4,
    16>;
```

## Selection Guide

| Goal | Configuration to try |
| --- | --- |
| General use | Default histogram sort with `WarpLevelMultiSplitWarpRank`. |
| Lower workspace for large inputs | `HistogramPass<FlattenedHistogram, SharedAtomicHistogram, WarpLevelMultiSplitWarpRank, ItemsPerThread>`. |
| Compare flag scan strategies | `FlagPrefixSumPass<BucketWiseScan>` vs. `FlagPrefixSumPass<FlattenedScan>`. |
| Fewer passes | Increase `RadixBits` within the pass policy limit. |
| Smaller key domain | Lower `KeyBits`. |
