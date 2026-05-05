# CUDA Scan

`algo::cuda::scan` provides in-place inclusive and exclusive prefix sum on CUDA
device memory. The implementation is selected with a compile-time configuration
type.

The public entry points are:

```cpp
#include <algo/cuda/scan.cuh>
```

```cpp
algo::cuda::scan::inclusive_sum<Config, T>(
    d_data, count, d_workspace, workspace_size, stream);

algo::cuda::scan::exclusive_sum<Config, T>(
    d_data, count, d_workspace, workspace_size, stream);

algo::cuda::scan::required_workspace_size<Config, T>(count);
```

Overloads without `Config` use the default configuration.

## Default Configuration

```cpp
using DefaultConfig =
    algo::cuda::scan::BlockBased<
        algo::cuda::scan::DecoupledLookback<
            algo::cuda::scan::WarpShuffleBlock<>>>;
```

The default uses a block-based decoupled lookback scan with a warp-shuffle block
scan. `WarpShuffleBlock<>` defaults to `WarpShuffleBlock<256, 4>`, so the
default tile size is `256 * 4` items.

## Configuration Model

The scan configuration has two top-level families.

| Config | Meaning |
| --- | --- |
| `BlockBased<BlockAlgorithm>` | Scans one tile per block and combines block results according to the selected block algorithm. |
| `Global<Algorithm>` | Runs a whole-array global scan algorithm. |

`BlockBased` is the main configurable path. Its nested policy structure is:

```cpp
BlockBased<
    ScanThenPropagate<
        HillisSteeleBlock<BlockSize, ItemsPerThread>>>
```

or:

```cpp
BlockBased<
    ReduceThenScan<
        BlellochBlock<BlockSize, ItemsPerThread, Layout>,
        BlockTileReduction>>
```

or:

```cpp
BlockBased<
    DecoupledLookback<
        WarpShuffleBlock<BlockSize, ItemsPerThread>>>
```

## Top-Level Config

| Policy | Parameters | Notes |
| --- | --- | --- |
| `BlockBased<BlockAlgorithm>` | `BlockAlgorithm` | Uses a per-block scan policy and a block algorithm that combines per-tile results for large inputs. |
| `Global<Algorithm>` | `Algorithm` | Uses a global scan implementation directly over the whole input. |

## `BlockBased<BlockAlgorithm>`

`BlockBased` contains a `BlockAlgorithm`. The block algorithm controls the
multi-block structure: how per-block work is combined into a whole-array scan.

### `BlockAlgorithm`

Block algorithms define how multi-block inputs are handled.

| Policy | Parameters | Default parameter values | Notes |
| --- | --- | --- | --- |
| `ScanThenPropagate<BlockScan>` | `BlockScan` | none | Each block scans its tile and writes block sums. The block sums are scanned, then added back to each tile. |
| `ReduceThenScan<BlockScan, ReducePolicy>` | `BlockScan`, `ReducePolicy` | `ReducePolicy = BlockTileReduction` | Reduces each tile first, scans the tile sums, then runs the per-block scan and applies offsets. |
| `DecoupledLookback<BlockScan>` | `BlockScan` | none | Each block publishes its tile aggregate and looks back over prior block records to compute its prefix. This avoids the separate block-offset propagation pass. |

`ScanThenPropagate`, `ReduceThenScan`, and `DecoupledLookback` produce the same
prefix-sum result. The difference is kernel structure, synchronization strategy,
and intermediate memory traffic. Use benchmarks to choose between them for a
target GPU and input distribution.

#### `BlockAlgorithm` > `BlockScan`

`ScanThenPropagate`, `ReduceThenScan`, and `DecoupledLookback` contain a
`BlockScan`. The block scan policy defines how one tile is scanned inside a CUDA
block.

| Policy | Parameters | Default values | Tile size | Constraints | Notes |
| --- | --- | --- | --- | --- | --- |
| `BlellochBlock<BlockSize, ItemsPerThread, Layout>` | `BlockSize`, `ItemsPerThread`, `Layout` | `256`, `2`, `DirectSharedLayout` | `BlockSize * ItemsPerThread` | `BlockSize > 0`, `ItemsPerThread > 0`; implementation requires both to be powers of two. | Shared-memory Blelloch scan. Supports shared-memory layout selection. |
| `HillisSteeleBlock<BlockSize, ItemsPerThread>` | `BlockSize`, `ItemsPerThread` | `256`, `2` | `BlockSize * ItemsPerThread` | `BlockSize > 0`, `ItemsPerThread > 0`; implementation requires both to be powers of two. | Shared-memory Hillis-Steele scan. |
| `WarpShuffleBlock<BlockSize, ItemsPerThread>` | `BlockSize`, `ItemsPerThread` | `256`, `4` | `BlockSize * ItemsPerThread` | `BlockSize > 0`, `BlockSize % 32 == 0`, `BlockSize <= 1024`, `ItemsPerThread > 0`. | Uses warp shuffle operations for intra-warp work. Used by the default configuration. |

##### `BlellochBlock` > `Layout`

`BlellochBlock` contains a `Layout` policy. The layout controls how logical
indices are mapped to shared-memory storage.

| Policy | Meaning |
| --- | --- |
| `DirectSharedLayout` | Maps logical shared-memory indices directly to storage indices. |
| `PaddedSharedLayout` | Adds padding based on 32 banks with 4-byte bank width to reduce bank conflicts. |

#### `ReduceThenScan` > `ReducePolicy`

`ReduceThenScan` contains a `ReducePolicy`. The reduce policy controls how block
tile sums are produced before the recursive scan over tile sums.

| Policy | Meaning |
| --- | --- |
| `BlockTileReduction` | Reduces each tile with a block-level reduction and writes one sum per tile. |

## `Global<Algorithm>`

`Global` contains an `Algorithm`. The algorithm controls the whole-array scan
implementation.

| Policy | Parameters | Default values | Workspace | Notes |
| --- | --- | --- | --- | --- |
| `BlellochGlobal<BlockSize>` | `BlockSize` | `256` | `sizeof(T) * next_power_of_two(count)` for `count > 1` | Pads the input to a power of two in workspace. |
| `HillisSteeleGlobal<BlockSize>` | `BlockSize` | `256` | `sizeof(T) * count` for `count > 1` | Alternates between the input and workspace across scan steps. |

Both global policies require `BlockSize > 0`.

## Workspace

Call `required_workspace_size<Config, T>(count)` before launching the scan.

```cpp
using Config = algo::cuda::scan::BlockBased<
    algo::cuda::scan::ScanThenPropagate<
        algo::cuda::scan::HillisSteeleBlock<256, 2>>>;

const std::size_t workspace_size =
    algo::cuda::scan::required_workspace_size<Config, std::uint32_t>(count);
```

For block-based configurations, workspace is not required when the input fits in
one tile. For larger inputs, workspace stores block sums and any recursive scan
workspace needed for those block sums.

For `DecoupledLookback`, workspace stores one lookback record per tile for
inputs larger than one tile. Each record contains the tile aggregate, the
published prefix, and a small state field. The record type is an implementation
detail; use `required_workspace_size` instead of computing this directly.

For global configurations, workspace is required for `count > 1`.

If the workspace pointer is null or too small when workspace is required,
`inclusive_sum` and `exclusive_sum` return `cudaErrorInvalidValue`.

## Supported Value Types

`T` must be an integral or floating-point type.

The scan operation is addition with `T{0}` as the identity. Floating-point
results may differ across configurations because the reduction order can differ.

## Examples

Default configuration:

```cpp
auto status = algo::cuda::scan::inclusive_sum<std::uint32_t>(
    d_data, count, d_workspace, workspace_size, stream);
```

Explicit Hillis-Steele block configuration:

```cpp
using Config = algo::cuda::scan::BlockBased<
    algo::cuda::scan::ScanThenPropagate<
        algo::cuda::scan::HillisSteeleBlock<256, 2>>>;

auto status = algo::cuda::scan::exclusive_sum<Config, std::uint32_t>(
    d_data, count, d_workspace, workspace_size, stream);
```

Explicit default-equivalent decoupled lookback configuration:

```cpp
using Config = algo::cuda::scan::BlockBased<
    algo::cuda::scan::DecoupledLookback<
        algo::cuda::scan::WarpShuffleBlock<>>>;
```

Blelloch block scan with padded shared memory:

```cpp
using Config = algo::cuda::scan::BlockBased<
    algo::cuda::scan::ScanThenPropagate<
        algo::cuda::scan::BlellochBlock<
            256,
            2,
            algo::cuda::scan::PaddedSharedLayout>>>;
```

Reduce-then-scan structure:

```cpp
using Config = algo::cuda::scan::BlockBased<
    algo::cuda::scan::ReduceThenScan<
        algo::cuda::scan::WarpShuffleBlock<256, 4>>>;
```

Decoupled lookback with a Blelloch block scan:

```cpp
using Config = algo::cuda::scan::BlockBased<
    algo::cuda::scan::DecoupledLookback<
        algo::cuda::scan::BlellochBlock<
            256,
            2,
            algo::cuda::scan::PaddedSharedLayout>>>;
```

Global Blelloch scan:

```cpp
using Config = algo::cuda::scan::Global<
    algo::cuda::scan::BlellochGlobal<256>>;
```

## Selection Guide

| Goal | Configuration to try |
| --- | --- |
| General use | Default configuration. |
| Compare block scan algorithms | Swap `HillisSteeleBlock`, `BlellochBlock`, and `WarpShuffleBlock` under the same `BlockBased` algorithm. |
| Evaluate shared-memory bank conflict behavior | Compare `BlellochBlock<..., DirectSharedLayout>` and `BlellochBlock<..., PaddedSharedLayout>`. |
| Compare multi-block structure | Compare `DecoupledLookback<BlockScan>`, `ScanThenPropagate<BlockScan>`, and `ReduceThenScan<BlockScan>`. |
| Avoid the separate offset-propagation pass | Try `DecoupledLookback<BlockScan>`. |
| Exercise a whole-array algorithm directly | Use `Global<BlellochGlobal<BlockSize>>` or `Global<HillisSteeleGlobal<BlockSize>>`. |
