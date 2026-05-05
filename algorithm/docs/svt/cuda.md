# CUDA SVT

`algo::svt::cuda` provides CUDA-side sparse voxel tree storage, queries, and
batched voxel edit operations. This document covers the policy configuration for
the CUDA edit pipeline under `svt/cuda`.

The public edit entry points are:

```cpp
#include <algo/svt/cuda/edit.cuh>
```

```cpp
algo::svt::cuda::place_voxel_edits<Config>(
    svo, edits, count, workspace, workspace_size, stream);

algo::svt::cuda::destroy_voxel_edits<Config>(
    svo, edits, count, workspace, workspace_size, stream);

algo::svt::cuda::apply_voxel_edits_workspace_size<Config>(count);
```

Overloads without `Config` use the default edit configuration.
`detail::apply_voxel_edits` is the shared internal implementation used by the
place and destroy wrappers.

## Default Configuration

```cpp
using DefaultEditConfig =
    algo::svt::cuda::EditConfig<
        algo::svt::cuda::CompactAllDepthAllocation<algo::svt::cuda::Threadwise>,
        algo::svt::cuda::HostLeafCountDispatch,
        algo::svt::cuda::Fused,
        algo::svt::cuda::Fused>;
```

The default coalesces voxel edits into leaf masks with the fused builder,
allocates missing paths with compact all-depth threadwise initialization,
dispatches later kernels over the compacted leaf count copied to the host, and
uses fused collapse.

## Configuration Model

`EditConfig` has four top-level policy slots:

```cpp
EditConfig<Allocation, Dispatch, Collapse, BuildLeafMasks>
```

| Slot | Meaning |
| --- | --- |
| `Allocation` | Chooses how missing node and leaf paths are materialized before applying leaf bits. |
| `Dispatch` | Chooses the capacity used when launching stages that consume compacted leaf masks. |
| `Collapse` | Chooses how uniform or empty materialized children are freed after edits. |
| `BuildLeafMasks` | Chooses how sorted voxel edits are reduced into one `LeafMask` per touched leaf. |

The edit pipeline always follows the same high-level order:

1. Convert voxel edits into compact leaf masks.
2. Allocate missing paths to every touched leaf.
3. Apply the requested place or destroy bits.
4. Collapse now-uniform subtrees.

Policies change the implementation strategy for these stages; they do not change
the intended edit result.

## Top-Level Policies

| Policy | Parameters | Valid slot | Notes |
| --- | --- | --- | --- |
| `EditConfig<Allocation, Dispatch, Collapse, BuildLeafMasks>` | `Allocation`, `Dispatch`, `Collapse`, `BuildLeafMasks` | top-level config | Selects all edit pipeline policies. |
| `ScanDepthwiseAllocation<Mode>` | `Mode` | `Allocation` | Records the first missing depth for each touched leaf, then emits only the needed request for each depth. |
| `PlainDepthwiseAllocation<Mode>` | `Mode` | `Allocation` | Walks one depth at a time, collecting and compacting allocation requests for that depth before moving deeper. |
| `CachedDepthwiseAllocation` | none | `Allocation` | Depthwise allocation that carries each leaf's current parent index forward between depths. |
| `AllDepthAllocation` | none | `Allocation` | Collects all missing node and leaf materialization requests, then initializes and links them from one allocation snapshot. |
| `CompactAllDepthAllocation<InitMode, StartDepthMode>` | `InitMode`, `StartDepthMode = RecoverStartDepth` | `Allocation` | Stores compact per-leaf node and leaf counts instead of one request record per missing level. |
| `VoxelCountDispatch` | none | `Dispatch` | Uses the original voxel edit count as the launch capacity for compacted leaf-mask consumers. |
| `HostLeafCountDispatch` | none | `Dispatch` | Copies the compacted leaf count to the host and uses it as the launch capacity. |
| `Fused` | none | `Collapse`, `BuildLeafMasks`, scan/plain allocation `Mode` | Uses fused scan post-processing where the selected stage supports it. |
| `Unfused` | none | `Collapse`, `BuildLeafMasks`, scan/plain allocation `Mode` | Uses separate mark, scan, and compact/reduce kernels. |
| `Threadwise` | none | compact allocation `InitMode` | Initializes each compact all-depth leaf path with one thread per touched leaf. |
| `Childwise` | none | compact allocation `InitMode` | Initializes compact all-depth node child slots with one thread per child and several leaves per block. |
| `RecoverStartDepth` | none | compact allocation `StartDepthMode` | Recovers start depth from scanned compact offsets. |
| `StoreStartDepth` | none | compact allocation `StartDepthMode` | Stores start depth during count collection. |
| `AtomicCas` | none | none in the current edit pipeline | Declared as a policy tag, but `svt/cuda` currently has no implementation specialization using it. |

## Allocation Policies

Allocation policies materialize any missing internal nodes and leaves needed for
the touched leaf masks.

| Policy | Parameters | Notes |
| --- | --- | --- |
| `ScanDepthwiseAllocation<Unfused>` | `Unfused` | Collects each leaf's missing-path state once. For each depth, marks needed emits, scans them, compacts requests, and allocates that depth. |
| `ScanDepthwiseAllocation<Fused>` | `Fused` | Uses the same missing-path state as the unfused variant, but combines per-depth emit marking and request compaction into fused scan callbacks. This is the default allocation policy. |
| `PlainDepthwiseAllocation<Unfused>` | `Unfused` | At every depth, walks the current tree state from the touched leaves, marks unique parent/child requests, scans offsets, compacts requests, and allocates the batch. |
| `PlainDepthwiseAllocation<Fused>` | `Fused` | Same depthwise request model as `PlainDepthwiseAllocation<Unfused>`, but fuses unique-request marking and compaction with the scan stage. |
| `CachedDepthwiseAllocation` | none | Initializes per-leaf parent indices once and advances them after each depth, avoiding repeated root-to-depth walks for later levels. |
| `AllDepthAllocation` | none | Computes all materialization flags across all depths, scans them once, initializes materialized storage, links it, and commits node and leaf counters. |
| `CompactAllDepthAllocation<Threadwise, RecoverStartDepth>` | `InitMode = Threadwise`, `StartDepthMode = RecoverStartDepth` | Counts materialized nodes and leaves per touched leaf, scans compact offsets, recovers start depth from scanned offsets during initialization, and initializes each leaf path from a single thread. |
| `CompactAllDepthAllocation<Childwise, RecoverStartDepth>` | `InitMode = Childwise`, `StartDepthMode = RecoverStartDepth` | Uses the same compact count representation and recovered start depth, but spreads node child-slot initialization across child threads. |
| `CompactAllDepthAllocation<Threadwise, StoreStartDepth>` | `InitMode = Threadwise`, `StartDepthMode = StoreStartDepth` | Stores each leaf's start depth while collecting counts, then initializes each leaf path from a single thread. |
| `CompactAllDepthAllocation<Childwise, StoreStartDepth>` | `InitMode = Childwise`, `StartDepthMode = StoreStartDepth` | Uses stored start depths and child-parallel node initialization. |

`CompactAllDepthAllocation<InitMode>` defaults `StartDepthMode` to
`RecoverStartDepth`.

## Allocation Mode Policies

`Fused` and `Unfused` appear as the `Mode` parameter for scan-depthwise and
plain-depthwise allocation.

| Policy | Meaning |
| --- | --- |
| `Fused` | Combines simple transform or compaction work into scan post-processing callbacks, reducing the number of standalone kernels and intermediate arrays for that stage. |
| `Unfused` | Keeps mark, scan, compact, reduce, or initialize work in separate kernels. This is usually easier to inspect and can be useful as a comparison point. |

The exact fused work depends on the allocation family. For example,
`ScanDepthwiseAllocation<Fused>` fuses per-depth request emission with scan.

## Compact Initialization Policies

`CompactAllDepthAllocation` has an `InitMode` parameter that controls how newly
materialized compact paths are initialized.

| Policy | Meaning |
| --- | --- |
| `Threadwise` | Launches one-dimensional blocks and initializes all nodes and the leaf for one touched leaf from a single thread. |
| `Childwise` | Launches blocks shaped as `kGroupSize` children by a fixed number of leaves. Each child thread initializes the corresponding child slot for the materialized nodes, while child `0` also initializes the leaf payload. |

## Compact Start-Depth Policies

`CompactAllDepthAllocation` has a second parameter that controls how the start
depth for each touched leaf is made available after the per-leaf counts have
been scanned.

| Policy | Meaning |
| --- | --- |
| `RecoverStartDepth` | Does not store start depth directly. It recovers it from the scanned node and leaf offsets. This keeps the compact representation smaller. |
| `StoreStartDepth` | Stores start depth during the count collection pass and reuses that value later. This avoids recovering the depth from scanned offsets. |

## Dispatch Policies

The build-leaf-mask stage writes a device-side `leaf_count`. Later stages also
need a launch capacity for arrays sized by the original edit count. Dispatch
policies choose that capacity.

| Policy | Meaning |
| --- | --- |
| `VoxelCountDispatch` | Uses the original voxel edit count. This avoids a device-to-host copy and is safe because the number of touched leaves cannot exceed the number of input edits. |
| `HostLeafCountDispatch` | Copies `leaf_count` from device to host, synchronizes the stream, and dispatches over the exact compacted leaf count. This can reduce later kernel work when many edits collapse into fewer leaves, but adds a host synchronization point. |

## Collapse Policies

Collapse runs from the maximum depth back toward the root after leaf bits have
been applied. It frees materialized children whose region can be represented by
the parent slot alone.

| Policy | Meaning |
| --- | --- |
| `Fused` | Collects free-request keys and compacts unique requests through fused scan callbacks before freeing the batch for each depth. |
| `Unfused` | Collects free-request keys, marks unique offsets, scans them, compacts unique requests, then frees the batch for each depth. |

Both policies use the same leaf masks and walk depths from `kMaxDepth` down to
`1`.

## Build-Leaf-Mask Policies

Before allocation, raw voxel edits are packed into sortable `(leaf_key,
leaf-local bit)` records and sorted. The build-leaf-mask policy reduces each run
of equal leaf keys into one `LeafMask`.

| Policy | Meaning |
| --- | --- |
| `Fused` | Uses fused scan callbacks to identify run starts, emit leaf masks, and write the final leaf count. |
| `Unfused` | Marks leaf-key run starts, scans run offsets, then reduces each run in a separate kernel. |

Invalid voxel coordinates are packed with `kInvalidSortKey` and dropped by both
builders.

## Examples

Default-equivalent configuration:

```cpp
using Config = algo::svt::cuda::EditConfig<
    algo::svt::cuda::ScanDepthwiseAllocation<algo::svt::cuda::Fused>,
    algo::svt::cuda::VoxelCountDispatch,
    algo::svt::cuda::Fused,
    algo::svt::cuda::Fused>;
```

Exact leaf-count dispatch with unfused collapse:

```cpp
using Config = algo::svt::cuda::EditConfig<
    algo::svt::cuda::ScanDepthwiseAllocation<algo::svt::cuda::Fused>,
    algo::svt::cuda::HostLeafCountDispatch,
    algo::svt::cuda::Unfused,
    algo::svt::cuda::Fused>;
```

Compact all-depth allocation that stores start depth:

```cpp
using Config = algo::svt::cuda::EditConfig<
    algo::svt::cuda::CompactAllDepthAllocation<
        algo::svt::cuda::Threadwise,
        algo::svt::cuda::StoreStartDepth>,
    algo::svt::cuda::VoxelCountDispatch,
    algo::svt::cuda::Fused,
    algo::svt::cuda::Fused>;
```

Unfused pipeline for comparison or debugging:

```cpp
using Config = algo::svt::cuda::EditConfig<
    algo::svt::cuda::PlainDepthwiseAllocation<algo::svt::cuda::Unfused>,
    algo::svt::cuda::VoxelCountDispatch,
    algo::svt::cuda::Unfused,
    algo::svt::cuda::Unfused>;
```
