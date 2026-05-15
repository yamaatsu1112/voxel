# CUDA SVT

`algo::svt::cuda` provides CUDA-side sparse voxel tree storage, queries, and
batched edit operations. Include the aggregate header for the full CUDA SVO API:

```cpp
#include <algo/svt/cuda.cuh>
```

The CUDA SVO world is fixed at `4096^3` voxels. Internal nodes have 8 children,
and each leaf stores a `4^3` voxel payload as two 32-bit masks. `GpuSvo` owns the
device allocations; kernels consume the non-owning `DeviceGpuSvo` returned by
`GpuSvo::view()`.

## Storage

```cpp
algo::svt::cuda::GpuSvo<> svo;
auto status = svo.status();
status = algo::svt::cuda::reset_svo(svo.view(), stream);
```

`GpuSvo<MaxNodeCount, MaxLeafCount>` allocates nodes, leaves, counters, and free
lists. Construction only allocates memory; call `reset_svo` before using the
tree. The defaults are:

| Capacity | Default |
| --- | --- |
| `MaxNodeCount` | `131072` |
| `MaxLeafCount` | `524288` |

## Voxel Edits

Voxel edit entry points accept absolute world coordinates:

```cpp
#include <algo/svt/cuda/voxel/edit.cuh>
```

```cpp
algo::svt::cuda::place_voxel_edits<Config>(
    svo, edits, count, workspace, workspace_size, stream);

algo::svt::cuda::destroy_voxel_edits<Config>(
    svo, edits, count, workspace, workspace_size, stream);

algo::svt::cuda::apply_voxel_edits_workspace_size<Config>(count);
```

Overloads without `Config` use the default voxel edit configuration:

```cpp
using DefaultEditConfig =
    algo::svt::cuda::EditConfig<
        algo::svt::cuda::CompactAllDepthAllocation<
            algo::svt::cuda::Threadwise>,
        algo::svt::cuda::HostLeafCountDispatch>;
```

The edit pipeline always:

1. Coalesces raw voxel edits into one `LeafMask` per touched leaf.
2. Allocates missing paths to every touched leaf.
3. Applies place or destroy bits to leaf payloads.
4. Collapses now-uniform subtrees.

`voxel_edits_to_leaf_masks` exposes the first stage separately for callers that
want the compacted `LeafMask` representation without mutating the tree.

## Voxel Edit Configuration

`EditConfig` has two top-level policy slots:

```cpp
EditConfig<Allocation, Dispatch>
```

| Slot | Meaning |
| --- | --- |
| `Allocation` | Chooses how missing node and leaf paths are materialized. |
| `Dispatch` | Chooses the launch capacity for stages that consume compacted leaf masks. |

Allocation policies:

| Policy | Notes |
| --- | --- |
| `ScanDepthwiseAllocation` | Records each leaf's missing path once, then emits needed requests depth by depth. |
| `PlainDepthwiseAllocation` | Walks tree state at every depth, collecting and allocating unique requests for that depth. |
| `CachedDepthwiseAllocation` | Carries current parent indices forward between depths to avoid repeated root-to-depth walks. |
| `AllDepthAllocation` | Collects all materialization requests, initializes storage, links it, and commits counters from one allocation snapshot. |
| `CompactAllDepthAllocation<InitMode, StartDepthMode, ScheduleMode>` | Stores compact per-leaf node and leaf counts instead of one request record per missing level. |

`CompactAllDepthAllocation` defaults to
`CompactAllDepthAllocation<InitMode, RecoverStartDepth, LeafwiseMaterialize>`.

Compact allocation sub-policies:

| Policy | Meaning |
| --- | --- |
| `Threadwise` | Initializes each compact path from one thread per touched leaf. |
| `Childwise` | Spreads node child-slot initialization across child threads. |
| `RecoverStartDepth` | Recovers the first missing depth from scanned compact offsets. |
| `StoreStartDepth` | Stores the first missing depth during count collection. |
| `LeafwiseMaterialize` | Materializes one compact path per touched leaf. |
| `Nodewise<OffsetSearch>` | Materializes node-wise requests by deriving node offsets from compact ranges. |
| `Nodewise<ExplicitRequests>` | Materializes node-wise requests from explicit request records. |

Dispatch policies:

| Policy | Meaning |
| --- | --- |
| `VoxelCountDispatch` | Uses the original voxel edit count as launch capacity. This avoids a host synchronization. |
| `HostLeafCountDispatch` | Copies the compacted leaf count to the host and launches later stages over the exact leaf count. |

## Terminal Edits

Terminal edits operate on already-coalesced terminal node and leaf inputs:

```cpp
#include <algo/svt/cuda/terminal/edit.cuh>
```

```cpp
algo::svt::cuda::place_terminal_edits<Config>(
    svo, nodes, node_count, leaves, leaf_count, workspace, workspace_size,
    stream);

algo::svt::cuda::destroy_terminal_edits<Config>(
    svo, nodes, node_count, leaves, leaf_count, workspace, workspace_size,
    stream);
```

The default terminal edit configuration is:

```cpp
using DefaultTerminalEditConfig =
    algo::svt::cuda::TerminalEditConfig<
        algo::svt::cuda::TerminalCompactAllDepthAllocation,
        algo::svt::cuda::TerminalFrontierRelease>;
```

Terminal edit policy slots:

| Slot | Policies |
| --- | --- |
| `Allocation` | `TerminalPlainDepthwiseAllocation`, `TerminalCompactAllDepthAllocation` |
| `Release` | `TerminalDepthwiseRelease`, `TerminalFrontierRelease` |

Workspace sizing for terminal edits is currently exposed through the detail
helper used by tests and benchmarks:

```cpp
const auto workspace_size =
    algo::svt::cuda::detail::apply_terminal_edits_workspace_size<
        Config::allocation,
        Config::release>(node_count, leaf_count, request_capacity);
```

Choose `request_capacity` large enough for the emitted terminal requests. The
request count depends on the covered terminal boxes after world-offset clipping;
tests and benchmarks size this explicitly from the input shape they generate.

## Examples

Default voxel edit:

```cpp
const auto workspace_size =
    algo::svt::cuda::apply_voxel_edits_workspace_size(count);

auto status = algo::svt::cuda::place_voxel_edits(
    svo.view(), d_edits, count, d_workspace, workspace_size, stream);
```

Compact all-depth allocation that stores start depth:

```cpp
using Config = algo::svt::cuda::EditConfig<
    algo::svt::cuda::CompactAllDepthAllocation<
        algo::svt::cuda::Threadwise,
        algo::svt::cuda::StoreStartDepth>,
    algo::svt::cuda::VoxelCountDispatch>;
```

Node-wise compact allocation:

```cpp
using Config = algo::svt::cuda::EditConfig<
    algo::svt::cuda::CompactAllDepthAllocation<
        algo::svt::cuda::Childwise,
        algo::svt::cuda::RecoverStartDepth,
        algo::svt::cuda::Nodewise<algo::svt::cuda::OffsetSearch>>,
    algo::svt::cuda::HostLeafCountDispatch>;
```

Terminal edit with depthwise release:

```cpp
using Config = algo::svt::cuda::TerminalEditConfig<
    algo::svt::cuda::TerminalCompactAllDepthAllocation,
    algo::svt::cuda::TerminalDepthwiseRelease>;
```
