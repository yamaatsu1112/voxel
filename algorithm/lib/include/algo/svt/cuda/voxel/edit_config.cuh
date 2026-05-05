#pragma once

namespace algo::svt::cuda {

// Policy tags for the edit pipeline. They are intentionally empty: template
// specializations in detail/ choose kernels and workspace layouts from the tag
// combination supplied through EditConfig.
struct AtomicCas {};
struct VoxelCountDispatch {};
struct HostLeafCountDispatch {};

struct Threadwise {};
struct Childwise {};

struct RecoverStartDepth {};
struct StoreStartDepth {};

struct LeafwiseMaterialize {};
struct OffsetSearch {};
struct ExplicitRequests {};
template <class RequestMode> struct Nodewise {};

struct ScanDepthwiseAllocation {};
struct PlainDepthwiseAllocation {};
struct CachedDepthwiseAllocation {};
struct AllDepthAllocation {};
template <class InitMode, class StartDepthMode = RecoverStartDepth,
          class ScheduleMode = LeafwiseMaterialize>
struct CompactAllDepthAllocation {};

template <class Allocation, class Dispatch> struct EditConfig {
    using allocation = Allocation;
    using dispatch = Dispatch;
};

} // namespace algo::svt::cuda
