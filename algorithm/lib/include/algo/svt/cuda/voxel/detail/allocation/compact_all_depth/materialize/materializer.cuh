#pragma once

namespace algo::svt::cuda::detail {

namespace allocation::compact_all_depth {

template <class ScheduleMode, class InitMode, class StartDepthMode>
struct Materializer;

} // namespace allocation::compact_all_depth

} // namespace algo::svt::cuda::detail

#include <algo/svt/cuda/voxel/detail/allocation/compact_all_depth/materialize/explicit_requests.cuh>
#include <algo/svt/cuda/voxel/detail/allocation/compact_all_depth/materialize/leafwise.cuh>
#include <algo/svt/cuda/voxel/detail/allocation/compact_all_depth/materialize/offset_search.cuh>
