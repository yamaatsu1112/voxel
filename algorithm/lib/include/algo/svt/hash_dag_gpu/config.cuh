#pragma once

#include <cstdint>

#ifndef __CUDACC__
#error "algo::svt::hash_dag_gpu requires CUDA compilation with nvcc"
#endif

namespace algo::svt::hash_dag_gpu {

// The current GPU DAG stores a fixed 1024^3 voxel world. Internal nodes are
// octree nodes (one bit per axis per level), and leaves cover 4^3 voxels that
// are represented as a single 64-bit occupancy mask.
inline constexpr std::uint32_t kWorldVoxelCountExp = 10;
inline constexpr std::uint32_t kWorldVoxelCount = 1u << kWorldVoxelCountExp;
inline constexpr std::uint32_t kBranchFactorExp = 1;
inline constexpr std::uint32_t kGroupSize = 8;
inline constexpr std::uint32_t kGroupSizeExp = 3;
inline constexpr std::uint32_t kLeafVoxelCountExp = 2;
inline constexpr std::uint32_t kLeafVoxelCount = 1u << kLeafVoxelCountExp;
inline constexpr std::uint32_t kLeafVoxelCountTotal =
    kLeafVoxelCount * kLeafVoxelCount * kLeafVoxelCount;
inline constexpr std::uint32_t kInnerDepth =
    (kWorldVoxelCountExp - kLeafVoxelCountExp) / kBranchFactorExp;
inline constexpr std::uint32_t kMaxItemWords = 1u + kGroupSize;

inline constexpr std::uint32_t kDefaultBucketCount = 1024;
inline constexpr std::uint32_t kDefaultOverflowSlabCount = 1024;

// References are compact 32-bit values. 0 and 1 are reserved sentinels for
// completely empty/full subtrees; all other values point into one of the
// per-item-width hash tables. The high nibble stores the item width so device
// code can choose the right table without extra metadata.
inline constexpr std::uint32_t kEmptyRef = 0u;
inline constexpr std::uint32_t kFullRef = 1u;
inline constexpr std::uint32_t kRefPtrMask = 0x0fffffffu;
inline constexpr std::uint32_t kRefItemWordsShift = 28u;

// Node item word 0 packs two 8-bit masks:
// - child_mask: child slots that contain table refs stored after the header.
// - fill_mask: child slots that are implicitly kFullRef.
// Missing bits in both masks are implicitly kEmptyRef.
inline constexpr std::uint32_t kChildMaskShift = 0u;
inline constexpr std::uint32_t kFillMaskShift = 8u;
inline constexpr std::uint32_t kNodeMask = 0xffu;

inline constexpr std::uint32_t kInvalidNodeKey = 0xffffffffu;

} // namespace algo::svt::hash_dag_gpu
