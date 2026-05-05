#pragma once

#include <cstdint>

#ifndef __CUDACC__
#error "algo::svt::cuda requires CUDA compilation with nvcc"
#endif

namespace algo::svt::cuda {

// World layout used by the CUDA SVO implementation.
//
// A voxel coordinate is in a 4096^3 world. The lowest 2 bits per axis address a
// 4^3 leaf payload, and the remaining bits are consumed one bit per axis at
// each internal tree depth. With kBranchFactorExp == 1 every internal node has
// 2 * 2 * 2 children, so one child choice is packed into 3 bits.
inline constexpr std::uint32_t kWorldVoxelCountExp = 12;
inline constexpr std::uint32_t kWorldVoxelCount = 1u << kWorldVoxelCountExp;
inline constexpr std::uint32_t kLeafVoxelCountExp = 2;
inline constexpr std::uint32_t kLeafVoxelCount = 1u << kLeafVoxelCountExp;
inline constexpr std::uint32_t kLeafCountPerAxis =
    kWorldVoxelCount / kLeafVoxelCount;
inline constexpr std::uint32_t kBranchFactorExp = 1;
inline constexpr std::uint32_t kMaxDepth =
    (kWorldVoxelCountExp - kLeafVoxelCountExp) / kBranchFactorExp;
inline constexpr std::uint32_t kGroupSize = 1u << (kBranchFactorExp * 3u);
inline constexpr std::uint32_t kGroupSizeExp = kBranchFactorExp * 3u;
static_assert(kMaxDepth * kGroupSizeExp <= 32u,
              "leaf keys are packed into std::uint32_t");
inline constexpr std::uint32_t kDefaultMaxNodeCount = 131072;
inline constexpr std::uint32_t kDefaultMaxLeafCount = 524288;
inline constexpr std::uint32_t kRootNodeIndex = 0;

// GpuSvoNode::child_data packs both topology and uniform-fill state:
// - low 30 bits: child node or leaf index
// - bit 30:      the child pointer is present
// - bit 31:      the child region is implicitly filled when no child exists
inline constexpr std::uint32_t kChildIndexMask = 0x3fffffffu;
inline constexpr std::uint32_t kChildMaskBit = 1u << 30u;
inline constexpr std::uint32_t kFilledBit = 1u << 31u;
inline constexpr std::uint32_t kInvalidSortKey = 0xffffffffu;
inline constexpr std::uint32_t kInvalidRequestKey = 0xffffffffu;

} // namespace algo::svt::cuda
