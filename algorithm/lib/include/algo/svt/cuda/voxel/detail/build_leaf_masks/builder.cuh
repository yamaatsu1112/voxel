#pragma once

#include <algo/cuda/scan/scan.cuh>
#include <algo/svt/cuda/detail/geometry.cuh>
#include <algo/svt/cuda/voxel/detail/workspace/voxel_to_leaf.cuh>
#include <algo/svt/cuda/voxel/types.cuh>

namespace algo::svt::cuda::detail {

namespace leaf_mask_builder {

struct Input {
    const std::uint32_t *keys;
    const std::uint32_t *local_bits;
    LeafMask *leaf_masks;
    std::uint32_t *leaf_count;
    std::uint32_t count;
};

struct Transform {
    __device__ static std::uint32_t run(std::uint32_t index,
                                        const Input &input) {
        const bool valid = input.keys[index] != kInvalidSortKey;
        const bool run_start =
            valid &&
            (index == 0u || input.keys[index - 1u] != input.keys[index]);
        return run_start ? 1u : 0u;
    }
};

struct PostScan {
    __device__ static void run(std::uint32_t index, const Input &input,
                               std::uint32_t transformed_value,
                               std::uint32_t prefix) {
        if (transformed_value != 0u) {
            std::uint32_t low = 0u;
            std::uint32_t high = 0u;
            for (std::uint32_t i = index;
                 i < input.count && input.keys[i] == input.keys[index]; ++i) {
                const std::uint32_t bit = input.local_bits[i];
                if (bit < 32u)
                    low |= 1u << bit;
                else
                    high |= 1u << (bit - 32u);
            }

            input.leaf_masks[prefix] = LeafMask{input.keys[index], low, high};
        }

        if (index == input.count - 1u)
            *input.leaf_count = prefix + transformed_value;
    }
};

} // namespace leaf_mask_builder

inline cudaError_t build_leaf_masks_impl(const VoxelToLeafWorkspace &workspace,
                                         LeafMask *leaf_masks,
                                         std::uint32_t *leaf_count,
                                         std::uint32_t count,
                                         cudaStream_t stream) {
    const leaf_mask_builder::Input input{workspace.keys, workspace.local_bits,
                                         leaf_masks, leaf_count, count};
    return algo::cuda::scan::exclusive_sum_fused<leaf_mask_builder::Transform,
                                                 leaf_mask_builder::PostScan>(
        input, count, workspace.scan_workspace, workspace.scan_workspace_size,
        stream);
}

} // namespace algo::svt::cuda::detail
