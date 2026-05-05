// Dynamic SVO Compute Shader - Slang version
// Handles voxel placement and deletion operations for dynamic voxel objects
// This shader is used for dynamic voxel objects that are rebuilt every frame

import svo;
import utils;
import prefix_sum;

// ============================================================================
// Layer Type Constants for Compute Shader
// ============================================================================

// Constants for compute shader layer types - Allocation
static const uint LAYER_TYPE_COLLECT_REQUESTS = 0u;
static const uint LAYER_TYPE_UNIQUE_FLAGS = 3u;
static const uint LAYER_TYPE_PREFIX_SUM = 4u;
static const uint LAYER_TYPE_COMPACT = 5u;
static const uint LAYER_TYPE_ALLOCATE_BATCH = 6u;
static const uint LAYER_TYPE_UPDATE_LEAVES = 7u;

// Constants for compute shader layer types - Freeing/Integration
static const uint LAYER_TYPE_COLLECT_FREE_REQUESTS = 8u;
static const uint LAYER_TYPE_FREE_BATCH = 9u;

// Constants for compute shader layer types - Dynamic SVO
static const uint LAYER_TYPE_CLEAR_SVO = 10u;


// ============================================================================
// Resource Bindings
// ============================================================================

const static uint NUM_SVO_DATA = 9;

// Set 0: Storage buffers + compute uniform
[[vk::binding(0, 0)]]
RWStructuredBuffer<SVONode> svoNodes[NUM_SVO_DATA];

[[vk::binding(1, 0)]]
RWStructuredBuffer<SVOLeaf> svoLeaves[NUM_SVO_DATA];

[[vk::binding(2, 0)]]
RWStructuredBuffer<SVOCounters> svoCounters[NUM_SVO_DATA];

[[vk::binding(3, 0)]]
RWStructuredBuffer<uint> svoFreeNodeIndices[NUM_SVO_DATA];

[[vk::binding(4, 0)]]
RWStructuredBuffer<uint> svoFreeLeafIndices[NUM_SVO_DATA];

[[vk::binding(5, 0)]]
StructuredBuffer<LeafEditCommand> leafEditCommands[NUM_SVO_DATA];

// Temporary buffers for parallel allocation
[[vk::binding(6, 0)]]
RWStructuredBuffer<NodeRequest> allocationRequestBuffer[NUM_SVO_DATA];

[[vk::binding(7, 0)]]
RWStructuredBuffer<NodeRequest> uniqueRequestBuffer[NUM_SVO_DATA];

[[vk::binding(8, 0)]]
RWStructuredBuffer<uint> flagBuffer[NUM_SVO_DATA];

[[vk::binding(9, 0)]]
RWStructuredBuffer<uint> prefixSumBuffer[NUM_SVO_DATA];

[[vk::binding(10, 0)]]
RWStructuredBuffer<uint> requestCountBuffer[NUM_SVO_DATA];

[[vk::binding(11, 0)]]
ConstantBuffer<UniformBufferObject> ubo;

[[vk::binding(12, 0)]]
StructuredBuffer<uint> commandCountBuffer;

// Push constants for compute shader dispatch
struct PushConstants {
    uint depth;
    uint layerType;
    uint phase; // Phase within parallel edit pipeline
    uint svoIndex; // Index of the SVO being processed (0 = occupancy, 1-8 = ID bit planes)
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pushConstants;

groupshared uint prefixSumSharedData[64];

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint threadIndex = dispatchThreadID.x;

    // CLEAR_SVO: Reset root node and counters (only thread 0 needs to run)
    if (pushConstants.layerType == LAYER_TYPE_CLEAR_SVO) {
        if (threadIndex == 0u) {
            resetGroupedSVO(svoNodes[pushConstants.svoIndex], svoCounters[pushConstants.svoIndex]);
        }
        return;
    }

    if (pushConstants.layerType == LAYER_TYPE_COLLECT_REQUESTS) {
        // Collect allocation requests
        // Each thread writes 0 or 1 to requestCountBuffer[threadIndex]
        uint commandCount = commandCountBuffer[pushConstants.svoIndex];

        if (threadIndex >= commandCount) {
            clearNodeRequest(allocationRequestBuffer[pushConstants.svoIndex], threadIndex);
            return;
        }
        uint paddedSize = nextPowerOf2(commandCount);
        if (threadIndex >= paddedSize) {
            return;
        }

        collectGroupedAllocationRequests(
            svoNodes[pushConstants.svoIndex], leafEditCommands[pushConstants.svoIndex], allocationRequestBuffer[pushConstants.svoIndex],
            requestCountBuffer[pushConstants.svoIndex], pushConstants.depth, threadIndex
        );
    } else if (pushConstants.layerType == LAYER_TYPE_UNIQUE_FLAGS) {
        // Compute unique flags
        uint commandCount = commandCountBuffer[pushConstants.svoIndex];
        groupedComputeUniqueFlags(allocationRequestBuffer[pushConstants.svoIndex], flagBuffer[pushConstants.svoIndex], prefixSumBuffer[pushConstants.svoIndex], commandCount, threadIndex);
    } else if (pushConstants.layerType == LAYER_TYPE_PREFIX_SUM) {
        uint commandCount = commandCountBuffer[pushConstants.svoIndex];
        uint paddedSize = nextPowerOf2(commandCount);
        uint numWG = max(1u, paddedSize / 64u);
        uint localIndex = threadIndex & 63u;
        uint groupIndex = threadIndex >> 6u;
        
        if (pushConstants.phase == 0u) {
            if (threadIndex >= numWG * 64u) {
                return;
            }
            prefixSumSharedLocal(
                prefixSumBuffer[pushConstants.svoIndex],
                requestCountBuffer[pushConstants.svoIndex],
                paddedSize,
                localIndex,
                groupIndex,
                threadIndex,
                prefixSumSharedData
            );
        } else if (pushConstants.phase == 1u) {
            if (threadIndex >= 64u) {
                return;
            }
            prefixSumSharedBlockScan(
                requestCountBuffer[pushConstants.svoIndex],
                numWG,
                localIndex,
                prefixSumSharedData
            );
        } else {
            if (threadIndex >= numWG * 64u) {
                return;
            }
            prefixSumSharedPropagate(
                prefixSumBuffer[pushConstants.svoIndex],
                requestCountBuffer[pushConstants.svoIndex],
                paddedSize,
                localIndex,
                groupIndex,
                threadIndex
            );
        }
    } else if (pushConstants.layerType == LAYER_TYPE_COMPACT) {

        // Compact unique pairs
        uint commandCount = commandCountBuffer[pushConstants.svoIndex];
        if (threadIndex >= commandCount) {
            return;
        }
        groupedCompactUniquePairs(
            allocationRequestBuffer[pushConstants.svoIndex], uniqueRequestBuffer[pushConstants.svoIndex], flagBuffer[pushConstants.svoIndex],
            prefixSumBuffer[pushConstants.svoIndex], commandCount, threadIndex
        );

        // Store nodeCount and leafCount for use in allocation
        if (threadIndex == 0u) {
            requestCountBuffer[pushConstants.svoIndex][1] = svoCounters[pushConstants.svoIndex][0].nodeCount;
            requestCountBuffer[pushConstants.svoIndex][2] = svoCounters[pushConstants.svoIndex][0].leafCount;
            requestCountBuffer[pushConstants.svoIndex][3] = svoCounters[pushConstants.svoIndex][0].freeNodeCount;
            requestCountBuffer[pushConstants.svoIndex][4] = svoCounters[pushConstants.svoIndex][0].freeLeafCount;
        }
    } else if (pushConstants.layerType == LAYER_TYPE_ALLOCATE_BATCH) {
        // Batch allocate nodes
        // Calculate unique count from prefix sum and flags
        uint commandCount = commandCountBuffer[pushConstants.svoIndex];
        uint uniqueCount = 0u;
        if (commandCount > 0u) {
            uniqueCount = prefixSumBuffer[pushConstants.svoIndex][commandCount - 1u];
        }

        if (threadIndex >= uniqueCount) {
            return;
        }

        // Get nodeCount and leafCount from buffer (stored in COMPACT phase)
        uint baseNodeCount = requestCountBuffer[pushConstants.svoIndex][1];
        uint baseLeafCount = requestCountBuffer[pushConstants.svoIndex][2];
        uint baseFreeNodeCount = requestCountBuffer[pushConstants.svoIndex][3];
        uint baseFreeLeafCount = requestCountBuffer[pushConstants.svoIndex][4];
        allocateGroupedNodesBatch(
            svoNodes[pushConstants.svoIndex],
            svoLeaves[pushConstants.svoIndex],
            svoFreeNodeIndices[pushConstants.svoIndex],
            svoFreeLeafIndices[pushConstants.svoIndex],
            uniqueRequestBuffer[pushConstants.svoIndex],
            pushConstants.depth,
            baseNodeCount,
            baseLeafCount,
            baseFreeNodeCount,
            baseFreeLeafCount,
            threadIndex
        );

        // Update nodeCount (only thread 0 needs to do this)
        if (threadIndex == 0u) {
            if (pushConstants.depth < MAX_DEPTH) {
                uint newAllocationCount = uniqueCount - clamp(baseFreeNodeCount, 0u, uniqueCount);
                svoCounters[pushConstants.svoIndex][0].nodeCount = baseNodeCount + newAllocationCount * GROUP_SIZE;
                svoCounters[pushConstants.svoIndex][0].freeNodeCount = baseFreeNodeCount - clamp(uniqueCount, 0u, baseFreeNodeCount);
            } else {
                uint newAllocationCount = uniqueCount - clamp(baseFreeLeafCount, 0u, uniqueCount);
                svoCounters[pushConstants.svoIndex][0].leafCount = baseLeafCount + newAllocationCount * GROUP_SIZE;
                svoCounters[pushConstants.svoIndex][0].freeLeafCount = baseFreeLeafCount - clamp(uniqueCount, 0u, baseFreeLeafCount);
            }
        }
    } else if (pushConstants.layerType == LAYER_TYPE_UPDATE_LEAVES) {
        // Update leaf values from edit commands
        uint commandCount = commandCountBuffer[pushConstants.svoIndex];
        if (threadIndex >= commandCount) {
            return;
        }
        updateGroupedLeafValues<OrPolicy>(
            svoNodes[pushConstants.svoIndex],
            svoLeaves[pushConstants.svoIndex],
            leafEditCommands[pushConstants.svoIndex],
            threadIndex
        );
    } else if (pushConstants.layerType == LAYER_TYPE_COLLECT_FREE_REQUESTS) {
        // Collect free requests from updated nodes/leaves at given depth
        // Each thread processes one edit command and checks if node/leaf at depth should be freed
        uint commandCount = commandCountBuffer[pushConstants.svoIndex];

        if (threadIndex >= commandCount) {
            clearNodeRequest(allocationRequestBuffer[pushConstants.svoIndex], threadIndex);
            return;
        }
        uint paddedSize = nextPowerOf2(commandCount);
        if (threadIndex >= paddedSize) {
            return;
        }

        collectGroupedFreeRequests(
            svoNodes[pushConstants.svoIndex], svoLeaves[pushConstants.svoIndex], leafEditCommands[pushConstants.svoIndex], allocationRequestBuffer[pushConstants.svoIndex],
            requestCountBuffer[pushConstants.svoIndex], pushConstants.depth, threadIndex
        );
    } else if (pushConstants.layerType == LAYER_TYPE_FREE_BATCH) {
        // Batch free nodes or leaves
        // Calculate unique count from prefix sum and flags
        uint commandCount = commandCountBuffer[pushConstants.svoIndex];
        uint uniqueCount = 0u;
        if (commandCount > 0u) {
            uniqueCount = prefixSumBuffer[pushConstants.svoIndex][commandCount - 1u];
        }

        if (threadIndex >= uniqueCount) {
            return;
        }

        if (pushConstants.depth == MAX_DEPTH) {
            // Freeing leaves
            freeGroupedLeavesBatch(
                svoNodes[pushConstants.svoIndex],
                svoLeaves[pushConstants.svoIndex],
                svoCounters[pushConstants.svoIndex],
                svoFreeLeafIndices[pushConstants.svoIndex],
                uniqueRequestBuffer[pushConstants.svoIndex],
                threadIndex
            );
        } else {
            // Freeing nodes at intermediate depth
            freeGroupedNodesBatch(
                svoNodes[pushConstants.svoIndex],
                svoCounters[pushConstants.svoIndex],
                svoFreeNodeIndices[pushConstants.svoIndex],
                uniqueRequestBuffer[pushConstants.svoIndex],
                threadIndex
            );
        }
    }
}
