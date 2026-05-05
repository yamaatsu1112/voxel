// SVO Types - Data Structures and Constants
// This module contains shared data structures and constants for SVO operations.

implementing svo;

public static const uint VOXEL_SIZE_NEG_EXP = 4;
public static const float VOXEL_SIZE = pow(2.0, -float(VOXEL_SIZE_NEG_EXP));
public static const float VOXEL_SIZE_INV = 1.0 / VOXEL_SIZE;
public static const uint WORLD_VOXEL_COUNT_EXP = 10u;

// ============================================================================
// Common Data Structures
// ============================================================================

public struct SVOLeaf {
    public uint voxelDataLow;   // Lower 32 bits of voxel data
    public uint voxelDataHigh;  // Upper 32 bits of voxel data

    // Get voxel value at specific bit offset
    public bool getVoxel(uint bitOffset) {
        uint wordIndex = bitOffset >> 5;  // 0 or 1
        uint bitInWord = bitOffset & 31;
        uint mask = 1u << bitInWord;
        uint voxelData = select(wordIndex == 0, this.voxelDataLow, this.voxelDataHigh);
        return (voxelData & mask) != 0;
    }

    // Set voxel value at specific bit offset (for compute shader)
    [mutating]
    public void setVoxelBit(uint bitOffset, bool value) {
        uint wordIndex = bitOffset >> 5;
        uint bitInWord = bitOffset & 31;
        uint mask = 1u << bitInWord;

        if (wordIndex == 0) {
            this.voxelDataLow = (this.voxelDataLow & ~mask) | (uint(value) << bitInWord);
        } else {
            this.voxelDataHigh = (this.voxelDataHigh & ~mask) | (uint(value) << bitInWord);
        }
    }

    // Check if leaf is empty
    public bool isEmpty() {
        return this.voxelDataLow == 0u && this.voxelDataHigh == 0u;
    }

    // Check if leaf is full
    public bool isFull() {
        return this.voxelDataLow == 0xFFFFFFFFu && this.voxelDataHigh == 0xFFFFFFFFu;
    }

    // Clear all voxels
    [mutating]
    public void clear() {
        this.voxelDataLow = 0u;
        this.voxelDataHigh = 0u;
    }

    // Fill all voxels
    [mutating]
    public void fill() {
        this.voxelDataLow = 0xFFFFFFFFu;
        this.voxelDataHigh = 0xFFFFFFFFu;
    }
};

// Leaf Edit Command structure - must match Rust LeafEditCommand
public struct LeafEditCommand {
    public uint leaf_x;
    public uint leaf_y;
    public uint leaf_z;
    public uint voxel_data_low;
    public uint voxel_data_high;
};

public struct SVOCounters {
    public uint nodeCount;
    public uint leafCount;
    public uint freeNodeCount;
    public uint freeLeafCount;
};

public struct SVO {}
