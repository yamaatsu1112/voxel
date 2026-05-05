# tools

Small utilities for inspecting voxel data structures and preparing input data.

## Build

```bash
cmake -S . -B build
cmake --build build
```

## Utilities

- `svt_node_count`: count nodes for SVT-related inputs
- `svo_forwarding_index_count`: inspect forwarding-index SVO layouts
- `svo_forwarding_packed_index_count`: inspect packed forwarding-index SVO layouts
- `voxelize.py`: convert simple input geometry into voxel data
