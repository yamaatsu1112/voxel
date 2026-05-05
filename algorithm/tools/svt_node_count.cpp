#include <algo/svt/hash_dag.hpp>
#include <algo/svt/svo.hpp>
#include <algo/svt/svo64.hpp>
#include <algo/svt/svo64_grouped.hpp>
#include <algo/svt/svo64_grouped_no_leaf.hpp>
#include <algo/svt/svdag.hpp>
#include <algo/svt/svo_grouped.hpp>
#include <algo/svt/svo_grouped_no_leaf.hpp>
#include <algo/svt/svo_hamming.hpp>
#include <algo/svt/svo_no_leaf.hpp>
#include <algo/svt/svo_forwarding.hpp>
#include <algo/svt/svo_forwarding_packed.hpp>
#include <algo/svt/svo_forwarding_packed_leaf.hpp>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Voxel {
    uint32_t x, y, z;
};

struct VoxelData {
    std::string name;
    uint32_t resolution = 0;
    std::vector<Voxel> voxels;
};

VoxelData load_voxel_file(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("failed to open: " + path);

    VoxelData data;
    data.name = std::filesystem::path(path).filename().string();

    std::string line;
    std::size_t line_number = 0;

    while (std::getline(input, line)) {
        ++line_number;
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);

        if (data.resolution == 0) {
            std::string keyword;
            iss >> keyword;
            if (keyword == "resolution") {
                iss >> data.resolution;
                if (!iss || data.resolution == 0) {
                    throw std::runtime_error(
                        path + ":" + std::to_string(line_number) +
                        ": invalid resolution");
                }
                continue;
            }
            throw std::runtime_error(
                path + ":" + std::to_string(line_number) +
                ": expected 'resolution N' header");
        }

        uint32_t x = 0, y = 0, z = 0;
        iss >> x >> y >> z;
        if (!iss) {
            throw std::runtime_error(
                path + ":" + std::to_string(line_number) +
                ": failed to parse voxel coordinates");
        }
        data.voxels.push_back({x, y, z});
    }

    if (data.resolution == 0) {
        throw std::runtime_error(path + ": missing resolution header");
    }
    return data;
}

struct Stats {
    std::string impl;
    std::size_t nodes;
    std::size_t leaves;
    std::size_t node_storage_bytes;
};

struct Options {
    uint32_t world_size = 0;
    std::vector<std::string> voxel_files;
};

bool in_bounds(const Voxel& v, uint32_t world_size) {
    return v.x < world_size && v.y < world_size && v.z < world_size;
}

uint32_t parse_world_size(std::string_view value) {
    std::size_t parsed = 0;
    const auto world_size = std::stoul(std::string(value), &parsed);
    if (parsed != value.size() || world_size == 0) {
        throw std::runtime_error("invalid world_size: " + std::string(value));
    }
    return static_cast<uint32_t>(world_size);
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--world-size") {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for --world-size");
            }
            options.world_size = parse_world_size(argv[++i]);
            continue;
        }
        options.voxel_files.emplace_back(argv[i]);
    }
    return options;
}

Stats measure_svo(const VoxelData& data, uint32_t requested_world_size) {
    const uint32_t max_depth =
        algo::svt::SVO::max_depth_for_world_size(requested_world_size);
    algo::svt::SVO svo(max_depth);
    for (const auto& v : data.voxels) {
        if (in_bounds(v, svo.world_voxel_count())) {
            svo.set_voxel(v.x, v.y, v.z, true);
        }
    }
    return {"svo", svo.node_count(), svo.leaf_count(),
            svo.node_storage_bytes()};
}

Stats measure_hash_dag(const VoxelData& data, uint32_t requested_world_size) {
    const uint32_t depth =
        algo::svt::HashDAG::depth_for_world_size(requested_world_size);
    algo::svt::HashDAG dag(depth);
    for (const auto& v : data.voxels) {
        if (in_bounds(v, dag.world_voxel_count())) {
            dag.set_voxel(v.x, v.y, v.z, true);
        }
    }
    dag.collect_garbage();
    return {"hash_dag", dag.node_count(), dag.leaf_count(),
            dag.allocated_word_count() * sizeof(uint32_t)};
}

Stats measure_svdag(const VoxelData& data, uint32_t requested_world_size) {
    const uint32_t max_level =
        algo::svt::SVDAGBuilder::max_level_for_world_size(requested_world_size);
    algo::svt::SVDAGBuilder builder(max_level);
    const uint32_t world_size = algo::svt::SVDAG::world_voxel_count(max_level);
    for (const auto& v : data.voxels) {
        if (in_bounds(v, world_size)) {
            builder.set_voxel(v.x, v.y, v.z, true);
        }
    }
    const auto dag = builder.build();
    return {"svdag", dag.node_count(), dag.leaf_count(),
            dag.node_storage_bytes()};
}

Stats measure_labeled_svdag(const VoxelData& data,
                            uint32_t requested_world_size) {
    const uint32_t max_level =
        algo::svt::SVDAGBuilder::max_level_for_world_size(requested_world_size);
    algo::svt::SVDAGBuilder builder(max_level);
    const uint32_t world_size =
        algo::svt::LabeledSVDAG::world_voxel_count(max_level);
    for (const auto& v : data.voxels) {
        if (in_bounds(v, world_size)) {
            builder.set_voxel(v.x, v.y, v.z, true);
        }
    }
    const auto dag = builder.build_labeled();
    return {"svdag_labeled", dag.node_count(), dag.leaf_count(),
            dag.node_storage_bytes()};
}

Stats measure_svo_no_leaf(const VoxelData& data, uint32_t requested_world_size) {
    const uint32_t max_depth =
        algo::svt::SVONoLeaf::max_depth_for_world_size(requested_world_size);
    algo::svt::SVONoLeaf svo(max_depth);
    for (const auto& v : data.voxels) {
        if (in_bounds(v, svo.world_voxel_count())) {
            svo.set_voxel(v.x, v.y, v.z, true);
        }
    }
    return {"svo_no_leaf", svo.node_count(), svo.leaf_count(),
            svo.node_storage_bytes()};
}

Stats measure_forwarding_svo(const VoxelData& data, uint32_t requested_world_size) {
    const uint32_t max_depth =
        algo::svt::SVOForwarding::max_depth_for_world_size(requested_world_size);
    algo::svt::SVOForwarding svo(max_depth);
    for (const auto& v : data.voxels) {
        if (in_bounds(v, svo.world_voxel_count())) {
            svo.set_voxel(v.x, v.y, v.z, true);
        }
    }
    return {"svo_forwarding", svo.node_count(), svo.leaf_count(),
            svo.node_storage_bytes()};
}

Stats measure_forwarding_packed_svo(const VoxelData& data,
                                    uint32_t requested_world_size) {
    const uint32_t max_depth =
        algo::svt::SVOForwardingPacked::max_depth_for_world_size(
            requested_world_size);
    algo::svt::SVOForwardingPacked svo(max_depth);
    for (const auto& v : data.voxels) {
        if (in_bounds(v, svo.world_voxel_count())) {
            svo.set_voxel(v.x, v.y, v.z, true);
        }
    }
    return {"svo_forwarding_packed", svo.node_count(), svo.leaf_count(),
            svo.node_storage_bytes()};
}

Stats measure_forwarding_packed_leaf_svo(const VoxelData& data,
                                         uint32_t requested_world_size) {
    const uint32_t max_depth =
        algo::svt::SVOForwardingPackedLeaf::max_depth_for_world_size(
            requested_world_size);
    algo::svt::SVOForwardingPackedLeaf svo(max_depth);
    for (const auto& v : data.voxels) {
        if (in_bounds(v, svo.world_voxel_count())) {
            svo.set_voxel(v.x, v.y, v.z, true);
        }
    }
    return {"svo_forwarding_packed_leaf", svo.node_count(), svo.leaf_count(),
            svo.node_storage_bytes()};
}

Stats measure_hamming_svo(const VoxelData& data, uint32_t requested_world_size) {
    const uint32_t max_depth =
        algo::svt::HammingSVO::max_depth_for_world_size(requested_world_size);
    algo::svt::HammingSVO svo(max_depth);
    for (const auto& v : data.voxels) {
        if (in_bounds(v, svo.world_voxel_count())) {
            svo.set_voxel(v.x, v.y, v.z, true);
        }
    }
    return {"svo_hamming", svo.node_count(), svo.leaf_count(),
            svo.node_storage_bytes()};
}

Stats measure_grouped_svo(const VoxelData& data, uint32_t requested_world_size) {
    const uint32_t max_depth =
        algo::svt::SVOGrouped::max_depth_for_world_size(requested_world_size);
    algo::svt::SVOGrouped svo(max_depth);
    for (const auto& v : data.voxels) {
        if (in_bounds(v, svo.world_voxel_count())) {
            svo.set_voxel(v.x, v.y, v.z, true);
        }
    }
    return {"svo_grouped", svo.node_count(), svo.leaf_count(),
            svo.node_storage_bytes()};
}

Stats measure_grouped_no_leaf_svo(const VoxelData& data,
                                  uint32_t requested_world_size) {
    const uint32_t max_depth =
        algo::svt::SVOGroupedNoLeaf::max_depth_for_world_size(
            requested_world_size);
    algo::svt::SVOGroupedNoLeaf svo(max_depth);
    for (const auto& v : data.voxels) {
        if (in_bounds(v, svo.world_voxel_count())) {
            svo.set_voxel(v.x, v.y, v.z, true);
        }
    }
    return {"svo_grouped_no_leaf", svo.node_count(), svo.leaf_count(),
            svo.node_storage_bytes()};
}

Stats measure_64_svo(const VoxelData& data, uint32_t requested_world_size) {
    const uint32_t max_depth =
        algo::svt::SVO64::max_depth_for_world_size(requested_world_size);
    algo::svt::SVO64 svo(max_depth);
    for (const auto& v : data.voxels) {
        if (in_bounds(v, svo.world_voxel_count())) {
            svo.set_voxel(v.x, v.y, v.z, true);
        }
    }
    return {"svo64", svo.node_count(), svo.leaf_count(),
            svo.node_storage_bytes()};
}

Stats measure_64_grouped_svo(const VoxelData& data,
                             uint32_t requested_world_size) {
    const uint32_t max_depth =
        algo::svt::SVO64Grouped::max_depth_for_world_size(
            requested_world_size);
    algo::svt::SVO64Grouped svo(max_depth);
    for (const auto& v : data.voxels) {
        if (in_bounds(v, svo.world_voxel_count())) {
            svo.set_voxel(v.x, v.y, v.z, true);
        }
    }
    return {"svo64_grouped", svo.node_count(), svo.leaf_count(),
            svo.node_storage_bytes()};
}

Stats measure_64_grouped_no_leaf_svo(const VoxelData& data,
                                     uint32_t requested_world_size) {
    const uint32_t max_depth =
        algo::svt::SVO64GroupedNoLeaf::max_depth_for_world_size(
            requested_world_size);
    algo::svt::SVO64GroupedNoLeaf svo(max_depth);
    for (const auto& v : data.voxels) {
        if (in_bounds(v, svo.world_voxel_count())) {
            svo.set_voxel(v.x, v.y, v.z, true);
        }
    }
    return {"svo64_grouped_no_leaf", svo.node_count(), svo.leaf_count(),
            svo.node_storage_bytes()};
}

void print_header() {
    std::cout
        << "dataset\tresolution\timpl\tnodes\tleaves\tnode_storage_bytes\ttotal_voxels\n";
}

void print_row(const VoxelData& data, const Stats& stats) {
    std::cout << data.name << '\t' << data.resolution << '\t' << stats.impl
              << '\t' << stats.nodes << '\t' << stats.leaves << '\t'
              << stats.node_storage_bytes << '\t' << data.voxels.size() << '\n';
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_options(argc, argv);
        if (options.voxel_files.empty()) {
            std::cerr << "Usage: svt_node_count [--world-size N] <voxel_file>...\n"
                      << "  Input files are generated by voxelize.py\n";
            return 1;
        }

        print_header();
        for (const auto& voxel_file : options.voxel_files) {
            const auto data = load_voxel_file(voxel_file);
            const uint32_t requested_world_size =
                options.world_size == 0 ? data.resolution : options.world_size;

            const auto svo_stats = measure_svo(data, requested_world_size);
            print_row(data, svo_stats);

            const auto hash_dag_stats =
                measure_hash_dag(data, requested_world_size);
            print_row(data, hash_dag_stats);

            const auto svdag_stats =
                measure_svdag(data, requested_world_size);
            print_row(data, svdag_stats);

            const auto labeled_svdag_stats =
                measure_labeled_svdag(data, requested_world_size);
            print_row(data, labeled_svdag_stats);

            const auto svo_no_leaf_stats =
                measure_svo_no_leaf(data, requested_world_size);
            print_row(data, svo_no_leaf_stats);

            const auto forwarding_stats =
                measure_forwarding_svo(data, requested_world_size);
            print_row(data, forwarding_stats);

            const auto forwarding_packed_stats =
                measure_forwarding_packed_svo(data, requested_world_size);
            print_row(data, forwarding_packed_stats);

            const auto forwarding_packed_leaf_stats =
                measure_forwarding_packed_leaf_svo(data, requested_world_size);
            print_row(data, forwarding_packed_leaf_stats);

            const auto hamming_stats =
                measure_hamming_svo(data, requested_world_size);
            print_row(data, hamming_stats);

            const auto grouped_stats =
                measure_grouped_svo(data, requested_world_size);
            print_row(data, grouped_stats);

            const auto grouped_no_leaf_stats =
                measure_grouped_no_leaf_svo(data, requested_world_size);
            print_row(data, grouped_no_leaf_stats);

            const auto svo64_stats =
                measure_64_svo(data, requested_world_size);
            print_row(data, svo64_stats);

            const auto grouped64_stats =
                measure_64_grouped_svo(data, requested_world_size);
            print_row(data, grouped64_stats);

            const auto grouped64_no_leaf_stats =
                measure_64_grouped_no_leaf_svo(data, requested_world_size);
            print_row(data, grouped64_no_leaf_stats);
        }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "svt_node_count: " << ex.what() << '\n';
        return 1;
    }
}
