#include <algo/svt/svo_forwarding.hpp>

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

struct Options {
    uint32_t world_size = 0;
    std::vector<std::string> voxel_files;
};

struct Stats {
    std::string dataset;
    uint32_t resolution = 0;
    uint32_t world_size = 0;
    uint32_t max_depth = 0;
    std::size_t total_voxels = 0;
    std::size_t nodes = 0;
    std::size_t forwarding_indices = 0;
    uint32_t pages = 0;
    std::size_t memory_bytes = 0;
};

VoxelData load_voxel_file(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open: " + path);
    }

    VoxelData data;
    data.name = std::filesystem::path(path).filename().string();

    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (line.empty() || line[0] == '#') {
            continue;
        }

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

        uint32_t x = 0;
        uint32_t y = 0;
        uint32_t z = 0;
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

uint32_t choose_no_leaf_depth(uint32_t requested_world_size) {
    uint32_t depth = 0;
    uint64_t world_size = 2;
    while (world_size < requested_world_size) {
        world_size <<= 1;
        ++depth;
    }
    return depth;
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

Stats measure_forwarding_indices(const VoxelData& data, uint32_t requested_world_size) {
    const uint32_t max_depth = choose_no_leaf_depth(requested_world_size);
    algo::svt::SVOForwarding svo(max_depth);
    for (const auto& voxel : data.voxels) {
        if (in_bounds(voxel, svo.world_voxel_count())) {
            svo.set_voxel(voxel.x, voxel.y, voxel.z, true);
        }
    }

    return {
        .dataset = data.name,
        .resolution = data.resolution,
        .world_size = requested_world_size,
        .max_depth = max_depth,
        .total_voxels = data.voxels.size(),
        .nodes = svo.node_count(),
        .forwarding_indices = svo.forwarded_cell_count(),
        .pages = svo.page_count(),
        .memory_bytes = svo.memory_usage_bytes(),
    };
}

void print_usage(const char* argv0) {
    std::cerr << "usage: " << argv0
              << " [--world-size N] <voxel-file> [<voxel-file> ...]\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        if (options.voxel_files.empty()) {
            print_usage(argv[0]);
            return 1;
        }

        std::cout
            << "dataset\tresolution\tworld_size\tmax_depth\ttotal_voxels\tnodes"
            << "\tforwarding_indices\tpages\tmemory_bytes\n";

        for (const auto& path : options.voxel_files) {
            const VoxelData data = load_voxel_file(path);
            const uint32_t world_size =
                options.world_size == 0 ? data.resolution : options.world_size;
            const Stats stats = measure_forwarding_indices(data, world_size);
            std::cout << stats.dataset << '\t'
                      << stats.resolution << '\t'
                      << stats.world_size << '\t'
                      << stats.max_depth << '\t'
                      << stats.total_voxels << '\t'
                      << stats.nodes << '\t'
                      << stats.forwarding_indices << '\t'
                      << stats.pages << '\t'
                      << stats.memory_bytes << '\n';
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
