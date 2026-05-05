#pragma once

#include <algo/svt/cuda/config.cuh>

#include <cstdint>

#ifdef __CUDACC__
#define ALGO_SVT_CUDA_HD __host__ __device__
#else
#define ALGO_SVT_CUDA_HD
#endif

namespace algo::svt::cuda {

struct TerminalNodeInput {
    std::uint32_t depth;
    std::uint32_t prefix;
    std::int32_t worldOffsetX;
    std::int32_t worldOffsetY;
    std::int32_t worldOffsetZ;
};

struct TerminalLeafInput {
    std::uint32_t leafPrefix;
    std::uint64_t mask64;
    std::int32_t worldOffsetX;
    std::int32_t worldOffsetY;
    std::int32_t worldOffsetZ;
};

struct CellWriteRequest {
    std::uint32_t level;
    std::uint32_t prefix;
};

static_assert(kMaxDepth <= 10u,
              "terminal leaf prefixes store at most 10 packed 3-bit levels");
static_assert(kMaxDepth * kGroupSizeExp <= 30u,
              "terminal request prefixes reserve high bits for tags");

struct TerminalBrickMask {
    std::uint32_t leafPrefix;
    std::uint64_t mask64;
};

struct TerminalRequest {
    std::uint32_t taggedPrefix;
    union {
        std::uint32_t cellLevel;
        std::uint64_t brickMask64;
    };
};
static_assert(sizeof(TerminalRequest) == 16u,
              "terminal requests keep the kind tag in prefix spare bits");

inline constexpr std::uint32_t kTerminalRequestBrickTag = 1u << 31u;
inline constexpr std::uint32_t kTerminalRequestPrefixMask =
    (1u << (kMaxDepth * kGroupSizeExp)) - 1u;

ALGO_SVT_CUDA_HD inline bool
terminal_request_is_brick(const TerminalRequest& request) {
    return (request.taggedPrefix & kTerminalRequestBrickTag) != 0u;
}

ALGO_SVT_CUDA_HD inline bool
terminal_request_is_cell(const TerminalRequest& request) {
    return !terminal_request_is_brick(request);
}

ALGO_SVT_CUDA_HD inline std::uint32_t
terminal_request_prefix(const TerminalRequest& request) {
    return request.taggedPrefix & kTerminalRequestPrefixMask;
}

ALGO_SVT_CUDA_HD inline CellWriteRequest
terminal_request_cell(const TerminalRequest& request) {
    return {request.cellLevel, terminal_request_prefix(request)};
}

ALGO_SVT_CUDA_HD inline TerminalBrickMask
terminal_request_brick(const TerminalRequest& request) {
    return {terminal_request_prefix(request), request.brickMask64};
}

ALGO_SVT_CUDA_HD inline TerminalRequest
make_terminal_cell_request(CellWriteRequest cell) {
    TerminalRequest request{};
    request.taggedPrefix = cell.prefix & kTerminalRequestPrefixMask;
    request.cellLevel = cell.level;
    return request;
}

ALGO_SVT_CUDA_HD inline TerminalRequest
make_terminal_brick_request(TerminalBrickMask brick) {
    TerminalRequest request{};
    request.taggedPrefix =
        kTerminalRequestBrickTag |
        (brick.leafPrefix & kTerminalRequestPrefixMask);
    request.brickMask64 = brick.mask64;
    return request;
}

} // namespace algo::svt::cuda

#undef ALGO_SVT_CUDA_HD
