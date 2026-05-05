#include "cuda_test_utils.cuh"

#include <algo/cuda/scan/scan.cuh>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

namespace {

using BlellochGlobal =
    algo::cuda::scan::Global<algo::cuda::scan::BlellochGlobal<256>>;
using BlellochShared =
    algo::cuda::scan::BlockBased<
        algo::cuda::scan::ScanThenPropagate<
            algo::cuda::scan::BlellochBlock<256, 2>>>;
using BlellochSharedReduceThenScan =
    algo::cuda::scan::BlockBased<
        algo::cuda::scan::ReduceThenScan<
            algo::cuda::scan::BlellochBlock<256, 2>>>;
using BlellochSharedDecoupledLookback =
    algo::cuda::scan::BlockBased<
        algo::cuda::scan::DecoupledLookback<
            algo::cuda::scan::BlellochBlock<256, 2>>>;
using BlellochSharedPadded =
    algo::cuda::scan::BlockBased<
        algo::cuda::scan::ScanThenPropagate<
            algo::cuda::scan::BlellochBlock<
                256, 2, algo::cuda::scan::PaddedSharedLayout>>>;
using BlellochSharedPaddedReduceThenScan =
    algo::cuda::scan::BlockBased<
        algo::cuda::scan::ReduceThenScan<
            algo::cuda::scan::BlellochBlock<
                256, 2, algo::cuda::scan::PaddedSharedLayout>>>;
using BlellochSharedPaddedDecoupledLookback =
    algo::cuda::scan::BlockBased<
        algo::cuda::scan::DecoupledLookback<
            algo::cuda::scan::BlellochBlock<
                256, 2, algo::cuda::scan::PaddedSharedLayout>>>;
using HillisSteeleGlobal =
    algo::cuda::scan::Global<algo::cuda::scan::HillisSteeleGlobal<256>>;
using HillisSteeleShared =
    algo::cuda::scan::BlockBased<
        algo::cuda::scan::ScanThenPropagate<
            algo::cuda::scan::HillisSteeleBlock<256, 2>>>;
using HillisSteeleSharedReduceThenScan =
    algo::cuda::scan::BlockBased<
        algo::cuda::scan::ReduceThenScan<
            algo::cuda::scan::HillisSteeleBlock<256, 2>>>;
using HillisSteeleSharedDecoupledLookback =
    algo::cuda::scan::BlockBased<
        algo::cuda::scan::DecoupledLookback<
            algo::cuda::scan::HillisSteeleBlock<256, 2>>>;
using WarpShuffle256x1 = algo::cuda::scan::BlockBased<
    algo::cuda::scan::ScanThenPropagate<
        algo::cuda::scan::WarpShuffleBlock<256, 1>>>;
using WarpShuffle256x1ReduceThenScan = algo::cuda::scan::BlockBased<
    algo::cuda::scan::ReduceThenScan<
        algo::cuda::scan::WarpShuffleBlock<256, 1>>>;
using WarpShuffle256x4 = algo::cuda::scan::BlockBased<
    algo::cuda::scan::ScanThenPropagate<
        algo::cuda::scan::WarpShuffleBlock<256, 4>>>;
using WarpShuffle256x4ReduceThenScan = algo::cuda::scan::BlockBased<
    algo::cuda::scan::ReduceThenScan<
        algo::cuda::scan::WarpShuffleBlock<256, 4>>>;
using WarpShuffle256x4DecoupledLookback = algo::cuda::scan::BlockBased<
    algo::cuda::scan::DecoupledLookback<
        algo::cuda::scan::WarpShuffleBlock<256, 4>>>;

using cuda_test::fill_device;
using cuda_test::has_cuda_device;
using cuda_test::managed_alloc;
using cuda_test::read_device;
using cuda_test::sync_cuda;

template <class T>
std::vector<T> cpu_inclusive_sum(const std::vector<T>& input) {
    std::vector<T> out(input.size());
    T acc = 0;
    for (std::size_t i = 0; i < input.size(); ++i) {
        acc += input[i];
        out[i] = acc;
    }
    return out;
}

template <class T>
std::vector<T> cpu_exclusive_sum(const std::vector<T>& input) {
    std::vector<T> out(input.size());
    T acc = 0;
    for (std::size_t i = 0; i < input.size(); ++i) {
        out[i] = acc;
        acc += input[i];
    }
    return out;
}

template <class T>
std::vector<T> cpu_exclusive_max(const std::vector<T>& input) {
    std::vector<T> out(input.size());
    T acc = 0;
    for (std::size_t i = 0; i < input.size(); ++i) {
        out[i] = acc;
        acc = std::max(acc, input[i]);
    }
    return out;
}

struct FusedScanInput {
    const std::uint32_t* values;
    std::uint32_t* transformed;
    std::uint32_t* prefixes;
    std::uint32_t* post_invocations;
};

struct TransformDoubleAddOne {
    __device__ static std::uint32_t run(std::uint32_t index,
                                        const FusedScanInput& input) {
        return input.values[index] * 2u + 1u;
    }
};

struct InclusiveStorePostScan {
    __device__ static void run(std::uint32_t index, const FusedScanInput& input,
                               std::uint32_t transformed_value,
                               std::uint32_t prefix) {
        input.transformed[index] = transformed_value;
        input.prefixes[index] = prefix;
        atomicAdd(&input.post_invocations[index], 1u);
    }
};

struct ExclusiveStorePostScan {
    __device__ static void run(std::uint32_t index, const FusedScanInput& input,
                               std::uint32_t transformed_value,
                               std::uint32_t prefix) {
        input.transformed[index] = transformed_value;
        input.prefixes[index] = prefix;
        atomicAdd(&input.post_invocations[index], 1u);
    }
};

template <class Config>
void expect_inclusive_scan_matches_reference(
    const std::vector<std::uint32_t>& input) {
    auto* data = managed_alloc<std::uint32_t>(input.size());
    ASSERT_NE(data, nullptr);
    const std::size_t workspace_size =
        algo::cuda::scan::required_workspace_size<Config, std::uint32_t>(
            static_cast<std::uint32_t>(input.size()));
    auto* workspace =
        managed_alloc<std::byte>(workspace_size == 0 ? 1 : workspace_size);
    ASSERT_NE(workspace, nullptr);

    fill_device(data, input);
    ASSERT_EQ((algo::cuda::scan::inclusive_sum<Config>(
                  data, static_cast<std::uint32_t>(input.size()), workspace,
                  workspace_size)),
              cudaSuccess);
    sync_cuda();
    EXPECT_EQ(read_device(data, input.size()), cpu_inclusive_sum(input));

    cudaFree(workspace);
    cudaFree(data);
}

template <class Config>
void expect_exclusive_scan_matches_reference(
    const std::vector<std::uint32_t>& input) {
    auto* data = managed_alloc<std::uint32_t>(input.size());
    ASSERT_NE(data, nullptr);
    const std::size_t workspace_size =
        algo::cuda::scan::required_workspace_size<Config, std::uint32_t>(
            static_cast<std::uint32_t>(input.size()));
    auto* workspace =
        managed_alloc<std::byte>(workspace_size == 0 ? 1 : workspace_size);
    ASSERT_NE(workspace, nullptr);

    fill_device(data, input);
    ASSERT_EQ((algo::cuda::scan::exclusive_sum<Config>(
                  data, static_cast<std::uint32_t>(input.size()), workspace,
                  workspace_size)),
              cudaSuccess);
    sync_cuda();
    EXPECT_EQ(read_device(data, input.size()), cpu_exclusive_sum(input));

    cudaFree(workspace);
    cudaFree(data);
}

TEST(CudaScan, RejectsInsufficientWorkspace) {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    auto* global_data = managed_alloc<std::uint32_t>(8);
    ASSERT_NE(global_data, nullptr);
    auto* shared_data = managed_alloc<std::uint32_t>(1025);
    ASSERT_NE(shared_data, nullptr);
    auto* workspace = managed_alloc<std::byte>(4);
    ASSERT_NE(workspace, nullptr);

    EXPECT_EQ((algo::cuda::scan::inclusive_sum<BlellochGlobal>(
                  global_data, 8, workspace, 4)),
              cudaErrorInvalidValue);
    EXPECT_EQ((algo::cuda::scan::inclusive_sum<BlellochShared>(
                  shared_data, 1024, workspace, 4)),
              cudaErrorInvalidValue);
    EXPECT_EQ((algo::cuda::scan::inclusive_sum<BlellochSharedPadded>(
                  shared_data, 1024, workspace, 4)),
              cudaErrorInvalidValue);
    EXPECT_EQ((algo::cuda::scan::exclusive_sum<HillisSteeleGlobal>(
                  global_data, 8, workspace, 4)),
              cudaErrorInvalidValue);
    EXPECT_EQ((algo::cuda::scan::exclusive_sum<HillisSteeleShared>(
                  shared_data, 1024, workspace, 4)),
              cudaErrorInvalidValue);
    EXPECT_EQ((algo::cuda::scan::exclusive_sum<HillisSteeleSharedReduceThenScan>(
                  shared_data, 1024, workspace, 4)),
              cudaErrorInvalidValue);
    EXPECT_EQ((algo::cuda::scan::exclusive_sum<BlellochSharedDecoupledLookback>(
                  shared_data, 1024, workspace, 4)),
              cudaErrorInvalidValue);
    EXPECT_EQ((algo::cuda::scan::exclusive_sum<
                  BlellochSharedPaddedDecoupledLookback>(
                  shared_data, 1024, workspace, 4)),
              cudaErrorInvalidValue);
    EXPECT_EQ((algo::cuda::scan::exclusive_sum<
                  HillisSteeleSharedDecoupledLookback>(
                  shared_data, 1024, workspace, 4)),
              cudaErrorInvalidValue);
    EXPECT_EQ((algo::cuda::scan::exclusive_sum<WarpShuffle256x4>(
                  shared_data, 1025, workspace, 4)),
              cudaErrorInvalidValue);
    EXPECT_EQ((algo::cuda::scan::exclusive_sum<WarpShuffle256x4ReduceThenScan>(
                  shared_data, 1025, workspace, 4)),
              cudaErrorInvalidValue);
    EXPECT_EQ((algo::cuda::scan::exclusive_sum<WarpShuffle256x4DecoupledLookback>(
                  shared_data, 1025, workspace, 4)),
              cudaErrorInvalidValue);

    cudaFree(workspace);
    cudaFree(shared_data);
    cudaFree(global_data);
}

TEST(CudaScan, DefaultInclusiveMatchesReferenceOnFixedInput) {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    auto input = std::vector<std::uint32_t>{3, 1, 4, 1, 5, 9, 2, 6, 5};
    auto* data = managed_alloc<std::uint32_t>(input.size());
    ASSERT_NE(data, nullptr);
    const std::size_t workspace_size =
        algo::cuda::scan::required_workspace_size<std::uint32_t>(
            static_cast<std::uint32_t>(input.size()));
    auto* workspace =
        managed_alloc<std::byte>(workspace_size == 0 ? 1 : workspace_size);
    ASSERT_NE(workspace, nullptr);

    fill_device(data, input);
    ASSERT_EQ((algo::cuda::scan::inclusive_sum(
                  data, static_cast<std::uint32_t>(input.size()), workspace,
                  workspace_size)),
              cudaSuccess);
    sync_cuda();
    EXPECT_EQ(read_device(data, input.size()), cpu_inclusive_sum(input));

    cudaFree(workspace);
    cudaFree(data);
}

TEST(CudaScan, DefaultExclusiveMatchesReferenceOnFixedInput) {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    auto input = std::vector<std::uint32_t>{3, 1, 4, 1, 5, 9, 2, 6, 5};
    auto* data = managed_alloc<std::uint32_t>(input.size());
    ASSERT_NE(data, nullptr);
    const std::size_t workspace_size =
        algo::cuda::scan::required_workspace_size<std::uint32_t>(
            static_cast<std::uint32_t>(input.size()));
    auto* workspace =
        managed_alloc<std::byte>(workspace_size == 0 ? 1 : workspace_size);
    ASSERT_NE(workspace, nullptr);

    fill_device(data, input);
    ASSERT_EQ((algo::cuda::scan::exclusive_sum(
                  data, static_cast<std::uint32_t>(input.size()), workspace,
                  workspace_size)),
              cudaSuccess);
    sync_cuda();
    EXPECT_EQ(read_device(data, input.size()), cpu_exclusive_sum(input));

    cudaFree(workspace);
    cudaFree(data);
}

TEST(CudaScan, HillisSteeleInclusiveMatchesReferenceOnFixedInput) {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    expect_inclusive_scan_matches_reference<HillisSteeleGlobal>(
        {3, 1, 4, 1, 5, 9, 2, 6, 5});
}

TEST(CudaScan, HillisSteeleExclusiveMatchesReferenceOnFixedInput) {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    expect_exclusive_scan_matches_reference<HillisSteeleGlobal>(
        {3, 1, 4, 1, 5, 9, 2, 6, 5});
}

TEST(CudaScan, InclusiveAlgorithmsHandleRandomNonPowerOfTwoInput) {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    std::mt19937 rng(123);
    std::uniform_int_distribution<std::uint32_t> dist(0, 9);

    std::vector<std::uint32_t> input(257);
    for (auto& value : input) value = dist(rng);

    expect_inclusive_scan_matches_reference<BlellochGlobal>(input);
    expect_inclusive_scan_matches_reference<BlellochShared>(input);
    expect_inclusive_scan_matches_reference<BlellochSharedReduceThenScan>(input);
    expect_inclusive_scan_matches_reference<BlellochSharedDecoupledLookback>(
        input);
    expect_inclusive_scan_matches_reference<BlellochSharedPadded>(input);
    expect_inclusive_scan_matches_reference<BlellochSharedPaddedReduceThenScan>(
        input);
    expect_inclusive_scan_matches_reference<
        BlellochSharedPaddedDecoupledLookback>(input);
    expect_inclusive_scan_matches_reference<HillisSteeleGlobal>(input);
    expect_inclusive_scan_matches_reference<HillisSteeleShared>(input);
    expect_inclusive_scan_matches_reference<HillisSteeleSharedReduceThenScan>(
        input);
    expect_inclusive_scan_matches_reference<HillisSteeleSharedDecoupledLookback>(
        input);
    expect_inclusive_scan_matches_reference<WarpShuffle256x1>(input);
    expect_inclusive_scan_matches_reference<WarpShuffle256x1ReduceThenScan>(
        input);
    expect_inclusive_scan_matches_reference<WarpShuffle256x4>(input);
    expect_inclusive_scan_matches_reference<WarpShuffle256x4ReduceThenScan>(
        input);
    expect_inclusive_scan_matches_reference<WarpShuffle256x4DecoupledLookback>(
        input);
}

TEST(CudaScan, ExclusiveAlgorithmsHandleRandomNonPowerOfTwoInput) {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    std::mt19937 rng(123);
    std::uniform_int_distribution<std::uint32_t> dist(0, 9);

    std::vector<std::uint32_t> input(257);
    for (auto& value : input) value = dist(rng);

    expect_exclusive_scan_matches_reference<BlellochGlobal>(input);
    expect_exclusive_scan_matches_reference<BlellochShared>(input);
    expect_exclusive_scan_matches_reference<BlellochSharedReduceThenScan>(input);
    expect_exclusive_scan_matches_reference<BlellochSharedDecoupledLookback>(
        input);
    expect_exclusive_scan_matches_reference<BlellochSharedPadded>(input);
    expect_exclusive_scan_matches_reference<BlellochSharedPaddedReduceThenScan>(
        input);
    expect_exclusive_scan_matches_reference<
        BlellochSharedPaddedDecoupledLookback>(input);
    expect_exclusive_scan_matches_reference<HillisSteeleGlobal>(input);
    expect_exclusive_scan_matches_reference<HillisSteeleShared>(input);
    expect_exclusive_scan_matches_reference<HillisSteeleSharedReduceThenScan>(
        input);
    expect_exclusive_scan_matches_reference<HillisSteeleSharedDecoupledLookback>(
        input);
    expect_exclusive_scan_matches_reference<WarpShuffle256x1>(input);
    expect_exclusive_scan_matches_reference<WarpShuffle256x1ReduceThenScan>(
        input);
    expect_exclusive_scan_matches_reference<WarpShuffle256x4>(input);
    expect_exclusive_scan_matches_reference<WarpShuffle256x4ReduceThenScan>(
        input);
    expect_exclusive_scan_matches_reference<WarpShuffle256x4DecoupledLookback>(
        input);
}

TEST(CudaScan, SharedAlgorithmsHandleMultiBlockInput) {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    std::mt19937 rng(456);
    std::uniform_int_distribution<std::uint32_t> dist(0, 9);

    std::vector<std::uint32_t> input(4097);
    for (auto& value : input) value = dist(rng);

    expect_inclusive_scan_matches_reference<BlellochShared>(input);
    expect_inclusive_scan_matches_reference<BlellochSharedReduceThenScan>(input);
    expect_inclusive_scan_matches_reference<BlellochSharedDecoupledLookback>(
        input);
    expect_inclusive_scan_matches_reference<BlellochSharedPadded>(input);
    expect_inclusive_scan_matches_reference<BlellochSharedPaddedReduceThenScan>(
        input);
    expect_inclusive_scan_matches_reference<
        BlellochSharedPaddedDecoupledLookback>(input);
    expect_inclusive_scan_matches_reference<HillisSteeleShared>(input);
    expect_inclusive_scan_matches_reference<HillisSteeleSharedReduceThenScan>(
        input);
    expect_inclusive_scan_matches_reference<HillisSteeleSharedDecoupledLookback>(
        input);
    expect_inclusive_scan_matches_reference<WarpShuffle256x1>(input);
    expect_inclusive_scan_matches_reference<WarpShuffle256x1ReduceThenScan>(
        input);
    expect_inclusive_scan_matches_reference<WarpShuffle256x4>(input);
    expect_inclusive_scan_matches_reference<WarpShuffle256x4ReduceThenScan>(
        input);
    expect_inclusive_scan_matches_reference<WarpShuffle256x4DecoupledLookback>(
        input);
    expect_exclusive_scan_matches_reference<BlellochShared>(input);
    expect_exclusive_scan_matches_reference<BlellochSharedReduceThenScan>(input);
    expect_exclusive_scan_matches_reference<BlellochSharedDecoupledLookback>(
        input);
    expect_exclusive_scan_matches_reference<BlellochSharedPadded>(input);
    expect_exclusive_scan_matches_reference<BlellochSharedPaddedReduceThenScan>(
        input);
    expect_exclusive_scan_matches_reference<
        BlellochSharedPaddedDecoupledLookback>(input);
    expect_exclusive_scan_matches_reference<HillisSteeleShared>(input);
    expect_exclusive_scan_matches_reference<HillisSteeleSharedReduceThenScan>(
        input);
    expect_exclusive_scan_matches_reference<HillisSteeleSharedDecoupledLookback>(
        input);
    expect_exclusive_scan_matches_reference<WarpShuffle256x1>(input);
    expect_exclusive_scan_matches_reference<WarpShuffle256x1ReduceThenScan>(
        input);
    expect_exclusive_scan_matches_reference<WarpShuffle256x4>(input);
    expect_exclusive_scan_matches_reference<WarpShuffle256x4ReduceThenScan>(
        input);
    expect_exclusive_scan_matches_reference<WarpShuffle256x4DecoupledLookback>(
        input);
}

TEST(CudaScan, ZeroAndSingleElementInputsAreValid) {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    auto* one = managed_alloc<std::uint32_t>(1);
    auto* workspace = managed_alloc<std::byte>(1);
    ASSERT_NE(one, nullptr);
    ASSERT_NE(workspace, nullptr);

    one[0] = 7;
    ASSERT_EQ((algo::cuda::scan::inclusive_sum<BlellochGlobal>(
                  one, 1, workspace, 1)),
              cudaSuccess);
    sync_cuda();
    EXPECT_EQ(one[0], 7u);

    one[0] = 7;
    ASSERT_EQ((algo::cuda::scan::inclusive_sum<BlellochShared>(
                  one, 1, workspace, 1)),
              cudaSuccess);
    sync_cuda();
    EXPECT_EQ(one[0], 7u);

    one[0] = 7;
    ASSERT_EQ((algo::cuda::scan::inclusive_sum<BlellochSharedPadded>(
                  one, 1, workspace, 1)),
              cudaSuccess);
    sync_cuda();
    EXPECT_EQ(one[0], 7u);

    one[0] = 7;
    ASSERT_EQ((algo::cuda::scan::exclusive_sum<HillisSteeleGlobal>(
                  one, 1, workspace, 1)),
              cudaSuccess);
    sync_cuda();
    EXPECT_EQ(one[0], 0u);

    one[0] = 7;
    ASSERT_EQ((algo::cuda::scan::exclusive_sum<HillisSteeleShared>(
                  one, 1, workspace, 1)),
              cudaSuccess);
    sync_cuda();
    EXPECT_EQ(one[0], 0u);

    one[0] = 7;
    ASSERT_EQ((algo::cuda::scan::exclusive_sum<BlellochShared>(
                  one, 1, workspace, 1)),
              cudaSuccess);
    sync_cuda();
    EXPECT_EQ(one[0], 0u);

    one[0] = 7;
    ASSERT_EQ((algo::cuda::scan::exclusive_sum<BlellochSharedPadded>(
                  one, 1, workspace, 1)),
              cudaSuccess);
    sync_cuda();
    EXPECT_EQ(one[0], 0u);

    one[0] = 7;
    ASSERT_EQ((algo::cuda::scan::inclusive_sum<WarpShuffle256x4>(
                  one, 1, workspace, 1)),
              cudaSuccess);
    sync_cuda();
    EXPECT_EQ(one[0], 7u);

    one[0] = 7;
    ASSERT_EQ((algo::cuda::scan::exclusive_sum<WarpShuffle256x4>(
                  one, 1, workspace, 1)),
              cudaSuccess);
    sync_cuda();
    EXPECT_EQ(one[0], 0u);

    one[0] = 7;
    ASSERT_EQ((algo::cuda::scan::inclusive_sum<WarpShuffle256x4DecoupledLookback>(
                  one, 1, workspace, 1)),
              cudaSuccess);
    sync_cuda();
    EXPECT_EQ(one[0], 7u);

    one[0] = 7;
    ASSERT_EQ((algo::cuda::scan::exclusive_sum<WarpShuffle256x4DecoupledLookback>(
                  one, 1, workspace, 1)),
              cudaSuccess);
    sync_cuda();
    EXPECT_EQ(one[0], 0u);

    EXPECT_EQ((algo::cuda::scan::inclusive_sum<BlellochGlobal>(
                  static_cast<std::uint32_t*>(nullptr), 0, nullptr, 0)),
              cudaSuccess);

    cudaFree(workspace);
    cudaFree(one);
}

TEST(CudaScan, FusedExclusiveRejectsInsufficientWorkspace) {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    auto* values = managed_alloc<std::uint32_t>(1025);
    auto* transformed = managed_alloc<std::uint32_t>(1025);
    auto* prefixes = managed_alloc<std::uint32_t>(1025);
    auto* post_invocations = managed_alloc<std::uint32_t>(1025);
    auto* workspace = managed_alloc<std::byte>(4);
    ASSERT_NE(values, nullptr);
    ASSERT_NE(transformed, nullptr);
    ASSERT_NE(prefixes, nullptr);
    ASSERT_NE(post_invocations, nullptr);
    ASSERT_NE(workspace, nullptr);

    FusedScanInput input{values, transformed, prefixes, post_invocations};
    EXPECT_EQ(
        (algo::cuda::scan::exclusive_sum_fused<TransformDoubleAddOne,
                                               ExclusiveStorePostScan>(
            input, 1025, workspace, 4)),
        cudaErrorInvalidValue);

    cudaFree(workspace);
    cudaFree(post_invocations);
    cudaFree(prefixes);
    cudaFree(transformed);
    cudaFree(values);
}

TEST(CudaScan, FusedInclusiveMatchesReferenceOnMultiBlockInput) {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    std::mt19937 rng(789);
    std::uniform_int_distribution<std::uint32_t> dist(0, 9);
    std::vector<std::uint32_t> input_values(4097);
    for (auto& value : input_values) value = dist(rng);

    std::vector<std::uint32_t> transformed_reference(input_values.size());
    for (std::size_t i = 0; i < input_values.size(); ++i)
        transformed_reference[i] = input_values[i] * 2u + 1u;
    const auto prefix_reference = cpu_inclusive_sum(transformed_reference);

    auto* values = managed_alloc<std::uint32_t>(input_values.size());
    auto* transformed = managed_alloc<std::uint32_t>(input_values.size());
    auto* prefixes = managed_alloc<std::uint32_t>(input_values.size());
    auto* post_invocations = managed_alloc<std::uint32_t>(input_values.size());
    ASSERT_NE(values, nullptr);
    ASSERT_NE(transformed, nullptr);
    ASSERT_NE(prefixes, nullptr);
    ASSERT_NE(post_invocations, nullptr);

    fill_device(values, input_values);
    std::fill_n(transformed, input_values.size(), 0u);
    std::fill_n(prefixes, input_values.size(), 0u);
    std::fill_n(post_invocations, input_values.size(), 0u);

    const std::size_t workspace_size =
        algo::cuda::scan::required_workspace_size_fused<std::uint32_t>(
            static_cast<std::uint32_t>(input_values.size()));
    auto* workspace =
        managed_alloc<std::byte>(workspace_size == 0 ? 1 : workspace_size);
    ASSERT_NE(workspace, nullptr);

    FusedScanInput input{values, transformed, prefixes, post_invocations};
    ASSERT_EQ(
        (algo::cuda::scan::inclusive_sum_fused<TransformDoubleAddOne,
                                               InclusiveStorePostScan>(
            input, static_cast<std::uint32_t>(input_values.size()), workspace,
            workspace_size)),
        cudaSuccess);
    sync_cuda();

    EXPECT_EQ(read_device(transformed, input_values.size()), transformed_reference);
    EXPECT_EQ(read_device(prefixes, input_values.size()), prefix_reference);
    EXPECT_EQ(read_device(post_invocations, input_values.size()),
              std::vector<std::uint32_t>(input_values.size(), 1u));

    cudaFree(workspace);
    cudaFree(post_invocations);
    cudaFree(prefixes);
    cudaFree(transformed);
    cudaFree(values);
}

TEST(CudaScan, FusedExclusiveMatchesReferenceOnMultiBlockInput) {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    std::mt19937 rng(987);
    std::uniform_int_distribution<std::uint32_t> dist(0, 9);
    std::vector<std::uint32_t> input_values(4097);
    for (auto& value : input_values) value = dist(rng);

    std::vector<std::uint32_t> transformed_reference(input_values.size());
    for (std::size_t i = 0; i < input_values.size(); ++i)
        transformed_reference[i] = input_values[i] * 2u + 1u;
    const auto prefix_reference = cpu_exclusive_sum(transformed_reference);

    auto* values = managed_alloc<std::uint32_t>(input_values.size());
    auto* transformed = managed_alloc<std::uint32_t>(input_values.size());
    auto* prefixes = managed_alloc<std::uint32_t>(input_values.size());
    auto* post_invocations = managed_alloc<std::uint32_t>(input_values.size());
    ASSERT_NE(values, nullptr);
    ASSERT_NE(transformed, nullptr);
    ASSERT_NE(prefixes, nullptr);
    ASSERT_NE(post_invocations, nullptr);

    fill_device(values, input_values);
    std::fill_n(transformed, input_values.size(), 0u);
    std::fill_n(prefixes, input_values.size(), 0u);
    std::fill_n(post_invocations, input_values.size(), 0u);

    const std::size_t workspace_size =
        algo::cuda::scan::required_workspace_size_fused<std::uint32_t>(
            static_cast<std::uint32_t>(input_values.size()));
    auto* workspace =
        managed_alloc<std::byte>(workspace_size == 0 ? 1 : workspace_size);
    ASSERT_NE(workspace, nullptr);

    FusedScanInput input{values, transformed, prefixes, post_invocations};
    ASSERT_EQ(
        (algo::cuda::scan::exclusive_sum_fused<TransformDoubleAddOne,
                                               ExclusiveStorePostScan>(
            input, static_cast<std::uint32_t>(input_values.size()), workspace,
            workspace_size)),
        cudaSuccess);
    sync_cuda();

    EXPECT_EQ(read_device(transformed, input_values.size()), transformed_reference);
    EXPECT_EQ(read_device(prefixes, input_values.size()), prefix_reference);
    EXPECT_EQ(read_device(post_invocations, input_values.size()),
              std::vector<std::uint32_t>(input_values.size(), 1u));

    cudaFree(workspace);
    cudaFree(post_invocations);
    cudaFree(prefixes);
    cudaFree(transformed);
    cudaFree(values);
}

TEST(CudaScan, WarpShuffleHandlesUint64Input) {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    auto input = std::vector<std::uint64_t>{1, 3, 5, 7, 9, 11, 13, 15, 17,
                                            19, 21, 23, 25, 27, 29, 31, 33};
    auto* data = managed_alloc<std::uint64_t>(input.size());
    ASSERT_NE(data, nullptr);

    const std::size_t workspace_size =
        algo::cuda::scan::required_workspace_size<WarpShuffle256x4,
                                                  std::uint64_t>(
            static_cast<std::uint32_t>(input.size()));
    auto* workspace =
        managed_alloc<std::byte>(workspace_size == 0 ? 1 : workspace_size);
    ASSERT_NE(workspace, nullptr);

    fill_device(data, input);
    ASSERT_EQ((algo::cuda::scan::inclusive_sum<WarpShuffle256x4>(
                  data, static_cast<std::uint32_t>(input.size()), workspace,
                  workspace_size)),
              cudaSuccess);
    sync_cuda();
    EXPECT_EQ(read_device(data, input.size()), cpu_inclusive_sum(input));

    fill_device(data, input);
    ASSERT_EQ((algo::cuda::scan::exclusive_sum<WarpShuffle256x4>(
                  data, static_cast<std::uint32_t>(input.size()), workspace,
                  workspace_size)),
              cudaSuccess);
    sync_cuda();
    EXPECT_EQ(read_device(data, input.size()), cpu_exclusive_sum(input));

    cudaFree(workspace);
    cudaFree(data);

    std::vector<std::uint64_t> large_input(2049);
    for (std::size_t i = 0; i < large_input.size(); ++i) {
        large_input[i] = static_cast<std::uint64_t>((i * 5) % 17);
    }

    auto* large_data = managed_alloc<std::uint64_t>(large_input.size());
    ASSERT_NE(large_data, nullptr);
    const std::size_t large_workspace_size =
        algo::cuda::scan::required_workspace_size<
            WarpShuffle256x4DecoupledLookback, std::uint64_t>(
            static_cast<std::uint32_t>(large_input.size()));
    auto* large_workspace = managed_alloc<std::byte>(large_workspace_size);
    ASSERT_NE(large_workspace, nullptr);

    fill_device(large_data, large_input);
    ASSERT_EQ((algo::cuda::scan::inclusive_sum<WarpShuffle256x4DecoupledLookback>(
                  large_data, static_cast<std::uint32_t>(large_input.size()),
                  large_workspace, large_workspace_size)),
              cudaSuccess);
    sync_cuda();
    EXPECT_EQ(read_device(large_data, large_input.size()),
              cpu_inclusive_sum(large_input));

    cudaFree(large_workspace);
    cudaFree(large_data);
}

TEST(CudaScan, DefaultExclusiveMaxUint64MatchesReference) {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    std::vector<std::uint64_t> input(2049);
    for (std::size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<std::uint64_t>((i * 37u) % 1021u);
    }

    auto* data = managed_alloc<std::uint64_t>(input.size());
    ASSERT_NE(data, nullptr);

    const std::size_t workspace_size =
        algo::cuda::scan::required_workspace_size<
            algo::cuda::scan::detail::DefaultConfig,
            algo::cuda::scan::Max<std::uint64_t>, std::uint64_t>(
            static_cast<std::uint32_t>(input.size()));
    auto* workspace = managed_alloc<std::byte>(workspace_size);
    ASSERT_NE(workspace, nullptr);

    fill_device(data, input);
    ASSERT_EQ((algo::cuda::scan::exclusive_scan<
                  algo::cuda::scan::Max<std::uint64_t>>(
                  data, static_cast<std::uint32_t>(input.size()), workspace,
                  workspace_size)),
              cudaSuccess);
    sync_cuda();
    EXPECT_EQ(read_device(data, input.size()), cpu_exclusive_max(input));

    cudaFree(workspace);
    cudaFree(data);
}

TEST(CudaScan, DecoupledLookbackReusesWorkspace) {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    std::vector<std::uint32_t> first(2049);
    std::vector<std::uint32_t> second(2049);
    for (std::size_t i = 0; i < first.size(); ++i) {
        first[i] = static_cast<std::uint32_t>(i % 7);
        second[i] = static_cast<std::uint32_t>((i * 3) % 11);
    }

    auto* data = managed_alloc<std::uint32_t>(first.size());
    ASSERT_NE(data, nullptr);

    const std::size_t workspace_size =
        algo::cuda::scan::required_workspace_size<
            WarpShuffle256x4DecoupledLookback, std::uint32_t>(
            static_cast<std::uint32_t>(first.size()));
    auto* workspace = managed_alloc<std::byte>(workspace_size);
    ASSERT_NE(workspace, nullptr);

    fill_device(data, first);
    ASSERT_EQ((algo::cuda::scan::inclusive_sum<WarpShuffle256x4DecoupledLookback>(
                  data, static_cast<std::uint32_t>(first.size()), workspace,
                  workspace_size)),
              cudaSuccess);
    sync_cuda();
    EXPECT_EQ(read_device(data, first.size()), cpu_inclusive_sum(first));

    fill_device(data, second);
    ASSERT_EQ((algo::cuda::scan::exclusive_sum<WarpShuffle256x4DecoupledLookback>(
                  data, static_cast<std::uint32_t>(second.size()), workspace,
                  workspace_size)),
              cudaSuccess);
    sync_cuda();
    EXPECT_EQ(read_device(data, second.size()), cpu_exclusive_sum(second));

    cudaFree(workspace);
    cudaFree(data);
}

} // namespace
