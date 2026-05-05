#include "cuda_test_utils.cuh"

#include <algo/cuda/sort/radix.cuh>
#include <algo/cuda/sort/sort.cuh>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

namespace {

using HistogramWlmsPass =
    algo::cuda::sort::HistogramPass<
        algo::cuda::sort::SharedAtomicHistogram,
        algo::cuda::sort::WarpLevelMultiSplitWarpRank, 4>;
using HistogramWarpBallotPass =
    algo::cuda::sort::HistogramPass<
        algo::cuda::sort::WarpBallotHistogram,
        algo::cuda::sort::WarpLevelMultiSplitWarpRank, 4>;
using HistogramWarpLevelMultiSplitPass =
    algo::cuda::sort::HistogramPass<
        algo::cuda::sort::WarpLevelMultiSplitHistogram,
        algo::cuda::sort::WarpLevelMultiSplitWarpRank, 4>;
using OneSweepWlmsPass =
    algo::cuda::sort::OneSweepPass<algo::cuda::sort::WarpBallotBlockHistogram,
                                   algo::cuda::sort::WarpLevelMultiSplitWarpRank,
                                   algo::cuda::sort::
                                       SharedAtomicGlobalOffsetsBlockHistogram,
                                   1>;
using OneSweepWlmsItems4Pass =
    algo::cuda::sort::OneSweepPass<algo::cuda::sort::WarpBallotBlockHistogram,
                                   algo::cuda::sort::WarpLevelMultiSplitWarpRank,
                                   algo::cuda::sort::
                                       SharedAtomicGlobalOffsetsBlockHistogram,
                                   4>;
using OneSweepWlmsItems8Pass =
    algo::cuda::sort::OneSweepPass<algo::cuda::sort::WarpBallotBlockHistogram,
                                   algo::cuda::sort::WarpLevelMultiSplitWarpRank,
                                   algo::cuda::sort::
                                       SharedAtomicGlobalOffsetsBlockHistogram,
                                   8>;

using DefaultSort =
    algo::cuda::sort::RadixSort<OneSweepWlmsItems4Pass, 512, 4>;
using Bit1Sort =
    algo::cuda::sort::RadixSort<algo::cuda::sort::FlagPrefixSumPass<>, 256, 1>;
using Bit4Sort =
    algo::cuda::sort::RadixSort<algo::cuda::sort::FlagPrefixSumPass<>, 256, 4>;
using Lower4BitSort =
    algo::cuda::sort::RadixSort<algo::cuda::sort::FlagPrefixSumPass<>, 256, 4,
                                4>;
using Lower8BitSort =
    algo::cuda::sort::RadixSort<algo::cuda::sort::FlagPrefixSumPass<>, 256, 4,
                                8>;
using FlattenedSort = algo::cuda::sort::RadixSort<
    algo::cuda::sort::FlagPrefixSumPass<algo::cuda::sort::FlattenedScan>, 256,
    4>;
using FlattenedLower8BitSort = algo::cuda::sort::RadixSort<
    algo::cuda::sort::FlagPrefixSumPass<algo::cuda::sort::FlattenedScan>, 256,
    4, 8>;
using HistogramSort =
    algo::cuda::sort::RadixSort<HistogramWlmsPass, 256, 4>;
using HistogramLower8BitSort =
    algo::cuda::sort::RadixSort<HistogramWlmsPass, 256, 4, 8>;
using HistogramMultiSplitSort =
    algo::cuda::sort::RadixSort<HistogramWlmsPass, 256, 4>;
using HistogramMultiSplitLower8BitSort =
    algo::cuda::sort::RadixSort<HistogramWlmsPass, 256, 4, 8>;
using HistogramWarpBallotSort =
    algo::cuda::sort::RadixSort<HistogramWarpBallotPass, 256, 4>;
using HistogramWarpBallotLower8BitSort =
    algo::cuda::sort::RadixSort<HistogramWarpBallotPass, 256, 4, 8>;
using HistogramWarpLevelMultiSplitSort =
    algo::cuda::sort::RadixSort<HistogramWarpLevelMultiSplitPass, 256, 4>;
using HistogramWarpLevelMultiSplitLower8BitSort =
    algo::cuda::sort::RadixSort<HistogramWarpLevelMultiSplitPass, 256, 4, 8>;
using OneSweepSort =
    algo::cuda::sort::RadixSort<OneSweepWlmsPass, 256, 4>;
using OneSweepLower8BitSort =
    algo::cuda::sort::RadixSort<OneSweepWlmsPass, 256, 4, 8>;
using OneSweepRadix8Sort =
    algo::cuda::sort::RadixSort<OneSweepWlmsPass, 256, 8>;
using OneSweepItems4Radix8Sort =
    algo::cuda::sort::RadixSort<OneSweepWlmsItems4Pass, 256, 8>;
using OneSweepItems8Radix8Sort =
    algo::cuda::sort::RadixSort<OneSweepWlmsItems8Pass, 256, 8>;

using cuda_test::fill_device;
using cuda_test::has_cuda_device;
using cuda_test::managed_alloc;
using cuda_test::read_device;
using cuda_test::sync_cuda;

template <int KeyBits> std::uint32_t lower_key_bits(std::uint32_t value) {
    if constexpr (KeyBits >= 32) {
        return value;
    } else {
        return value & ((1u << KeyBits) - 1u);
    }
}

template <class Config>
void expect_sort_matches_reference(const std::vector<std::uint32_t>& input) {
    auto* data =
        managed_alloc<std::uint32_t>(input.size() == 0 ? 1 : input.size());
    ASSERT_NE(data, nullptr);
    const std::size_t workspace_size =
        algo::cuda::sort::required_workspace_size<Config, std::uint32_t>(
            static_cast<std::uint32_t>(input.size()));
    auto* workspace =
        managed_alloc<std::byte>(workspace_size == 0 ? 1 : workspace_size);
    ASSERT_NE(workspace, nullptr);

    fill_device(data, input);
    ASSERT_EQ((algo::cuda::sort::sort_keys<Config>(
                  data, static_cast<std::uint32_t>(input.size()), workspace,
                  workspace_size)),
              cudaSuccess);
    sync_cuda();

    auto expected = input;
    std::sort(expected.begin(), expected.end());
    EXPECT_EQ(read_device(data, input.size()), expected);

    cudaFree(workspace);
    cudaFree(data);
}

template <class Config>
void expect_sort_matches_lower_bits_reference(
    const std::vector<std::uint32_t>& input) {
    auto* data =
        managed_alloc<std::uint32_t>(input.size() == 0 ? 1 : input.size());
    ASSERT_NE(data, nullptr);
    const std::size_t workspace_size =
        algo::cuda::sort::required_workspace_size<Config, std::uint32_t>(
            static_cast<std::uint32_t>(input.size()));
    auto* workspace =
        managed_alloc<std::byte>(workspace_size == 0 ? 1 : workspace_size);
    ASSERT_NE(workspace, nullptr);

    fill_device(data, input);
    ASSERT_EQ((algo::cuda::sort::sort_keys<Config>(
                  data, static_cast<std::uint32_t>(input.size()), workspace,
                  workspace_size)),
              cudaSuccess);
    sync_cuda();

    auto expected = input;
    std::stable_sort(expected.begin(), expected.end(),
                     [](const auto lhs, const auto rhs) {
                         return lower_key_bits<Config::kKeyBits>(lhs) <
                                lower_key_bits<Config::kKeyBits>(rhs);
                     });
    EXPECT_EQ(read_device(data, input.size()), expected);

    cudaFree(workspace);
    cudaFree(data);
}

template <class Config>
void expect_sort_pairs_matches_reference(
    const std::vector<std::uint32_t>& input_keys,
    const std::vector<std::uint32_t>& input_values) {
    ASSERT_EQ(input_keys.size(), input_values.size());

    const std::size_t count = input_keys.size();
    auto* keys = managed_alloc<std::uint32_t>(count == 0 ? 1 : count);
    ASSERT_NE(keys, nullptr);
    auto* values = managed_alloc<std::uint32_t>(count == 0 ? 1 : count);
    ASSERT_NE(values, nullptr);
    const std::size_t workspace_size =
        algo::cuda::sort::required_pairs_workspace_size<Config, std::uint32_t,
                                                        std::uint32_t>(
            static_cast<std::uint32_t>(count));
    auto* workspace =
        managed_alloc<std::byte>(workspace_size == 0 ? 1 : workspace_size);
    ASSERT_NE(workspace, nullptr);

    fill_device(keys, input_keys);
    fill_device(values, input_values);
    ASSERT_EQ((algo::cuda::sort::sort_pairs<Config>(
                  keys, values, static_cast<std::uint32_t>(count), workspace,
                  workspace_size)),
              cudaSuccess);
    sync_cuda();

    std::vector<std::pair<std::uint32_t, std::uint32_t>> expected;
    expected.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        expected.emplace_back(input_keys[i], input_values[i]);
    }
    std::stable_sort(
        expected.begin(), expected.end(),
        [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });

    const auto sorted_keys = read_device(keys, count);
    const auto sorted_values = read_device(values, count);
    std::vector<std::pair<std::uint32_t, std::uint32_t>> actual;
    actual.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        actual.emplace_back(sorted_keys[i], sorted_values[i]);
    }

    EXPECT_EQ(actual, expected);

    cudaFree(workspace);
    cudaFree(values);
    cudaFree(keys);
}

template <class Config>
void expect_sort_pairs_matches_lower_bits_reference(
    const std::vector<std::uint32_t>& input_keys,
    const std::vector<std::uint32_t>& input_values) {
    ASSERT_EQ(input_keys.size(), input_values.size());

    const std::size_t count = input_keys.size();
    auto* keys = managed_alloc<std::uint32_t>(count == 0 ? 1 : count);
    ASSERT_NE(keys, nullptr);
    auto* values = managed_alloc<std::uint32_t>(count == 0 ? 1 : count);
    ASSERT_NE(values, nullptr);
    const std::size_t workspace_size =
        algo::cuda::sort::required_pairs_workspace_size<Config, std::uint32_t,
                                                        std::uint32_t>(
            static_cast<std::uint32_t>(count));
    auto* workspace =
        managed_alloc<std::byte>(workspace_size == 0 ? 1 : workspace_size);
    ASSERT_NE(workspace, nullptr);

    fill_device(keys, input_keys);
    fill_device(values, input_values);
    ASSERT_EQ((algo::cuda::sort::sort_pairs<Config>(
                  keys, values, static_cast<std::uint32_t>(count), workspace,
                  workspace_size)),
              cudaSuccess);
    sync_cuda();

    std::vector<std::pair<std::uint32_t, std::uint32_t>> expected;
    expected.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        expected.emplace_back(input_keys[i], input_values[i]);
    }
    std::stable_sort(expected.begin(), expected.end(),
                     [](const auto& lhs, const auto& rhs) {
                         return lower_key_bits<Config::kKeyBits>(lhs.first) <
                                lower_key_bits<Config::kKeyBits>(rhs.first);
                     });

    const auto sorted_keys = read_device(keys, count);
    const auto sorted_values = read_device(values, count);
    std::vector<std::pair<std::uint32_t, std::uint32_t>> actual;
    actual.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        actual.emplace_back(sorted_keys[i], sorted_values[i]);
    }

    EXPECT_EQ(actual, expected);

    cudaFree(workspace);
    cudaFree(values);
    cudaFree(keys);
}

TEST(CudaSort, RejectsInsufficientWorkspace) {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    auto* data = managed_alloc<std::uint32_t>(64);
    ASSERT_NE(data, nullptr);
    auto* workspace = managed_alloc<std::byte>(4);
    ASSERT_NE(workspace, nullptr);

    EXPECT_EQ(
        (algo::cuda::sort::sort_keys<DefaultSort>(data, 64, workspace, 4)),
        cudaErrorInvalidValue);

    cudaFree(workspace);
    cudaFree(data);
}

TEST(CudaSort, SortPairsRejectsInsufficientWorkspace) {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    auto* keys = managed_alloc<std::uint32_t>(64);
    ASSERT_NE(keys, nullptr);
    auto* values = managed_alloc<std::uint32_t>(64);
    ASSERT_NE(values, nullptr);
    auto* workspace = managed_alloc<std::byte>(4);
    ASSERT_NE(workspace, nullptr);

    EXPECT_EQ((algo::cuda::sort::sort_pairs<DefaultSort>(keys, values, 64,
                                                         workspace, 4)),
              cudaErrorInvalidValue);

    cudaFree(workspace);
    cudaFree(values);
    cudaFree(keys);
}

TEST(CudaSort, DefaultSortMatchesReferenceOnFixedInput) {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    expect_sort_matches_reference<DefaultSort>({3, 1, 4, 1, 5, 9, 2, 6, 5});
}

TEST(CudaSort, Bit1SortMatchesReferenceOnFixedInput) {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    expect_sort_matches_reference<Bit1Sort>({9, 7, 5, 3, 1, 8, 6, 4, 2, 0});
}

TEST(CudaSort, Bit4SortHandlesRandomNonPowerOfTwoInput) {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    std::mt19937 rng(123);
    std::uniform_int_distribution<std::uint32_t> dist(0, 4095);

    std::vector<std::uint32_t> input(257);
    for (auto& value : input)
        value = dist(rng);

    expect_sort_matches_reference<Bit4Sort>(input);
}

TEST(CudaSort, LowerBitSortIgnoresUpperBits) {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    expect_sort_matches_lower_bits_reference<Lower4BitSort>(
        {0x30u, 0x21u, 0x12u, 0x03u, 0x34u, 0x25u, 0x16u, 0x07u});
}

TEST(CudaSort, LowerBitSortHandlesMultiplePasses) {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    expect_sort_matches_lower_bits_reference<Lower8BitSort>(
        {0x1ffu, 0x281u, 0x142u, 0x003u, 0x3c0u, 0x224u, 0x165u, 0x026u});
}

TEST(CudaSort, DefaultSortHandlesDuplicatesAndZeros) {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    expect_sort_matches_reference<DefaultSort>(
        {0, 42, 0, 42, 7, 7, 7, 1, 0, 999, 42, 3});
}

TEST(CudaSort, FlattenedSortMatchesReferenceOnFixedInput) {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    expect_sort_matches_reference<FlattenedSort>({3, 1, 4, 1, 5, 9, 2, 6, 5});
}

TEST(CudaSort, FlattenedSortHandlesDuplicatesAndZeros) {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    expect_sort_matches_reference<FlattenedSort>(
        {0, 42, 0, 42, 7, 7, 7, 1, 0, 999, 42, 3});
}

TEST(CudaSort, FlattenedLowerBitSortHandlesMultiplePasses) {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    expect_sort_matches_lower_bits_reference<FlattenedLower8BitSort>(
        {0x1ffu, 0x281u, 0x142u, 0x003u, 0x3c0u, 0x224u, 0x165u, 0x026u});
}

TEST(CudaSort, HistogramSortMatchesReferenceOnFixedInput) {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    expect_sort_matches_reference<HistogramSort>({3, 1, 4, 1, 5, 9, 2, 6, 5});
}

TEST(CudaSort, HistogramSortHandlesRandomNonPowerOfTwoInput) {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    std::mt19937 rng(123);
    std::uniform_int_distribution<std::uint32_t> dist(0, 4095);

    std::vector<std::uint32_t> input(257);
    for (auto& value : input)
        value = dist(rng);

    expect_sort_matches_reference<HistogramSort>(input);
}

TEST(CudaSort, HistogramLowerBitSortHandlesMultiplePasses) {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    expect_sort_matches_lower_bits_reference<HistogramLower8BitSort>(
        {0x1ffu, 0x281u, 0x142u, 0x003u, 0x3c0u, 0x224u, 0x165u, 0x026u});
}

TEST(CudaSort, HistogramMultiSplitSortHandlesRandomNonPowerOfTwoInput) {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    std::mt19937 rng(123);
    std::uniform_int_distribution<std::uint32_t> dist(0, 4095);

    std::vector<std::uint32_t> input(257);
    for (auto& value : input)
        value = dist(rng);

    expect_sort_matches_reference<HistogramMultiSplitSort>(input);
}

TEST(CudaSort, HistogramMultiSplitLowerBitSortHandlesMultiplePasses) {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    expect_sort_matches_lower_bits_reference<HistogramMultiSplitLower8BitSort>(
        {0x1ffu, 0x281u, 0x142u, 0x003u, 0x3c0u, 0x224u, 0x165u, 0x026u});
}

TEST(CudaSort, HistogramWarpBallotSortHandlesRandomNonPowerOfTwoInput) {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    std::mt19937 rng(789);
    std::uniform_int_distribution<std::uint32_t> dist(0, 4095);

    std::vector<std::uint32_t> input(257);
    for (auto& value : input)
        value = dist(rng);

    expect_sort_matches_reference<HistogramWarpBallotSort>(input);
}

TEST(CudaSort, HistogramWarpBallotLowerBitSortHandlesMultiplePasses) {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    expect_sort_matches_lower_bits_reference<HistogramWarpBallotLower8BitSort>(
        {0x1ffu, 0x281u, 0x142u, 0x003u, 0x3c0u, 0x224u, 0x165u, 0x026u});
}

TEST(CudaSort, HistogramWarpLevelMultiSplitSortHandlesRandomNonPowerOfTwoInput) {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    std::mt19937 rng(987);
    std::uniform_int_distribution<std::uint32_t> dist(0, 4095);

    std::vector<std::uint32_t> input(257);
    for (auto& value : input)
        value = dist(rng);

    expect_sort_matches_reference<HistogramWarpLevelMultiSplitSort>(input);
}

TEST(CudaSort, HistogramWarpLevelMultiSplitLowerBitSortHandlesMultiplePasses) {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    expect_sort_matches_lower_bits_reference<
        HistogramWarpLevelMultiSplitLower8BitSort>(
        {0x1ffu, 0x281u, 0x142u, 0x003u, 0x3c0u, 0x224u, 0x165u, 0x026u});
}

TEST(CudaSort, OneSweepSortMatchesReferenceOnFixedInput) {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    expect_sort_matches_reference<OneSweepSort>({3, 1, 4, 1, 5, 9, 2, 6, 5});
}

TEST(CudaSort, OneSweepSortHandlesRandomNonPowerOfTwoInput) {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    std::mt19937 rng(123);
    std::uniform_int_distribution<std::uint32_t> dist(0, 4095);

    std::vector<std::uint32_t> input(257);
    for (auto& value : input)
        value = dist(rng);

    expect_sort_matches_reference<OneSweepSort>(input);
}

TEST(CudaSort, OneSweepSortHandlesRandomMultiTileInput) {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    std::mt19937 rng(456);
    std::uniform_int_distribution<std::uint32_t> dist;

    std::vector<std::uint32_t> input(4099);
    for (auto& value : input)
        value = dist(rng);

    expect_sort_matches_reference<OneSweepSort>(input);
}

TEST(CudaSort, OneSweepLowerBitSortHandlesMultiplePasses) {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    expect_sort_matches_lower_bits_reference<OneSweepLower8BitSort>(
        {0x1ffu, 0x281u, 0x142u, 0x003u, 0x3c0u, 0x224u, 0x165u, 0x026u});
}

TEST(CudaSort, OneSweepRadix8SortHandlesRandomInput) {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    std::mt19937 rng(321);
    std::uniform_int_distribution<std::uint32_t> dist;

    std::vector<std::uint32_t> input(1025);
    for (auto& value : input)
        value = dist(rng);

    expect_sort_matches_reference<OneSweepRadix8Sort>(input);
}

TEST(CudaSort, OneSweepItemsPerThreadRadix8SortHandlesRandomInput) {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    std::mt19937 rng(654);
    std::uniform_int_distribution<std::uint32_t> dist;

    std::vector<std::uint32_t> input(8197);
    for (auto& value : input)
        value = dist(rng);

    expect_sort_matches_reference<OneSweepItems4Radix8Sort>(input);
    expect_sort_matches_reference<OneSweepItems8Radix8Sort>(input);
}

TEST(CudaSort, SortPairsPreservesValuesOnFixedInput) {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    expect_sort_pairs_matches_reference<DefaultSort>(
        {3, 1, 4, 1, 5, 9, 2, 6, 5}, {30, 10, 40, 11, 50, 90, 20, 60, 51});
}

TEST(CudaSort, SortPairsHandlesDuplicatesStably) {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    expect_sort_pairs_matches_reference<DefaultSort>(
        {0, 42, 0, 42, 7, 7, 7, 1, 0, 999, 42, 3},
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
}

TEST(CudaSort, SortPairsHandlesRandomNonPowerOfTwoInput) {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    std::mt19937 rng(123);
    std::uniform_int_distribution<std::uint32_t> dist(0, 4095);

    std::vector<std::uint32_t> keys(257);
    std::vector<std::uint32_t> values(257);
    for (std::size_t i = 0; i < keys.size(); ++i) {
        keys[i] = dist(rng);
        values[i] = static_cast<std::uint32_t>(i);
    }

    expect_sort_pairs_matches_reference<Bit4Sort>(keys, values);
}

TEST(CudaSort, SortPairsWithLowerBitsPreservesEquivalentKeyOrder) {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    expect_sort_pairs_matches_lower_bits_reference<Lower4BitSort>(
        {0x11u, 0x21u, 0x01u, 0x12u, 0x02u, 0x33u}, {0u, 1u, 2u, 3u, 4u, 5u});
}

TEST(CudaSort, SortPairsHandlesEmptyAndSingleElementInputs) {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    expect_sort_pairs_matches_reference<DefaultSort>({}, {});
    expect_sort_pairs_matches_reference<DefaultSort>({7}, {70});
}

TEST(CudaSort, FlattenedSortPairsPreservesValues) {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    expect_sort_pairs_matches_reference<FlattenedSort>(
        {0, 42, 0, 42, 7, 7, 7, 1, 0, 999, 42, 3},
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
}

TEST(CudaSort, FlattenedSortPairsWithLowerBitsPreservesEquivalentKeyOrder) {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    expect_sort_pairs_matches_lower_bits_reference<FlattenedLower8BitSort>(
        {0x041u, 0x081u, 0x001u, 0x022u, 0x062u, 0x003u},
        {0u, 1u, 2u, 3u, 4u, 5u});
}

TEST(CudaSort, HistogramSortPairsPreservesValues) {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    expect_sort_pairs_matches_reference<HistogramSort>(
        {0, 42, 0, 42, 7, 7, 7, 1, 0, 999, 42, 3},
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
}

TEST(CudaSort, HistogramSortPairsWithLowerBitsPreservesEquivalentKeyOrder) {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    expect_sort_pairs_matches_lower_bits_reference<HistogramLower8BitSort>(
        {0x041u, 0x081u, 0x001u, 0x022u, 0x062u, 0x003u},
        {0u, 1u, 2u, 3u, 4u, 5u});
}

TEST(CudaSort, OneSweepSortPairsPreservesValues) {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    expect_sort_pairs_matches_reference<OneSweepSort>(
        {0, 42, 0, 42, 7, 7, 7, 1, 0, 999, 42, 3},
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
}

TEST(CudaSort, OneSweepSortPairsWithLowerBitsPreservesEquivalentKeyOrder) {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    expect_sort_pairs_matches_lower_bits_reference<OneSweepLower8BitSort>(
        {0x041u, 0x081u, 0x001u, 0x022u, 0x062u, 0x003u},
        {0u, 1u, 2u, 3u, 4u, 5u});
}

TEST(CudaSort, OneSweepSortPairsPreservesLargeEqualDigitRunOrder) {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    std::vector<std::uint32_t> keys(4097, 7u);
    std::vector<std::uint32_t> values(keys.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<std::uint32_t>(i);
    }

    expect_sort_pairs_matches_reference<OneSweepSort>(keys, values);
}

TEST(CudaSort, OneSweepItemsPerThreadSortPairsPreservesValues) {
    cudaError_t status = cudaSuccess;
    if (!has_cuda_device(&status)) {
        GTEST_SKIP() << "CUDA device is not available: "
                     << cudaGetErrorString(status);
    }

    std::vector<std::uint32_t> keys(8197);
    std::vector<std::uint32_t> values(keys.size());
    std::mt19937 rng(987);
    std::uniform_int_distribution<std::uint32_t> dist;
    for (std::size_t i = 0; i < keys.size(); ++i) {
        keys[i] = dist(rng);
        values[i] = static_cast<std::uint32_t>(i);
    }

    expect_sort_pairs_matches_reference<OneSweepItems4Radix8Sort>(keys, values);
    expect_sort_pairs_matches_reference<OneSweepItems8Radix8Sort>(keys, values);
}

} // namespace
