#pragma once

TEST(CudaTerminalEditValidation, EmptyApplyStillValidatesWorkspace) {
    expect_cuda_device_or_skip();

    TestGpuSvo svo;
    ASSERT_EQ(algo::svt::cuda::reset_svo(svo.view()), cudaSuccess);

    const auto status = algo::svt::cuda::detail::apply_terminal_edits<
        algo::svt::cuda::DefaultTerminalEditConfig::allocation,
        algo::svt::cuda::DefaultTerminalEditConfig::release>(
        svo.view(), nullptr, 0u, nullptr, 0u, algo::svt::cuda::EditOp::Place,
        nullptr, 0u);
    EXPECT_EQ(status, cudaErrorInvalidValue);
}
