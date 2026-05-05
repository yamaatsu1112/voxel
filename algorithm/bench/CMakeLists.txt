include(FetchContent)

FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.9.1
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googlebenchmark)

if(ALGO_HAS_CUDA)
  add_executable(bench_cuda_scan bench_cuda_scan.cu)
  target_link_libraries(bench_cuda_scan PRIVATE algo benchmark::benchmark_main)
  set_target_properties(bench_cuda_scan PROPERTIES CUDA_STANDARD 17
                                                   CUDA_STANDARD_REQUIRED ON)

  add_executable(bench_cuda_sort bench_cuda_sort.cu)
  target_link_libraries(bench_cuda_sort PRIVATE algo benchmark::benchmark_main)
  set_target_properties(bench_cuda_sort PROPERTIES CUDA_STANDARD 17
                                                   CUDA_STANDARD_REQUIRED ON)

  add_executable(bench_cuda_hash_table bench_cuda_hash_table.cu)
  target_link_libraries(bench_cuda_hash_table PRIVATE algo benchmark::benchmark_main)
  set_target_properties(bench_cuda_hash_table PROPERTIES CUDA_STANDARD 17
                                                         CUDA_STANDARD_REQUIRED ON)

  add_executable(bench_cuda_hash_dag_gpu bench_cuda_hash_dag_gpu.cu)
  target_link_libraries(bench_cuda_hash_dag_gpu PRIVATE algo benchmark::benchmark_main)
  set_target_properties(bench_cuda_hash_dag_gpu PROPERTIES CUDA_STANDARD 17
                                                           CUDA_STANDARD_REQUIRED ON)

  add_executable(bench_cuda_svt bench_cuda_svt.cu)
  target_link_libraries(bench_cuda_svt PRIVATE algo benchmark::benchmark_main)
  set_target_properties(bench_cuda_svt PROPERTIES CUDA_STANDARD 17
                                                  CUDA_STANDARD_REQUIRED ON)

  add_executable(bench_cuda_svt_phases bench_cuda_svt_phases.cu)
  target_link_libraries(bench_cuda_svt_phases PRIVATE algo benchmark::benchmark_main)
  set_target_properties(bench_cuda_svt_phases PROPERTIES CUDA_STANDARD 17
                                                         CUDA_STANDARD_REQUIRED ON)

  add_executable(bench_cuda_svt_terminal bench_cuda_svt_terminal.cu)
  target_link_libraries(bench_cuda_svt_terminal PRIVATE algo benchmark::benchmark_main)
  set_target_properties(bench_cuda_svt_terminal PROPERTIES CUDA_STANDARD 17
                                                           CUDA_STANDARD_REQUIRED ON)

  add_executable(bench_cuda_svt_terminal_phases bench_cuda_svt_terminal_phases.cu)
  target_link_libraries(bench_cuda_svt_terminal_phases PRIVATE algo benchmark::benchmark_main)
  set_target_properties(bench_cuda_svt_terminal_phases PROPERTIES CUDA_STANDARD 17
                                                                  CUDA_STANDARD_REQUIRED ON)
endif()
