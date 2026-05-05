include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.15.2
)
FetchContent_MakeAvailable(googletest)

add_executable(test_svt_svo svt/test_svt_svo.cpp)
target_link_libraries(test_svt_svo PRIVATE algo GTest::gtest_main)
add_test(NAME test_svt_svo COMMAND test_svt_svo)

add_executable(test_svt_svo_no_leaf svt/test_svt_svo_no_leaf.cpp)
target_link_libraries(test_svt_svo_no_leaf PRIVATE algo GTest::gtest_main)
add_test(NAME test_svt_svo_no_leaf COMMAND test_svt_svo_no_leaf)

add_executable(test_svt_svo_forwarding svt/test_svt_svo_forwarding.cpp)
target_link_libraries(test_svt_svo_forwarding PRIVATE algo GTest::gtest_main)
add_test(NAME test_svt_svo_forwarding COMMAND test_svt_svo_forwarding)

add_executable(test_svt_svo_forwarding_packed svt/test_svt_svo_forwarding_packed.cpp)
target_link_libraries(test_svt_svo_forwarding_packed PRIVATE algo GTest::gtest_main)
add_test(NAME test_svt_svo_forwarding_packed COMMAND test_svt_svo_forwarding_packed)

add_executable(test_svt_svo_forwarding_packed_leaf
               svt/test_svt_svo_forwarding_packed_leaf.cpp)
target_link_libraries(test_svt_svo_forwarding_packed_leaf PRIVATE algo
                                                          GTest::gtest_main)
add_test(NAME test_svt_svo_forwarding_packed_leaf COMMAND test_svt_svo_forwarding_packed_leaf)

add_executable(test_svt_svo64 svt/test_svt_svo64.cpp)
target_link_libraries(test_svt_svo64 PRIVATE algo GTest::gtest_main)
add_test(NAME test_svt_svo64 COMMAND test_svt_svo64)

add_executable(test_svt_svo_hamming svt/test_svt_svo_hamming.cpp)
target_link_libraries(test_svt_svo_hamming PRIVATE algo GTest::gtest_main)
add_test(NAME test_svt_svo_hamming COMMAND test_svt_svo_hamming)

add_executable(test_svt_hash_dag svt/test_svt_hash_dag.cpp)
target_link_libraries(test_svt_hash_dag PRIVATE algo GTest::gtest_main)
add_test(NAME test_svt_hash_dag COMMAND test_svt_hash_dag)

add_executable(test_svt_svdag svt/test_svt_svdag.cpp)
target_link_libraries(test_svt_svdag PRIVATE algo GTest::gtest_main)
add_test(NAME test_svt_svdag COMMAND test_svt_svdag)

add_executable(test_svt_svo_grouped svt/test_svt_svo_grouped.cpp)
target_link_libraries(test_svt_svo_grouped PRIVATE algo GTest::gtest_main)
add_test(NAME test_svt_svo_grouped COMMAND test_svt_svo_grouped)

add_executable(test_svt_svo_grouped_no_leaf svt/test_svt_svo_grouped_no_leaf.cpp)
target_link_libraries(test_svt_svo_grouped_no_leaf PRIVATE algo GTest::gtest_main)
add_test(NAME test_svt_svo_grouped_no_leaf COMMAND test_svt_svo_grouped_no_leaf)

add_executable(test_svt_svo64_grouped svt/test_svt_svo64_grouped.cpp)
target_link_libraries(test_svt_svo64_grouped PRIVATE algo GTest::gtest_main)
add_test(NAME test_svt_svo64_grouped COMMAND test_svt_svo64_grouped)

add_executable(test_svt_svo64_grouped_no_leaf svt/test_svt_svo64_grouped_no_leaf.cpp)
target_link_libraries(test_svt_svo64_grouped_no_leaf PRIVATE algo GTest::gtest_main)
add_test(NAME test_svt_svo64_grouped_no_leaf COMMAND test_svt_svo64_grouped_no_leaf)

if(ALGO_HAS_CUDA)
  add_executable(test_cuda_cuckoo_hash cuda/test_cuda_cuckoo_hash.cu)
  target_link_libraries(test_cuda_cuckoo_hash PRIVATE algo GTest::gtest_main)
  set_target_properties(test_cuda_cuckoo_hash PROPERTIES CUDA_STANDARD 17
                                                        CUDA_STANDARD_REQUIRED ON)
  add_test(NAME test_cuda_cuckoo_hash COMMAND test_cuda_cuckoo_hash)

  add_executable(test_cuda_slab_hash cuda/test_cuda_slab_hash.cu)
  target_link_libraries(test_cuda_slab_hash PRIVATE algo GTest::gtest_main)
  set_target_properties(test_cuda_slab_hash PROPERTIES CUDA_STANDARD 17
                                                       CUDA_STANDARD_REQUIRED ON)
  add_test(NAME test_cuda_slab_hash COMMAND test_cuda_slab_hash)

  add_executable(test_cuda_slab_hash_set cuda/test_cuda_slab_hash_set.cu)
  target_link_libraries(test_cuda_slab_hash_set PRIVATE algo GTest::gtest_main)
  set_target_properties(test_cuda_slab_hash_set PROPERTIES CUDA_STANDARD 17
                                                           CUDA_STANDARD_REQUIRED ON)
  add_test(NAME test_cuda_slab_hash_set COMMAND test_cuda_slab_hash_set)

  add_executable(test_cuda_slab_alloc cuda/test_cuda_slab_alloc.cu)
  target_link_libraries(test_cuda_slab_alloc PRIVATE algo GTest::gtest_main)
  set_target_properties(test_cuda_slab_alloc PROPERTIES CUDA_STANDARD 17
                                                        CUDA_STANDARD_REQUIRED ON)
  add_test(NAME test_cuda_slab_alloc COMMAND test_cuda_slab_alloc)

  add_executable(test_cuda_acceleration_hash cuda/test_cuda_acceleration_hash.cu)
  target_link_libraries(test_cuda_acceleration_hash PRIVATE algo GTest::gtest_main)
  set_target_properties(test_cuda_acceleration_hash PROPERTIES CUDA_STANDARD 17
                                                               CUDA_STANDARD_REQUIRED ON)
  add_test(NAME test_cuda_acceleration_hash COMMAND test_cuda_acceleration_hash)

  add_executable(test_cuda_hash_dag_gpu cuda/test_cuda_hash_dag_gpu.cu)
  target_link_libraries(test_cuda_hash_dag_gpu PRIVATE algo GTest::gtest_main)
  set_target_properties(test_cuda_hash_dag_gpu PROPERTIES CUDA_STANDARD 17
                                                          CUDA_STANDARD_REQUIRED ON)
  add_test(NAME test_cuda_hash_dag_gpu COMMAND test_cuda_hash_dag_gpu)

  add_executable(test_cuda_scan cuda/test_cuda_scan.cu)
  target_link_libraries(test_cuda_scan PRIVATE algo GTest::gtest_main)
  set_target_properties(test_cuda_scan PROPERTIES CUDA_STANDARD 17
                                                 CUDA_STANDARD_REQUIRED ON)
  add_test(NAME test_cuda_scan COMMAND test_cuda_scan)

  add_executable(test_cuda_sort cuda/test_cuda_sort.cu)
  target_link_libraries(test_cuda_sort PRIVATE algo GTest::gtest_main)
  set_target_properties(test_cuda_sort PROPERTIES CUDA_STANDARD 17
                                                 CUDA_STANDARD_REQUIRED ON)
  add_test(NAME test_cuda_sort COMMAND test_cuda_sort)

  add_executable(test_cuda_svt cuda/svt/test_cuda_svt.cu)
  target_link_libraries(test_cuda_svt PRIVATE algo GTest::gtest_main)
  set_target_properties(test_cuda_svt PROPERTIES CUDA_STANDARD 17
                                                CUDA_STANDARD_REQUIRED ON)
  add_test(NAME test_cuda_svt COMMAND test_cuda_svt)

  add_executable(test_cuda_terminal_edit
    cuda/terminal_edit/test_cuda_terminal_edit.cu)
  target_link_libraries(test_cuda_terminal_edit PRIVATE algo GTest::gtest_main)
  set_target_properties(test_cuda_terminal_edit PROPERTIES CUDA_STANDARD 17
                                                           CUDA_STANDARD_REQUIRED ON)
  add_test(NAME test_cuda_terminal_edit COMMAND test_cuda_terminal_edit)

endif()
