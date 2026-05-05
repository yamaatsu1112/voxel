add_executable(svt_node_count svt_node_count.cpp)
target_link_libraries(svt_node_count PRIVATE algo)

add_executable(svo_forwarding_index_count svo_forwarding_index_count.cpp)
target_link_libraries(svo_forwarding_index_count PRIVATE algo)

add_executable(svo_forwarding_packed_index_count
               svo_forwarding_packed_index_count.cpp)
target_link_libraries(svo_forwarding_packed_index_count PRIVATE algo)
