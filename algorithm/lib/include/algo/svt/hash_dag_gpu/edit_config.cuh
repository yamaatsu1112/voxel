#pragma once

namespace algo::svt::hash_dag_gpu {

struct Fused {};
struct HostLeafCountDispatch {};

template <class Mode> struct ScanDepthwiseRequests {};

template <class RequestBuilder, class Dispatch> struct EditConfig {
  using request_builder = RequestBuilder;
  using dispatch = Dispatch;
};

} // namespace algo::svt::hash_dag_gpu
