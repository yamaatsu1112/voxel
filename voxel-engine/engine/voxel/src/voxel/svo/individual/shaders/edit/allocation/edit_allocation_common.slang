module edit_allocation_common;

import svo;

public static const uint INVALID_REQUEST_NODE_INDEX = 0xFFFFFFFFu;

public bool hasNodeRequest(NodeRequest request) {
    return request.nodeIndex != INVALID_REQUEST_NODE_INDEX;
}

public bool sameNodeRequest(NodeRequest a, NodeRequest b) {
    return a.nodeIndex == b.nodeIndex && a.childIndex == b.childIndex;
}

public void clearNodeRequest(
    RWStructuredBuffer<NodeRequest> requestBuffer,
    uint threadIndex
) {
    NodeRequest request;
    request.nodeIndex = INVALID_REQUEST_NODE_INDEX;
    request.childIndex = 0u;
    requestBuffer[threadIndex] = request;
}
