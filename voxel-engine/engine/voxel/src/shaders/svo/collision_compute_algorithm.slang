implementing svo;

public struct CollisionResultData {
    public float3 position;
    public float3 collided;
}

static const float COLLISION_EPSILON = 0.001;

public CollisionResultData computeCollisionResult<T : ISVOAccessor>(
    T svo,
    float3 pos,
    float3 halfExt,
    float3 velocity
) {
    float3 currentPos = pos - velocity;
    float3 remainingVel = velocity;
    float3 collided = float3(0, 0, 0);

    float3 worldHalfVoxelCount = float3(float(WORLD_VOXEL_COUNT) * 0.5);

    for (int iteration = 0; iteration < 3; iteration++) {
        if (
            abs(remainingVel.x) < COLLISION_EPSILON &&
            abs(remainingVel.y) < COLLISION_EPSILON &&
            abs(remainingVel.z) < COLLISION_EPSILON
        ) {
            break;
        }

        float3 endPos = currentPos + remainingVel;
        float3 sweepMin = min(currentPos, endPos) - halfExt;
        float3 sweepMax = max(currentPos, endPos) + halfExt;

        int3 minVoxel = int3(
            int(floor(sweepMin.x * VOXEL_SIZE_INV + worldHalfVoxelCount.x)),
            int(floor(sweepMin.y * VOXEL_SIZE_INV + worldHalfVoxelCount.y)),
            int(floor(sweepMin.z * VOXEL_SIZE_INV + worldHalfVoxelCount.z))
        );
        int3 maxVoxel = int3(
            int(floor(sweepMax.x * VOXEL_SIZE_INV + worldHalfVoxelCount.x)),
            int(floor(sweepMax.y * VOXEL_SIZE_INV + worldHalfVoxelCount.y)),
            int(floor(sweepMax.z * VOXEL_SIZE_INV + worldHalfVoxelCount.z))
        );

        int maxWorldVoxel = int(WORLD_VOXEL_COUNT) - 1;
        minVoxel = clamp(minVoxel, int3(0, 0, 0), int3(maxWorldVoxel, maxWorldVoxel, maxWorldVoxel));
        maxVoxel = clamp(maxVoxel, int3(0, 0, 0), int3(maxWorldVoxel, maxWorldVoxel, maxWorldVoxel));

        float bestT = 1e20;
        int bestHitAxis = -1;

        for (int vz = minVoxel.z; vz <= maxVoxel.z; ++vz) {
            for (int vy = minVoxel.y; vy <= maxVoxel.y; ++vy) {
                for (int vx = minVoxel.x; vx <= maxVoxel.x; ++vx) {
                    if (!svo.getVoxel(uint(vx), uint(vy), uint(vz))) {
                        continue;
                    }

                    float3 voxelMin = (float3(vx, vy, vz) - worldHalfVoxelCount) * VOXEL_SIZE;
                    float3 voxelMax = voxelMin + float3(VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE);
                    float3 expandedMin = voxelMin - halfExt;
                    float3 expandedMax = voxelMax + halfExt;

                    float3 entryT;
                    float3 exitT;

                    for (int axis = 0; axis < 3; axis++) {
                        float axisVel =
                            (axis == 0) ? remainingVel.x :
                            ((axis == 1) ? remainingVel.y : remainingVel.z);
                        float axisPos =
                            (axis == 0) ? currentPos.x :
                            ((axis == 1) ? currentPos.y : currentPos.z);
                        float axisMin =
                            (axis == 0) ? expandedMin.x :
                            ((axis == 1) ? expandedMin.y : expandedMin.z);
                        float axisMax =
                            (axis == 0) ? expandedMax.x :
                            ((axis == 1) ? expandedMax.y : expandedMax.z);

                        if (abs(axisVel) < 1e-8) {
                            if (axisPos < axisMin || axisPos > axisMax) {
                                if (axis == 0) {
                                    entryT.x = 1e20;
                                    exitT.x = -1e20;
                                } else if (axis == 1) {
                                    entryT.y = 1e20;
                                    exitT.y = -1e20;
                                } else {
                                    entryT.z = 1e20;
                                    exitT.z = -1e20;
                                }
                            } else {
                                if (axis == 0) {
                                    entryT.x = -1e20;
                                    exitT.x = 1e20;
                                } else if (axis == 1) {
                                    entryT.y = -1e20;
                                    exitT.y = 1e20;
                                } else {
                                    entryT.z = -1e20;
                                    exitT.z = 1e20;
                                }
                            }
                        } else {
                            float invVel = 1.0 / axisVel;
                            float t1 = (axisMin - axisPos) * invVel;
                            float t2 = (axisMax - axisPos) * invVel;
                            float tEntry = min(t1, t2);
                            float tExit = max(t1, t2);

                            if (axis == 0) {
                                entryT.x = tEntry;
                                exitT.x = tExit;
                            } else if (axis == 1) {
                                entryT.y = tEntry;
                                exitT.y = tExit;
                            } else {
                                entryT.z = tEntry;
                                exitT.z = tExit;
                            }
                        }
                    }

                    float tEnter = max(entryT.x, max(entryT.y, entryT.z));
                    float tExit = min(exitT.x, min(exitT.y, exitT.z));

                    if (tEnter > tExit || tEnter > 1.0 || tExit < 0.0) {
                        continue;
                    }

                    if (tEnter < 0.0) {
                        tEnter = 0.0;
                    }

                    if (tEnter < bestT) {
                        bestT = tEnter;

                        if (entryT.x >= entryT.y && entryT.x >= entryT.z) {
                            bestHitAxis = 0;
                        } else if (entryT.y >= entryT.x && entryT.y >= entryT.z) {
                            bestHitAxis = 1;
                        } else {
                            bestHitAxis = 2;
                        }
                    }
                }
            }
        }

        if (bestHitAxis < 0) {
            currentPos += remainingVel;
            break;
        }

        float correctedT = max(bestT - COLLISION_EPSILON, 0.0);
        currentPos += remainingVel * correctedT;
        remainingVel = (1.0 - bestT) * remainingVel;

        if (bestHitAxis == 0) {
            collided.x = 1.0;
            remainingVel.x = 0.0;
        } else if (bestHitAxis == 1) {
            collided.y = 1.0;
            remainingVel.y = 0.0;
        } else {
            collided.z = 1.0;
            remainingVel.z = 0.0;
        }
    }

    CollisionResultData result;
    result.position = currentPos;
    result.collided = collided;
    return result;
}
