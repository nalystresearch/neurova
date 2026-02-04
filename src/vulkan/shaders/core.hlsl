/**
 * Neurova HLSL Compute Shaders
 * GPU compute kernels for Vulkan/DirectX
 */

// ============================================================================
// Core Operations
// ============================================================================

// Add two arrays
[[vk::binding(0, 0)]] RWStructuredBuffer<float> BufferA : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<float> BufferB : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<float> BufferOut : register(u2);

struct AddParams {
    uint count;
};
[[vk::push_constant]] ConstantBuffer<AddParams> params;

[numthreads(256, 1, 1)]
void add_cs(uint3 DTid : SV_DispatchThreadID) {
    if (DTid.x < params.count) {
        BufferOut[DTid.x] = BufferA[DTid.x] + BufferB[DTid.x];
    }
}

[numthreads(256, 1, 1)]
void multiply_cs(uint3 DTid : SV_DispatchThreadID) {
    if (DTid.x < params.count) {
        BufferOut[DTid.x] = BufferA[DTid.x] * BufferB[DTid.x];
    }
}

[numthreads(256, 1, 1)]
void scale_cs(uint3 DTid : SV_DispatchThreadID, float scale) {
    if (DTid.x < params.count) {
        BufferA[DTid.x] *= scale;
    }
}
