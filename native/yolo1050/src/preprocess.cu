#include "preprocess.hpp"
namespace y1050 {
__global__ void packed_to_nchw(const uint8_t* source, float* output, int pixels) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pixels) return;
    output[i] = source[i * 3 + 2] * (1.0f / 255);
    output[pixels + i] = source[i * 3 + 1] * (1.0f / 255);
    output[pixels * 2 + i] = source[i * 3] * (1.0f / 255);
}
__device__ int interpolate_channel(uchar4 a, uchar4 b, uchar4 c, uchar4 d, ResizeEntry x, ResizeEntry y, int channel) {
    int av = channel == 0 ? a.z : channel == 1 ? a.y : a.x;
    int bv = channel == 0 ? b.z : channel == 1 ? b.y : b.x;
    int cv = channel == 0 ? c.z : channel == 1 ? c.y : c.x;
    int dv = channel == 0 ? d.z : channel == 1 ? d.y : d.x;
    return ((av * x.a0 + bv * x.a1) * y.a0 + (cv * x.a0 + dv * x.a1) * y.a1 + (1 << 21)) >> 22;
}
__global__ void texture_to_nchw(cudaTextureObject_t texture, float* output, Geometry g,
                              const ResizeEntry* xs, const ResizeEntry* ys) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int pixels = g.width * g.height; if (i >= pixels) return;
    int x = i % g.width - g.left, y = i / g.width - g.top;
    if (x < 0 || y < 0 || x >= g.resized_w || y >= g.resized_h) {
        for (int ch = 0; ch < 3; ++ch) output[ch * pixels + i] = 114.0f / 255;
        return;
    }
    // Identical fixed-point coefficients are reused by CPU capture and calibration preparation.
    auto rx = xs[x], ry = ys[y];
    auto a = tex2D<uchar4>(texture, rx.first + .5f, ry.first + .5f);
    auto b = tex2D<uchar4>(texture, rx.second + .5f, ry.first + .5f);
    auto c = tex2D<uchar4>(texture, rx.first + .5f, ry.second + .5f);
    auto d = tex2D<uchar4>(texture, rx.second + .5f, ry.second + .5f);
    for (int ch = 0; ch < 3; ++ch) {
        output[ch * pixels + i] = interpolate_channel(a, b, c, d, rx, ry, ch) * (1.0f / 255);
    }
}
void preprocess_bgr(const uint8_t* source, float* destination, const Geometry& g, cudaStream_t stream) {
    int n = g.width * g.height; packed_to_nchw<<<(n + 255) / 256, 256, 0, stream>>>(source, destination, n);
    cuda_check(cudaGetLastError(), "BGR preprocessing");
}
cudaTextureObject_t preprocess_texture(cudaArray_t source, float* destination, const Geometry& g,
                                      const ResizeEntry* xs, const ResizeEntry* ys, cudaStream_t stream) {
    cudaResourceDesc resource{}; resource.resType = cudaResourceTypeArray; resource.res.array.array = source;
    cudaTextureDesc description{}; description.addressMode[0] = cudaAddressModeClamp;
    description.addressMode[1] = cudaAddressModeClamp; description.filterMode = cudaFilterModePoint;
    description.readMode = cudaReadModeElementType;
    cudaTextureObject_t texture = 0;
    cuda_check(cudaCreateTextureObject(&texture, &resource, &description, nullptr), "Create capture texture");
    int n = g.width * g.height; texture_to_nchw<<<(n + 255) / 256, 256, 0, stream>>>(texture, destination, g, xs, ys);
    auto error = cudaGetLastError();
    if (error != cudaSuccess) { cudaDestroyTextureObject(texture); cuda_check(error, "Texture preprocessing"); }
    // The caller destroys this object only after the completion event.
    return texture;
}
}
