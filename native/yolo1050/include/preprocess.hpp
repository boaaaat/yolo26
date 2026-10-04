#pragma once
#include "common.hpp"
namespace y1050 {
void preprocess_bgr(const uint8_t* source, float* destination, const Geometry&, cudaStream_t);
cudaTextureObject_t preprocess_texture(cudaArray_t source, float* destination, const Geometry&,
                                      const ResizeEntry* x, const ResizeEntry* y, cudaStream_t);
}
