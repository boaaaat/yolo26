#pragma once
#include "common.hpp"
#include "preprocess.hpp"
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <NvInferPlugin.h>
namespace y1050 {
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) std::cerr << "[TensorRT] " << message << '\n';
    }
};
template<class T> using TrtPtr = std::unique_ptr<T>;
struct EngineArtifact { fs::path path; Json metadata; };
Json engine_identity(const Options&);
EngineArtifact engine_artifact(const Options&, bool build_only);
class Runner {
    Logger logger_;
    TrtPtr<nvinfer1::IRuntime> runtime_;
    TrtPtr<nvinfer1::ICudaEngine> engine_;
    TrtPtr<nvinfer1::IExecutionContext> context_;
    cudaStream_t stream_ = nullptr;
    cudaEvent_t done_ = nullptr, start_ = nullptr, preprocessed_ = nullptr, inferred_ = nullptr, downloaded_ = nullptr;
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t executable_ = nullptr;
    uint8_t* raw_ = nullptr;
    ResizeEntry *xs_ = nullptr, *ys_ = nullptr;
    float *input_ = nullptr, *output_ = nullptr, *host_output_ = nullptr;
    cudaGraphicsResource_t mapped_ = nullptr;
    cudaTextureObject_t texture_ = 0;
    size_t output_bytes_ = 0;
    bool timing_ = false, pending_ = false;
    void cleanup() noexcept;
public:
    const Geometry geometry;
    int candidates = 0;
    Runner(const EngineArtifact&, const Geometry&, const Options&);
    ~Runner() { cleanup(); }
    Runner(const Runner&) = delete;
    Runner& operator=(const Runner&) = delete;
    void submit(const uint8_t* pinned_bgr, cudaGraphicsResource_t resource);
    const float* finish(float& preprocessing_ms, float& model_ms, float& transfer_ms);
};
}
