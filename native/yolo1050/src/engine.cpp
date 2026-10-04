#include "engine.hpp"
#include <cuda_d3d11_interop.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
namespace y1050 {
namespace {
size_t cudnn_version() {
    auto library = LoadLibraryW(L"cudnn64_8.dll");
    require(library != nullptr, "cuDNN 8.9.0 runtime DLL is missing from the deployment DLL search path");
    using GetVersion = size_t(*)();
    auto get_version = reinterpret_cast<GetVersion>(GetProcAddress(library, "cudnnGetVersion"));
    auto version = get_version ? get_version() : 0;
    FreeLibrary(library);
    require(version == 8900, "Matching cuDNN 8.9.0 runtime required for this TensorRT deployment");
    return version;
}
Geometry export_geometry(const Json& manifest) {
    require(manifest.at("schema") == 1 && manifest.at("nms_free") == true &&
            manifest.at("preprocessing").at("id") == "y1050-bilinear11-u8-rgb-nchw-v1",
            "Unsupported export manifest/preprocessing");
    auto size = manifest.at("screen_size").get<std::vector<int>>();
    require(size.size() == 2, "Invalid export screen size");
    auto g = Geometry::make(size[0], size[1], 1024);
    require(manifest.at("input_shape") == Json::array({1, 3, g.height, g.width}), "Export geometry mismatch");
    require(manifest.at("names").get<std::vector<std::string>>() ==
            std::vector<std::string>({"dead", "enemy", "teammate"}), "Expected dead/enemy/teammate class order");
    return g;
}
class BuildLock {
    HANDLE handle_ = INVALID_HANDLE_VALUE;
public:
    explicit BuildLock(const fs::path& path) {
        handle_ = CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_ALWAYS, 0, nullptr);
        require(handle_ != INVALID_HANDLE_VALUE, "Engine cache is locked by another process: " + path.string());
    }
    ~BuildLock() { CloseHandle(handle_); }
};
class Calibrator : public nvinfer1::IInt8EntropyCalibrator2 {
    Json manifest_;
    fs::path directory_, cache_;
    Geometry geometry_;
    size_t cursor_ = 0;
    std::vector<char> cached_;
    uint8_t* raw_ = nullptr;
    float* input_ = nullptr;
    cudaStream_t stream_ = nullptr;
public:
    std::string failure;
    Calibrator(const fs::path& manifest, const fs::path& cache, const Geometry& geometry)
        : manifest_(read_json(manifest)), directory_(manifest.parent_path()), cache_(cache), geometry_(geometry) {
        require(manifest_.at("schema") == 1 && manifest_.at("split") == "train" &&
                manifest_.at("preprocessing_id") == "y1050-bilinear11-u8-rgb-nchw-v1" &&
                manifest_.at("input_shape") == Json::array({1, 3, geometry.height, geometry.width}) &&
                !manifest_.at("images").empty(), "Calibration must be training-only and match export preprocessing");
        try {
            cuda_check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "Calibration stream");
            size_t bytes = size_t(geometry.width) * geometry.height * 3;
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&raw_), bytes), "Calibration image allocation");
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&input_), bytes * sizeof(float)), "Calibration tensor allocation");
        } catch (...) { if (raw_) cudaFree(raw_); if (stream_) cudaStreamDestroy(stream_); throw; }
    }
    ~Calibrator() { cudaFree(input_); cudaFree(raw_); cudaStreamDestroy(stream_); }
    int getBatchSize() const noexcept override { return 1; }
    bool getBatch(void* bindings[], const char*[], int count) noexcept override {
        try {
            if (cursor_ == manifest_.at("images").size()) return false;
            require(count == 1, "Expected one calibration image binding");
            const auto& image = manifest_.at("images").at(cursor_++);
            auto path = resolve_path(directory_, image.at("path").get<std::string>());
            require(hash_file(path) == image.at("sha256"), "Calibration image changed: " + path.string());
            auto encoded = read_bytes(path);
            require(!encoded.empty() && encoded.size() <= 2147483647, "Invalid calibration PNG size");
            cv::Mat encoded_view(1, int(encoded.size()), CV_8UC1, encoded.data());
            auto matrix = cv::imdecode(encoded_view, cv::IMREAD_COLOR);
            require(!matrix.empty() && matrix.cols == geometry_.width && matrix.rows == geometry_.height &&
                    matrix.isContinuous(), "Calibration PNG must contain the exact padded model input");
            size_t bytes = size_t(geometry_.width) * geometry_.height * 3;
            cuda_check(cudaMemcpyAsync(raw_, matrix.data, bytes, cudaMemcpyHostToDevice, stream_), "Calibration upload");
            preprocess_bgr(raw_, input_, geometry_, stream_);
            cuda_check(cudaStreamSynchronize(stream_), "Calibration preprocessing");
            bindings[0] = input_; return true;
        } catch (const std::exception& error) {
            failure = error.what(); std::cerr << "Calibration failed: " << failure << '\n'; return false;
        }
    }
    const void* readCalibrationCache(size_t& length) noexcept override {
        try {
            if (fs::exists(cache_)) cached_ = read_bytes(cache_);
            length = cached_.size(); return cached_.empty() ? nullptr : cached_.data();
        } catch (const std::exception& error) { failure = error.what(); length = 0; return nullptr; }
    }
    void writeCalibrationCache(const void* data, size_t length) noexcept override {
        try { write_bytes(cache_, data, length); }
        catch (const std::exception& error) { failure = error.what(); }
    }
};
bool valid_artifact(const fs::path& path, const Json& identity) {
    try {
        auto meta = read_json(path.string() + ".json");
        return meta.at("identity") == identity && meta.at("engine_sha256") == hash_file(path);
    } catch (const std::exception&) { return false; }
}
void build(const Options& options, const Json& identity, const fs::path& destination) {
    Logger logger; auto exported = read_json(options.export_manifest); auto geometry = export_geometry(exported);
    require(initLibNvInferPlugins(&logger, ""), "TensorRT plugin initialization failed");
    auto onnx = resolve_path(options.export_manifest.parent_path(), exported.at("onnx").get<std::string>());
    TrtPtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
    require(bool(builder), "TensorRT builder creation failed");
    auto flags = 1U << static_cast<unsigned>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    TrtPtr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(flags));
    require(bool(network), "TensorRT network creation failed");
    TrtPtr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));
    require(bool(parser), "ONNX parser creation failed");
    auto model = read_bytes(onnx);
    if (!parser->parse(model.data(), model.size())) {
        std::string errors = "TensorRT 8.6 ONNX import failed:\n";
        for (int i = 0; i < parser->getNbErrors(); ++i) errors += std::string(parser->getError(i)->desc()) + "\n";
        throw std::runtime_error(errors);
    }
    require(network->getNbInputs() == 1 && network->getNbOutputs() == 1 &&
            network->getInput(0)->getType() == nvinfer1::DataType::kFLOAT &&
            network->getOutput(0)->getType() == nvinfer1::DataType::kFLOAT, "FP32 image/output I/O required");
    auto linear = 1U << static_cast<unsigned>(nvinfer1::TensorFormat::kLINEAR);
    network->getInput(0)->setAllowedFormats(linear);
    network->getOutput(0)->setAllowedFormats(linear);
    TrtPtr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    require(bool(config), "TensorRT builder configuration failed");
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, size_t(options.workspace_mib) << 20);
    config->setBuilderOptimizationLevel(5);
    config->setMaxAuxStreams(0);
    bool explicit_int8 = exported.value("quantization", "fp32") == "qat-int8";
    require(!explicit_int8 || options.precision == "int8", "QAT export cannot produce an FP32 baseline engine");
    std::unique_ptr<Calibrator> calibrator;
    if (options.precision == "int8") {
        require(builder->platformHasFastInt8(), "This GPU does not expose fast INT8");
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
        if (!explicit_int8) {
            calibrator = std::make_unique<Calibrator>(options.calibration_manifest,
                                                    destination.string() + ".calibration", geometry);
            config->setInt8Calibrator(calibrator.get());
        }
    }
    require(!explicit_int8 || options.fp32_layers.empty(),
            "QAT layer precision comes from Q/DQ nodes; disable quantizers before export instead");
    auto fp32_patterns = options.fp32_layers;
    if (options.precision == "int8" && !explicit_int8) {
        auto sensitive = exported.value("sensitive_fp32_layer_patterns", std::vector<std::string>{});
        fp32_patterns.insert(fp32_patterns.end(), sensitive.begin(), sensitive.end());
    }
    for (const auto& pattern : fp32_patterns) {
        require(!pattern.empty(), "Empty FP32 layer pattern"); bool found = false;
        for (int i = 0; i < network->getNbLayers(); ++i) {
            auto layer = network->getLayer(i);
            if (std::string(layer->getName()).find(pattern) != std::string::npos) {
                layer->setPrecision(nvinfer1::DataType::kFLOAT); found = true;
                for (int out = 0; out < layer->getNbOutputs(); ++out)
                    if (layer->getOutput(out)->getType() == nvinfer1::DataType::kFLOAT)
                        layer->setOutputType(out, nvinfer1::DataType::kFLOAT);
            }
        }
        require(found, "FP32 layer pattern matched no layers: " + pattern);
    }
    if (!fp32_patterns.empty()) config->setFlag(nvinfer1::BuilderFlag::kOBEY_PRECISION_CONSTRAINTS);
    auto timing_path = destination.string() + ".timing"; std::vector<char> timing_bytes;
    if (fs::exists(timing_path)) timing_bytes = read_bytes(timing_path);
    TrtPtr<nvinfer1::ITimingCache> timing(config->createTimingCache(timing_bytes.data(), timing_bytes.size()));
    require(bool(timing) && config->setTimingCache(*timing, false), "TensorRT timing cache rejected");
    std::cout << "Building " << options.precision << " engine on the target GPU; close Roblox first. "
              << "This includes TensorRT's internal tactic timing, not an FPS benchmark.\n";
    TrtPtr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
    require(!calibrator || calibrator->failure.empty(), calibrator ? calibrator->failure : "Calibration failed");
    require(bool(plan), "TensorRT engine build failed; inspect parser/calibration/memory diagnostics");
    // Recheck artifacts after the builder finishes to reject concurrent preparation.
    require(engine_identity(options) == identity, "Export or calibration changed while building");
    write_bytes(destination, plan->data(), plan->size());
    TrtPtr<nvinfer1::IHostMemory> timing_data(config->getTimingCache()->serialize());
    if (timing_data) write_bytes(timing_path, timing_data->data(), timing_data->size());
    write_json(destination.string() + ".json", Json{{"identity", identity}, {"export", exported},
        {"engine_sha256", hash_file(destination)}, {"qualified", false}});
}
} // namespace
Json engine_identity(const Options& o) {
    auto manifest = read_json(o.export_manifest); auto g = export_geometry(manifest);
    auto onnx = resolve_path(o.export_manifest.parent_path(), manifest.at("onnx").get<std::string>());
    require(hash_file(onnx) == manifest.at("onnx_sha256"), "ONNX changed; prepare a new export");
    cudaDeviceProp gpu{}; cuda_check(cudaGetDeviceProperties(&gpu, o.gpu), "GPU properties");
    require(gpu.major == 6 && gpu.minor == 1, "This deployment requires an SM 6.1 Pascal GPU, such as GTX 1050");
    int driver = 0, runtime = 0;
    cuda_check(cudaDriverGetVersion(&driver), "Driver version"); cuda_check(cudaRuntimeGetVersion(&runtime), "CUDA version");
    require(runtime >= 11080 && runtime < 12000, "CUDA 11.8 runtime required");
    require(getInferLibVersion() == NV_TENSORRT_VERSION, "TensorRT DLL/header version mismatch");
    Json identity{{"schema", 1}, {"model", manifest}, {"precision", o.precision},
        {"workspace_mib", o.workspace_mib}, {"fp32_layers", o.fp32_layers}, {"gpu", gpu.name},
        {"capability", {gpu.major, gpu.minor}}, {"gpu_memory", gpu.totalGlobalMem},
        {"gpu_uuid", hash_bytes(gpu.uuid.bytes, sizeof(gpu.uuid.bytes))},
        {"driver", driver}, {"cuda_runtime", runtime}, {"tensorrt", getInferLibVersion()},
        {"tensorrt_build", NV_TENSORRT_BUILD}, {"cudnn", cudnn_version()}};
    if (o.precision == "int8" && manifest.value("quantization", "fp32") != "qat-int8")
        identity["calibration_manifest_sha256"] = hash_file(o.calibration_manifest);
    return identity;
}
EngineArtifact engine_artifact(const Options& o, bool build_only) {
    cuda_check(cudaSetDevice(o.gpu), "Select inference GPU");
    auto identity = engine_identity(o); auto key = identity.dump();
    auto path = o.engine.value_or(o.cache / ("yolo1050-" + hash_bytes(key.data(), key.size()).substr(0, 24) + ".engine"));
    if (!valid_artifact(path, identity)) {
        require(build_only, "No matching Pascal engine. Run --build-only on the GTX 1050 with Roblox closed");
        fs::create_directories(path.parent_path()); BuildLock lock(path.string() + ".lock");
        if (!valid_artifact(path, identity)) build(o, identity, path);
    }
    return {path, read_json(path.string() + ".json")};
}
Runner::Runner(const EngineArtifact& artifact, const Geometry& g, const Options& o)
    : geometry(g), timing_(o.stage_timing) {
    try {
        initLibNvInferPlugins(&logger_, "");
        runtime_.reset(nvinfer1::createInferRuntime(logger_)); require(bool(runtime_), "TensorRT runtime creation");
        auto bytes = read_bytes(artifact.path);
        engine_.reset(runtime_->deserializeCudaEngine(bytes.data(), bytes.size()));
        require(bool(engine_), "Cannot deserialize engine");
        require(engine_->getNbIOTensors() == 2, "Expected one image input and one NMS-free output");
        std::string input_name, output_name;
        for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
            auto name = engine_->getIOTensorName(i);
            require(engine_->getTensorDataType(name) == nvinfer1::DataType::kFLOAT &&
                    engine_->getTensorLocation(name) == nvinfer1::TensorLocation::kDEVICE &&
                    engine_->getTensorFormat(name) == nvinfer1::TensorFormat::kLINEAR, "FP32 linear device I/O required");
            if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) input_name = name;
            else output_name = name;
        }
        require(!input_name.empty() && !output_name.empty(), "Missing image or output binding");
        auto input_shape = engine_->getTensorShape(input_name.c_str());
        auto output_shape = engine_->getTensorShape(output_name.c_str());
        require(input_shape.nbDims == 4 && input_shape.d[0] == 1 && input_shape.d[1] == 3 &&
                input_shape.d[2] == g.height && input_shape.d[3] == g.width, "Engine does not match display geometry");
        require(output_shape.nbDims == 3 && output_shape.d[0] == 1 && output_shape.d[1] > 0 &&
                output_shape.d[1] <= 300 && output_shape.d[2] == 6, "Expected [1, <=300, 6] output");
        candidates = output_shape.d[1]; output_bytes_ = size_t(candidates) * 6 * sizeof(float);
        size_t raw_bytes = size_t(g.width) * g.height * 3, free = 0, total = 0;
        cuda_check(cudaMemGetInfo(&free, &total), "Memory headroom");
        size_t estimate = engine_->getDeviceMemorySize() + raw_bytes * 5 + output_bytes_;
        require(free > estimate + (size_t(512) << 20), "Insufficient VRAM; retain at least 512 MiB free headroom");
        std::cout << "Engine context + image buffers estimate: " << estimate / 1048576.0 << " MiB; free "
                  << free / 1048576.0 << " MiB before context allocation (CUDA/library/display memory is additional).\n";
        context_.reset(engine_->createExecutionContext()); require(bool(context_), "Execution context allocation failed");
        cuda_check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "Inference stream");
        cuda_check(cudaEventCreateWithFlags(&done_, cudaEventBlockingSync | cudaEventDisableTiming), "Completion event");
        if (timing_) {
            cuda_check(cudaEventCreate(&start_), "Timing event"); cuda_check(cudaEventCreate(&preprocessed_), "Timing event");
            cuda_check(cudaEventCreate(&inferred_), "Timing event");
            cuda_check(cudaEventCreate(&downloaded_), "Timing event");
        }
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&raw_), raw_bytes), "Raw image allocation");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&input_), raw_bytes * sizeof(float)), "Input allocation");
        auto xs = resize_axis(g.screen_w, g.resized_w), ys = resize_axis(g.screen_h, g.resized_h);
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&xs_), xs.size() * sizeof(ResizeEntry)), "Resize X table");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&ys_), ys.size() * sizeof(ResizeEntry)), "Resize Y table");
        cuda_check(cudaMemcpyAsync(xs_, xs.data(), xs.size() * sizeof(ResizeEntry), cudaMemcpyHostToDevice, stream_), "Resize X upload");
        cuda_check(cudaMemcpyAsync(ys_, ys.data(), ys.size() * sizeof(ResizeEntry), cudaMemcpyHostToDevice, stream_), "Resize Y upload");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&output_), output_bytes_), "Output allocation");
        cuda_check(cudaHostAlloc(reinterpret_cast<void**>(&host_output_), output_bytes_, cudaHostAllocDefault), "Pinned output");
        require(context_->setTensorAddress(input_name.c_str(), input_) &&
                context_->setTensorAddress(output_name.c_str(), output_), "Cannot bind engine buffers");
        cuda_check(cudaMemsetAsync(input_, 0, raw_bytes * sizeof(float), stream_), "Initialize model input");
        require(context_->enqueueV3(stream_), "TensorRT startup initialization failed");
        cuda_check(cudaStreamSynchronize(stream_), "TensorRT startup initialization");
        if (o.graph) {
            auto begun = cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal);
            bool enqueued = begun == cudaSuccess && context_->enqueueV3(stream_);
            auto ended = begun == cudaSuccess ? cudaStreamEndCapture(stream_, &graph_) : begun;
            auto instantiated = ended == cudaSuccess && enqueued && graph_
                ? cudaGraphInstantiate(&executable_, graph_, nullptr, nullptr, 0) : cudaErrorStreamCaptureInvalidated;
            if (instantiated != cudaSuccess) {
                std::cerr << "CUDA graph unavailable; using enqueueV3: " << cudaGetErrorString(instantiated) << '\n';
                if (graph_) cudaGraphDestroy(graph_); graph_ = nullptr; executable_ = nullptr;
                cudaGetLastError(); cuda_check(cudaStreamSynchronize(stream_), "Graph fallback");
            }
        }
        std::cout << "TensorRT ready; CUDA graph " << (executable_ ? "on" : "off") << ".\n";
    } catch (...) { cleanup(); throw; }
}
void Runner::submit(const uint8_t* pinned_bgr, cudaGraphicsResource_t resource) {
    require(!pending_, "Only one inference can be in flight"); pending_ = true;
    if (timing_) cuda_check(cudaEventRecord(start_, stream_), "Timing start");
    if (resource) {
        cuda_check(cudaGraphicsMapResources(1, &resource, stream_), "Map D3D capture"); mapped_ = resource;
        cudaArray_t array = nullptr;
        cuda_check(cudaGraphicsSubResourceGetMappedArray(&array, resource, 0, 0), "Capture array");
        texture_ = preprocess_texture(array, input_, geometry, xs_, ys_, stream_);
    } else {
        require(pinned_bgr != nullptr, "Missing capture buffer");
        cuda_check(cudaMemcpyAsync(raw_, pinned_bgr, size_t(geometry.width) * geometry.height * 3,
                                  cudaMemcpyHostToDevice, stream_), "Image upload");
        preprocess_bgr(raw_, input_, geometry, stream_);
    }
    if (timing_) cuda_check(cudaEventRecord(preprocessed_, stream_), "Preprocessing event");
    if (executable_) cuda_check(cudaGraphLaunch(executable_, stream_), "Inference graph");
    else require(context_->enqueueV3(stream_), "TensorRT enqueue failed");
    if (timing_) cuda_check(cudaEventRecord(inferred_, stream_), "Inference event");
    cuda_check(cudaMemcpyAsync(host_output_, output_, output_bytes_, cudaMemcpyDeviceToHost, stream_), "Detection download");
    if (timing_) cuda_check(cudaEventRecord(downloaded_, stream_), "Download event");
    // Graphics mapping stays outside the TensorRT graph.
    if (mapped_) {
        cuda_check(cudaGraphicsUnmapResources(1, &mapped_, stream_), "Unmap capture"); mapped_ = nullptr;
    }
    cuda_check(cudaEventRecord(done_, stream_), "Completion event");
}
const float* Runner::finish(float& preprocess_ms, float& model_ms, float& transfer_ms) {
    require(pending_, "No inference pending"); cuda_check(cudaEventSynchronize(done_), "Result completion");
    pending_ = false;
    if (texture_) { cudaDestroyTextureObject(texture_); texture_ = 0; }
    preprocess_ms = model_ms = transfer_ms = 0;
    if (timing_) {
        cuda_check(cudaEventElapsedTime(&preprocess_ms, start_, preprocessed_), "Preprocess timing");
        cuda_check(cudaEventElapsedTime(&model_ms, preprocessed_, inferred_), "Model timing");
        cuda_check(cudaEventElapsedTime(&transfer_ms, inferred_, downloaded_), "Download timing");
    }
    return host_output_;
}
void Runner::cleanup() noexcept {
    if (stream_) cudaStreamSynchronize(stream_);
    if (mapped_) { cudaGraphicsUnmapResources(1, &mapped_, stream_); cudaStreamSynchronize(stream_); mapped_ = nullptr; }
    if (texture_) cudaDestroyTextureObject(texture_);
    if (executable_) cudaGraphExecDestroy(executable_);
    if (graph_) cudaGraphDestroy(graph_);
    cudaFreeHost(host_output_); cudaFree(output_); cudaFree(input_); cudaFree(raw_); cudaFree(xs_); cudaFree(ys_);
    if (done_) cudaEventDestroy(done_); if (start_) cudaEventDestroy(start_);
    if (preprocessed_) cudaEventDestroy(preprocessed_); if (inferred_) cudaEventDestroy(inferred_);
    if (downloaded_) cudaEventDestroy(downloaded_);
    if (stream_) cudaStreamDestroy(stream_);
    host_output_ = output_ = input_ = nullptr; raw_ = nullptr; xs_ = ys_ = nullptr; stream_ = nullptr;
    done_ = start_ = preprocessed_ = inferred_ = downloaded_ = nullptr;
    graph_ = nullptr; executable_ = nullptr; texture_ = 0;
    context_.reset(); engine_.reset(); runtime_.reset(); pending_ = false;
}
}
