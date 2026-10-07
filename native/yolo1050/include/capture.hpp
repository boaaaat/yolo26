#pragma once
#include "common.hpp"
#include <d3d11.h>
#include <dxgi1_2.h>
#include <wrl/client.h>
#include <opencv2/core.hpp>
namespace y1050 {
using Microsoft::WRL::ComPtr;
struct CaptureSlot {
    enum Status { free, writing, ready, processing } status = free;
    ComPtr<ID3D11Texture2D> texture;
    cudaGraphicsResource_t resource = nullptr;
    uint8_t* pinned = nullptr;
    cv::Mat padded;
    uint64_t generation = 0, sequence = 0;
    Clock::time_point captured{};
    double capture_ms = 0, resize_ms = 0;
};
struct CaptureCounts {
    uint64_t captured = 0, dropped = 0, timeouts = 0, pointer_only = 0, repeated = 0, interruptions = 0;
};
class Capture {
    const Options& options_;
    State& state_;
    Geometry geometry_;
    ComPtr<IDXGIAdapter1> adapter_;
    ComPtr<IDXGIOutput1> output_;
    ComPtr<ID3D11Device> device_;
    ComPtr<ID3D11DeviceContext> context_;
    ComPtr<IDXGIOutputDuplication> duplication_;
    ComPtr<ID3D11Texture2D> staging_;
    std::vector<ResizeEntry> xs_, ys_;
    std::array<CaptureSlot, 3> slots_;
    std::mutex mutex_;
    std::condition_variable condition_;
    CaptureSlot* latest_ = nullptr;
    CaptureCounts counts_;
    std::exception_ptr error_;
    std::thread worker_;
    std::atomic<bool> stopping_{false};
    bool interop_ = false;
    int64_t last_present_ = 0;
    void initialize();
    bool acquire(CaptureSlot&, Clock::time_point);
    void run() noexcept;
    void cleanup() noexcept;
public:
    Capture(const Options&, State&, const Geometry&);
    ~Capture();
    Capture(const Capture&) = delete;
    Capture& operator=(const Capture&) = delete;
    void start();
    CaptureSlot* take();
    void release(CaptureSlot*);
    void discard();
    CaptureCounts counts();
    void check();
    void close();
    bool interop() const { return interop_; }
};
}
