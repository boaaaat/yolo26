#include "capture.hpp"
#include <cuda_d3d11_interop.h>
#include <opencv2/imgproc.hpp>
#include <emmintrin.h>
namespace y1050 {
namespace {
struct AcquiredFrame {
    IDXGIOutputDuplication* duplication = nullptr;
    bool active = false;
    ~AcquiredFrame() { if (active) duplication->ReleaseFrame(); }
};
std::string adapter_name(const DXGI_ADAPTER_DESC1& description) {
    int length = WideCharToMultiByte(CP_UTF8, 0, description.Description, -1, nullptr, 0, nullptr, nullptr);
    if (length <= 0) return "unknown adapter";
    std::string name(size_t(length), '\0');
    WideCharToMultiByte(CP_UTF8, 0, description.Description, -1, name.data(), length, nullptr, nullptr);
    name.pop_back(); return name;
}
// SSE2 is part of Windows x64's baseline. Keep the calibration's integer
// coefficients and rounding: vectorize the four BGRA channels, then retain BGR.
__m128i multiply_low32(__m128i values, __m128i weights) {
    auto even = _mm_mul_epu32(values, weights);
    auto odd = _mm_mul_epu32(_mm_srli_si128(values, 4), weights);
    return _mm_unpacklo_epi32(_mm_shuffle_epi32(even, _MM_SHUFFLE(2, 0, 2, 0)),
                             _mm_shuffle_epi32(odd, _MM_SHUFFLE(2, 0, 2, 0)));
}
__m128i horizontal_bgra(const uint8_t* row, const ResizeEntry& x, __m128i weights) {
    uint32_t first = 0, second = 0;
    std::memcpy(&first, row + size_t(x.first) * 4, sizeof(first));
    std::memcpy(&second, row + size_t(x.second) * 4, sizeof(second));
    auto zero = _mm_setzero_si128();
    auto a = _mm_unpacklo_epi8(_mm_cvtsi32_si128(static_cast<int>(first)), zero);
    auto b = _mm_unpacklo_epi8(_mm_cvtsi32_si128(static_cast<int>(second)), zero);
    return _mm_madd_epi16(_mm_unpacklo_epi16(a, b), weights);
}
void resize_bgra_pinned(const uint8_t* source, size_t pitch, uint8_t* destination,
                        const Geometry& g, const std::vector<ResizeEntry>& xs,
                        const std::vector<ResizeEntry>& ys) {
    auto rounding = _mm_set1_epi32(1 << 21), zero = _mm_setzero_si128();
    for (int y = 0; y < g.resized_h; ++y) {
        const auto& ry = ys[y];
        auto row0 = source + size_t(ry.first) * pitch;
        auto row1 = source + size_t(ry.second) * pitch;
        auto dst = destination + (size_t(y + g.top) * g.width + g.left) * 3;
        auto y0 = _mm_set1_epi32(ry.a0), y1 = _mm_set1_epi32(ry.a1);
        for (int x = 0; x < g.resized_w; ++x) {
            const auto& rx = xs[x];
            auto weights = _mm_set1_epi32(rx.a0 | (rx.a1 << 16));
            auto upper = horizontal_bgra(row0, rx, weights);
            auto lower = horizontal_bgra(row1, rx, weights);
            auto value = _mm_srli_epi32(_mm_add_epi32(rounding,
                _mm_add_epi32(multiply_low32(upper, y0), multiply_low32(lower, y1))), 22);
            auto bytes = _mm_packus_epi16(_mm_packs_epi32(value, zero), zero);
            auto pixel = static_cast<uint32_t>(_mm_cvtsi128_si32(bytes));
            dst[x * 3] = static_cast<uint8_t>(pixel);
            dst[x * 3 + 1] = static_cast<uint8_t>(pixel >> 8);
            dst[x * 3 + 2] = static_cast<uint8_t>(pixel >> 16);
        }
    }
}
}
Capture::Capture(const Options& options, State& state, const Geometry& geometry)
    : options_(options), state_(state), geometry_(geometry) {
    try { initialize(); } catch (...) { cleanup(); throw; }
}
void Capture::initialize() {
    cuda_check(cudaSetDevice(options_.gpu), "Capture inference GPU");
    ComPtr<IDXGIFactory1> factory; hr_check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "DXGI factory");
    bool found = false; HRESULT capture_error = S_OK;
    const char* failed_operation = "Desktop Duplication";
    for (UINT ai = 0; !found; ++ai) {
        ComPtr<IDXGIAdapter1> adapter; HRESULT result = factory->EnumAdapters1(ai, &adapter);
        if (result == DXGI_ERROR_NOT_FOUND) break; hr_check(result, "Enumerate adapters");
        if (options_.capture_adapter >= 0 && ai != UINT(options_.capture_adapter)) continue;
        DXGI_ADAPTER_DESC1 adapter_description{};
        hr_check(adapter->GetDesc1(&adapter_description), "Capture adapter description");
        if (adapter_description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
        auto name = adapter_name(adapter_description);
        for (UINT oi = 0; ; ++oi) {
            ComPtr<IDXGIOutput> output; result = adapter->EnumOutputs(oi, &output);
            if (result == DXGI_ERROR_NOT_FOUND) break; hr_check(result, "Enumerate outputs");
            if (options_.capture_output >= 0 && oi != UINT(options_.capture_output)) continue;
            DXGI_OUTPUT_DESC desc{}; hr_check(output->GetDesc(&desc), "Display description");
            if (!desc.AttachedToDesktop || desc.DesktopCoordinates.left != 0 || desc.DesktopCoordinates.top != 0 ||
                desc.DesktopCoordinates.right != geometry_.screen_w || desc.DesktopCoordinates.bottom != geometry_.screen_h)
                continue;
            require(desc.Rotation == DXGI_MODE_ROTATION_IDENTITY, "Rotated displays require an explicit rotation implementation");
            // Hybrid drivers can expose the primary desktop through more than one
            // adapter. A matching rectangle does not prove DuplicateOutput works.
            // Own candidate resources locally until the display can be duplicated.
            ComPtr<IDXGIOutput1> candidate_output;
            hr_check(output.As(&candidate_output), "DXGI output interface");
            ComPtr<ID3D11Device> candidate_device;
            ComPtr<ID3D11DeviceContext> candidate_context;
            ComPtr<IDXGIOutputDuplication> candidate_duplication;
            D3D_FEATURE_LEVEL feature{};
            result = D3D11CreateDevice(adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr,
                D3D11_CREATE_DEVICE_BGRA_SUPPORT, nullptr, 0, D3D11_SDK_VERSION,
                &candidate_device, &feature, &candidate_context);
            failed_operation = "D3D11 capture device";
            if (SUCCEEDED(result)) {
                failed_operation = "Desktop Duplication";
                result = candidate_output->DuplicateOutput(candidate_device.Get(), &candidate_duplication);
            }
            if (FAILED(result)) {
                capture_error = result;
                std::cerr << "Capture candidate adapter " << ai << " (" << name << "), output " << oi
                          << ": " << failed_operation << " failed, HRESULT 0x" << std::hex
                          << static_cast<uint32_t>(result) << std::dec;
                if (result == DXGI_ERROR_UNSUPPORTED) std::cerr << " (DXGI_ERROR_UNSUPPORTED)";
                std::cerr << ". Trying other matching display outputs.\n";
                continue;
            }
            adapter_ = adapter; output_ = candidate_output; device_ = candidate_device;
            context_ = candidate_context; duplication_ = candidate_duplication; found = true;
            std::cout << "Capture adapter " << ai << " (" << name << "), output " << oi
                      << " (calibrated primary display). D3D feature level 0x" << std::hex
                      << static_cast<unsigned>(feature) << std::dec << ".\n"; break;
        }
    }
    if (!found && capture_error == DXGI_ERROR_UNSUPPORTED)
        throw std::runtime_error("Desktop Duplication is unsupported on the matching display adapters (0x887A0004). "
            "On an Intel/NVIDIA hybrid laptop, close this program, set runtime/yolo1050.exe to Power saving "
            "(Intel graphics) in Windows Settings > System > Display > Graphics, then restart step 3. "
            "CUDA inference still uses the configured NVIDIA GPU. Use a local Windows session with Roblox "
            "windowed/borderless; if this persists, check the Intel display driver. Do not rebuild engines for this capture error.");
    if (!found && FAILED(capture_error)) hr_check(capture_error, failed_operation);
    require(found, "No capture output matches the calibrated primary display");
    DXGI_OUTDUPL_DESC description{}; duplication_->GetDesc(&description);
    require(description.ModeDesc.Width == UINT(geometry_.screen_w) &&
            description.ModeDesc.Height == UINT(geometry_.screen_h) &&
            description.ModeDesc.Format == DXGI_FORMAT_B8G8R8A8_UNORM, "Unsupported desktop capture geometry/format");
    int cuda_device = -1;
    cudaDeviceProp inference_gpu{};
    cuda_check(cudaGetDeviceProperties(&inference_gpu, options_.gpu), "Capture inference GPU properties");
    std::cout << "Inference GPU " << options_.gpu << " (" << inference_gpu.name << ") via CUDA.\n";
    auto mapping = cudaD3D11GetDevice(&cuda_device, adapter_.Get());
    interop_ = mapping == cudaSuccess && cuda_device == options_.gpu && options_.capture_backend != "cpu";
    if (mapping != cudaSuccess) cudaGetLastError();
    require(options_.capture_backend != "cuda" || interop_, "Requested CUDA capture is unavailable on this display adapter");
    D3D11_TEXTURE2D_DESC desc{}; desc.Width = geometry_.screen_w; desc.Height = geometry_.screen_h;
    desc.MipLevels = 1; desc.ArraySize = 1; desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    desc.SampleDesc.Count = 1; desc.Usage = D3D11_USAGE_DEFAULT; desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    for (auto& slot : slots_) {
        hr_check(device_->CreateTexture2D(&desc, nullptr, &slot.texture), "Owned capture texture");
        if (interop_) {
            auto error = cudaGraphicsD3D11RegisterResource(&slot.resource, slot.texture.Get(), cudaGraphicsRegisterFlagsNone);
            if (error == cudaSuccess) error = cudaGraphicsResourceSetMapFlags(slot.resource, cudaGraphicsMapFlagsReadOnly);
            if (error != cudaSuccess) {
                if (options_.capture_backend == "cuda") cuda_check(error, "Register CUDA capture resource");
                std::cerr << "CUDA texture registration unavailable (" << cudaGetErrorString(error)
                          << "); using pinned CPU capture.\n"; cudaGetLastError(); interop_ = false;
            }
        }
    }
    if (!interop_) {
        for (auto& slot : slots_) {
            if (slot.resource) { cudaGraphicsUnregisterResource(slot.resource); slot.resource = nullptr; }
            slot.texture.Reset();
            size_t bytes = size_t(geometry_.width) * geometry_.height * 3;
            cuda_check(cudaHostAlloc(reinterpret_cast<void**>(&slot.pinned), bytes, cudaHostAllocDefault), "Pinned capture pool");
            slot.padded = cv::Mat(geometry_.height, geometry_.width, CV_8UC3, slot.pinned);
            slot.padded.setTo(cv::Scalar(114, 114, 114));
        }
        desc.Usage = D3D11_USAGE_STAGING; desc.BindFlags = 0; desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        hr_check(device_->CreateTexture2D(&desc, nullptr, &staging_), "Desktop readback texture");
        xs_ = resize_axis(geometry_.screen_w, geometry_.resized_w);
        ys_ = resize_axis(geometry_.screen_h, geometry_.resized_h);
        std::cout << "Capture: D3D readback -> SSE2 CPU bilinear resize -> pinned BGR -> CUDA. "
                     "No direct CUDA access to the selected display.\n";
    } else std::cout << "Capture: owned D3D11 textures -> fused CUDA resize/conversion; no full-frame CPU readback.\n";
    cv::setNumThreads(1);
}
bool Capture::acquire(CaptureSlot& slot, Clock::time_point started) {
    DXGI_OUTDUPL_FRAME_INFO info{}; ComPtr<IDXGIResource> frame;
    auto result = duplication_->AcquireNextFrame(5, &info, &frame);
    if (result == DXGI_ERROR_WAIT_TIMEOUT) {
        std::lock_guard<std::mutex> lock(mutex_); ++counts_.timeouts; return false;
    }
    if (result == DXGI_ERROR_ACCESS_LOST) {
        state_.pause(); discard(); duplication_.Reset(); last_present_ = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_); ++counts_.interruptions;
        }
        std::cerr << "Desktop capture was interrupted. Paused; press = after display capture recovers.\n";
        require(GetSystemMetrics(SM_CXSCREEN) == geometry_.screen_w &&
                GetSystemMetrics(SM_CYSCREEN) == geometry_.screen_h, "Display changed; restart and recalibrate");
        hr_check(output_->DuplicateOutput(device_.Get(), &duplication_), "Restore Desktop Duplication");
        return false;
    }
    hr_check(result, "Acquire desktop frame"); AcquiredFrame release{duplication_.Get(), true};
    // Pointer-only notifications and already acquired desktop presents are not fresh detector frames.
    if (!info.LastPresentTime.QuadPart) {
        std::lock_guard<std::mutex> lock(mutex_); ++counts_.pointer_only; return false;
    }
    if (info.LastPresentTime.QuadPart == last_present_) {
        std::lock_guard<std::mutex> lock(mutex_); ++counts_.repeated; return false;
    }
    last_present_ = info.LastPresentTime.QuadPart;
    ComPtr<ID3D11Texture2D> texture; hr_check(frame.As(&texture), "Captured texture");
    D3D11_TEXTURE2D_DESC actual{}; texture->GetDesc(&actual);
    require(actual.Width == UINT(geometry_.screen_w) && actual.Height == UINT(geometry_.screen_h),
            "Display geometry changed; restart and recalibrate");
    auto acquired = Clock::now();
    if (interop_) {
        context_->CopyResource(slot.texture.Get(), texture.Get()); context_->Flush();
    } else {
        context_->CopyResource(staging_.Get(), texture.Get());
        D3D11_MAPPED_SUBRESOURCE mapped{};
        hr_check(context_->Map(staging_.Get(), 0, D3D11_MAP_READ, 0, &mapped), "Map desktop readback");
        try {
            // Resize BGRA straight into the padded pinned BGR slot, without a full-frame CPU conversion.
            resize_bgra_pinned(static_cast<const uint8_t*>(mapped.pData), mapped.RowPitch,
                               slot.pinned, geometry_, xs_, ys_);
        } catch (...) { context_->Unmap(staging_.Get(), 0); throw; }
        context_->Unmap(staging_.Get(), 0);
    }
    slot.capture_ms = milliseconds(acquired - started); slot.resize_ms = milliseconds(Clock::now() - acquired);
    slot.captured = started; return true;
}
void Capture::start() { require(!worker_.joinable(), "Capture already started"); worker_ = std::thread(&Capture::run, this); }
void Capture::run() noexcept {
    try {
        cuda_check(cudaSetDevice(options_.gpu), "Capture worker GPU"); Deadline timer;
        auto period = std::chrono::nanoseconds(1'000'000'000 / options_.capture_hz);
        auto next = Clock::now();
        while (!stopping_ && !state_.shutdown) {
            if (!state_.running) { discard(); std::this_thread::sleep_for(std::chrono::milliseconds(5)); next = Clock::now(); continue; }
            timer.wait(next, stopping_, &state_.running); if (stopping_ || !state_.running) continue;
            auto started = Clock::now();
            next = started - next >= period ? started + period : next + period;
            uint64_t generation;
            { std::lock_guard<std::mutex> lock(state_.mutex); generation = state_.arm_generation; }
            CaptureSlot* slot = nullptr;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                for (auto& item : slots_) if (item.status == CaptureSlot::free) { slot = &item; item.status = CaptureSlot::writing; break; }
                if (!slot) ++counts_.dropped;
            }
            if (!slot) continue;
            bool fresh = acquire(*slot, started);
            std::lock_guard<std::mutex> lock(mutex_);
            if (!fresh || !state_.running) { slot->status = CaptureSlot::free; continue; }
            slot->generation = generation; slot->sequence = ++counts_.captured;
            if (latest_) { latest_->status = CaptureSlot::free; ++counts_.dropped; }
            latest_ = slot; slot->status = CaptureSlot::ready; condition_.notify_one();
        }
    } catch (...) {
        { std::lock_guard<std::mutex> lock(mutex_); error_ = std::current_exception(); condition_.notify_all(); }
        state_.pause(); state_.shutdown = true;
    }
}
CaptureSlot* Capture::take() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait_for(lock, std::chrono::milliseconds(5), [&] { return latest_ || error_ || stopping_; });
    if (error_) std::rethrow_exception(error_);
    auto slot = latest_; latest_ = nullptr; if (slot) slot->status = CaptureSlot::processing; return slot;
}
void Capture::release(CaptureSlot* slot) {
    std::lock_guard<std::mutex> lock(mutex_); slot->status = CaptureSlot::free; condition_.notify_one();
}
void Capture::discard() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (latest_) { latest_->status = CaptureSlot::free; latest_ = nullptr; ++counts_.dropped; }
}
CaptureCounts Capture::counts() { std::lock_guard<std::mutex> lock(mutex_); return counts_; }
void Capture::check() { std::lock_guard<std::mutex> lock(mutex_); if (error_) std::rethrow_exception(error_); }
void Capture::close() {
    stopping_ = true; condition_.notify_all(); if (worker_.joinable()) worker_.join();
}
void Capture::cleanup() noexcept {
    for (auto& slot : slots_) {
        if (slot.resource) cudaGraphicsUnregisterResource(slot.resource);
        if (slot.pinned) cudaFreeHost(slot.pinned);
        slot.resource = nullptr; slot.pinned = nullptr;
    }
}
Capture::~Capture() { close(); cleanup(); }
}
