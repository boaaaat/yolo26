#include "capture.hpp"
#include "controls.hpp"
#include "engine.hpp"
#include <iomanip>
#include <numeric>
#include <cctype>
#include <cstring>
namespace y1050 {
namespace {
std::atomic<State*> active_state{nullptr};
BOOL WINAPI console_control(DWORD event) {
    if (event == CTRL_C_EVENT || event == CTRL_BREAK_EVENT || event == CTRL_CLOSE_EVENT) {
        if (auto state = active_state.load()) state->shutdown = true; return TRUE;
    }
    return FALSE;
}
class ShutdownGuard {
    State& state_;
public:
    explicit ShutdownGuard(State& state) : state_(state) {
        active_state = &state; require(SetConsoleCtrlHandler(console_control, TRUE), "Console shutdown handler");
    }
    ~ShutdownGuard() { state_.pause(); state_.shutdown = true; SetConsoleCtrlHandler(console_control, FALSE); active_state = nullptr; }
};
std::string foreground_process() {
    DWORD pid = 0; GetWindowThreadProcessId(GetForegroundWindow(), &pid);
    HANDLE process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid);
    if (!process) return "unknown";
    wchar_t path[32768]{}; DWORD length = DWORD(std::size(path));
    bool queried = QueryFullProcessImageNameW(process, 0, path, &length) != FALSE;
    CloseHandle(process); if (!queried) return "unknown";
    std::wstring name(path, length); auto slash = name.find_last_of(L"\\/");
    if (slash != std::wstring::npos) name.erase(0, slash + 1);
    int bytes = WideCharToMultiByte(CP_UTF8, 0, name.data(), int(name.size()), nullptr, 0, nullptr, nullptr);
    if (bytes <= 0) return "unknown";
    std::string result(size_t(bytes), '\0');
    WideCharToMultiByte(CP_UTF8, 0, name.data(), int(name.size()), result.data(), bytes, nullptr, nullptr);
    return result;
}
class Telemetry {
    HMODULE library_ = nullptr;
    void* device_ = nullptr;
    using Init = int(*)(); using Shutdown = int(*)(); using Handle = int(*)(const char*, void**);
    using Temperature = int(*)(void*, unsigned, unsigned*); using Clocks = int(*)(void*, unsigned, unsigned*);
    Shutdown shutdown_ = nullptr; Temperature temperature_ = nullptr; Clocks clocks_ = nullptr;
    bool initialized_ = false;
public:
    explicit Telemetry(int gpu) {
        wchar_t directory[MAX_PATH]{};
        if (!GetSystemDirectoryW(directory, MAX_PATH)) return;
        library_ = LoadLibraryW((fs::path(directory) / "nvml.dll").c_str()); if (!library_) return;
        auto init = reinterpret_cast<Init>(GetProcAddress(library_, "nvmlInit_v2"));
        shutdown_ = reinterpret_cast<Shutdown>(GetProcAddress(library_, "nvmlShutdown"));
        auto handle = reinterpret_cast<Handle>(GetProcAddress(library_, "nvmlDeviceGetHandleByPciBusId_v2"));
        temperature_ = reinterpret_cast<Temperature>(GetProcAddress(library_, "nvmlDeviceGetTemperature"));
        clocks_ = reinterpret_cast<Clocks>(GetProcAddress(library_, "nvmlDeviceGetClockInfo"));
        if (!init || !shutdown_ || !handle || init() != 0) return;
        initialized_ = true; char pci[32]{};
        if (cudaDeviceGetPCIBusId(pci, sizeof(pci), gpu) != cudaSuccess || handle(pci, &device_) != 0) device_ = nullptr;
    }
    ~Telemetry() { if (initialized_) shutdown_(); if (library_) FreeLibrary(library_); }
    void report() {
        size_t free = 0, total = 0; cuda_check(cudaMemGetInfo(&free, &total), "Runtime memory telemetry");
        std::cout << "GPU VRAM used/free (all processes): " << (total - free) / 1048576.0 << "/"
                  << free / 1048576.0 << " MiB";
        unsigned temperature = 0, clock = 0;
        if (device_ && temperature_ && temperature_(device_, 0, &temperature) == 0) std::cout << "; " << temperature << " C";
        if (device_ && clocks_ && clocks_(device_, 1, &clock) == 0) std::cout << "; SM " << clock << " MHz";
        std::cout << "; foreground " << foreground_process() << ".\n";
    }
};
struct Statistics {
    uint64_t completed = 0, published = 0, stale = 0;
    uint64_t enemy_frames = 0, target_frames = 0, malformed = 0, invalid_boxes = 0;
    std::array<uint64_t, 3> detections{};
    std::array<float, 3> max_score{};
    std::vector<double> ages;
    double capture = 0, resize = 0, completion = 0, preprocess_gpu = 0, model_gpu = 0, download_gpu = 0;
    Statistics() { ages.reserve(4096); }
    void reset() {
        completed = published = stale = 0; ages.clear();
        enemy_frames = target_frames = malformed = invalid_boxes = 0; detections = {}; max_score = {};
        capture = resize = completion = preprocess_gpu = model_gpu = download_gpu = 0;
    }
    void observe(const DetectionSummary& summary, int enemy) {
        for (size_t i = 0; i < detections.size(); ++i) {
            detections[i] += summary.above_threshold[i];
            max_score[i] = std::max(max_score[i], summary.max_score[i]);
        }
        enemy_frames += summary.above_threshold[enemy] > 0; target_frames += summary.target_selected;
        malformed += summary.malformed; invalid_boxes += summary.invalid_boxes;
    }
    void report(double elapsed, CaptureCounts before, CaptureCounts after, const Options& options) {
        std::cout << std::fixed << std::setprecision(1)
            << "Fresh desktop inference " << completed / elapsed << " FPS; published " << published / elapsed
            << " FPS; capture " << (after.captured - before.captured) / elapsed
            << " FPS; superseded/dropped " << after.dropped - before.dropped
            << "; capture timeouts " << after.timeouts - before.timeouts
            << "; pointer-only " << after.pointer_only - before.pointer_only
            << "; repeated presents " << after.repeated - before.repeated << "; stale/disarmed " << stale << ".\n";
        std::cout << std::setprecision(3) << "Detections in interval (confidence >= " << options.confidence
            << "): dead " << detections[0] << ", enemy " << detections[1] << ", teammate " << detections[2]
            << "; frames with enemy " << enemy_frames << '/' << completed
            << "; target selected " << target_frames << '/' << published
            << "; max scores dead/enemy/teammate " << max_score[0] << '/' << max_score[1] << '/' << max_score[2]
            << "; malformed rows " << malformed << ", invalid boxes " << invalid_boxes << ".\n";
        if (!ages.empty()) {
            double mean = std::accumulate(ages.begin(), ages.end(), 0.0) / ages.size();
            std::sort(ages.begin(), ages.end());
            double p95 = ages[std::min(ages.size() - 1, size_t(std::ceil(ages.size() * .95) - 1))];
            std::cout << std::setprecision(2) << "Mean ms: acquire " << capture / completed
                << ", capture copy/CPU resize " << resize / completed << ", transfer+GPU+result " << completion / completed
                << "; capture-start to publication " << mean << " (p95 " << p95 << ")";
            if (options.stage_timing) std::cout << "; GPU upload/preprocess " << preprocess_gpu / completed
                                 << ", model " << model_gpu / completed << ", result download " << download_gpu / completed;
            std::cout << ".\n";
        }
    }
};
std::string lower(std::string value) {
    for (auto& c : value) c = char(std::tolower(static_cast<unsigned char>(c))); return value;
}
void run(const Options& options, const EngineArtifact& artifact) {
    SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    Geometry geometry = Geometry::make(GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN), options.edge);
    require(artifact.metadata.at("export").at("input_shape") == Json::array({1, 3, geometry.height, geometry.width}),
            "Export shape does not fit the current primary display. Prepare a new export, rebuild, and recalibrate");
    auto names = artifact.metadata.at("export").at("names").get<std::vector<std::string>>();
    int enemy = -1;
    for (int i = 0; i < int(names.size()); ++i) if (lower(names[i]) == lower(options.enemy_name)) {
        require(enemy == -1, "Multiple enemy classes"); enemy = i;
    }
    require(enemy >= 0, "Enemy class not found");
    std::cout << "Runtime update v6; engine precision " << artifact.metadata.at("identity").at("precision").get<std::string>()
              << "; confidence " << options.confidence << "; controls " << (options.controls_enabled ? "on" : "off")
              << "; enemy class ID " << enemy << ".\n";
    std::cout << "Full display " << geometry.screen_w << 'x' << geometry.screen_h << " -> " << geometry.width
              << 'x' << geometry.height << "; 1024 long edge, no crop.\n";
    bool qualified = options.qualification.is_object() &&
                     options.qualification.value("engine_sha256", "") == artifact.metadata.at("engine_sha256").get<std::string>();
    std::cout << (qualified ? "Qualification recorded from user-supplied measurements.\n" :
                              "Performance/accuracy qualification is pending; 60 FPS and game impact are unverified.\n");
    State state; load_calibration(state, options, geometry);
    std::cout << "Calibrated locked cursor: " << state.locked_x << ',' << state.locked_y << ".\n";
    // Destruction order keeps GPU resources alive until in-flight reads have completed.
    Capture capture(options, state, geometry);
    Runner runner(artifact, geometry, options);
    Controls controls(state, options);
    std::unique_ptr<Overlay> overlay;
    if (options.overlay) overlay = std::make_unique<Overlay>(state, options, geometry, names);
    ShutdownGuard shutdown(state);
    Telemetry telemetry(options.gpu);
    capture.start(); controls.start(); if (overlay) overlay->start();
    Deadline timer;
    auto period = std::chrono::nanoseconds(1'000'000'000 / options.inference_hz);
    auto next = Clock::now(), reporting = next; CaptureCounts previous = capture.counts();
    Statistics statistics;
    std::cout << "Ready. Press = to arm, - to pause, Ctrl+C to exit. Target " << options.inference_hz
              << " fresh FPS; overlay " << (options.overlay ? "requested" : "off") << ", recording/collection off.\n";
    while (!state.shutdown) {
        capture.check(); controls.check(); if (overlay) overlay->check();
        if (!state.running) {
            capture.discard(); std::this_thread::sleep_for(std::chrono::milliseconds(5));
            next = reporting = Clock::now(); previous = capture.counts(); statistics.reset(); continue;
        }
        timer.wait(next, state.shutdown, &state.running); if (!state.running || state.shutdown) continue;
        auto slot = capture.take();
        if (slot) {
            uint64_t generation;
            { std::lock_guard<std::mutex> lock(state.mutex); generation = state.arm_generation; }
            auto submitted = Clock::now();
            if (!state.running || slot->generation != generation || seconds(submitted - slot->captured) > options.max_age) {
                ++statistics.stale; capture.release(slot);
            } else {
                // Preserve the 60 Hz timeline across small wakeup jitter; skip missed ticks after an overrun.
                next = submitted - next >= period ? submitted + period : next + period;
                // On failure no capture slot is released until Runner destruction drains its stream.
                runner.submit(slot->pinned, slot->resource);
                float preprocessing = 0, model = 0, transfer = 0;
                auto output = runner.finish(preprocessing, model, transfer); auto completed = Clock::now();
                ++statistics.completed;
                DetectionSummary summary;
                bool accepted = publish(state, options, geometry, output, runner.candidates, enemy,
                                        slot->captured, slot->generation, summary);
                statistics.observe(summary, enemy);
                if (accepted) ++statistics.published; else ++statistics.stale;
                auto publication = Clock::now();
                statistics.capture += slot->capture_ms; statistics.resize += slot->resize_ms;
                statistics.completion += milliseconds(completed - submitted);
                statistics.preprocess_gpu += preprocessing; statistics.model_gpu += model;
                statistics.download_gpu += transfer;
                if (accepted && statistics.ages.size() < 4096)
                    statistics.ages.push_back(milliseconds(publication - slot->captured));
                capture.release(slot);
            }
        }
        auto now = Clock::now();
        if (seconds(now - reporting) >= options.report_seconds) {
            auto current = capture.counts(); statistics.report(seconds(now - reporting), previous, current, options);
            telemetry.report(); previous = current; reporting = now; statistics.reset();
        }
    }
    capture.check(); controls.check(); if (overlay) overlay->check();
}
}
}
int wmain(int argc, wchar_t** argv) {
    try {
        y1050::fs::path config; bool build_only = false, view_detections = false; std::string precision;
        for (int i = 1; i < argc; ++i) {
            std::wstring arg = argv[i];
            if (arg == L"--config" && i + 1 < argc) config = argv[++i];
            else if (arg == L"--precision" && i + 1 < argc) {
                std::wstring p = argv[++i];
                y1050::require(p == L"fp32" || p == L"int8", "--precision must be fp32 or int8");
                precision = p == L"fp32" ? "fp32" : "int8";
            } else if (arg == L"--build-only") build_only = true;
            else if (arg == L"--view-detections") view_detections = true;
            else if (arg == L"--help" || arg == L"-h") {
                std::cout << "yolo1050.exe --config settings.json [--precision fp32|int8] [--build-only] [--view-detections]\n"
                             "Build engines on the GTX 1050 with Roblox closed. Live startup never builds engines.\n"; return 0;
            } else throw std::runtime_error("Unknown/incomplete argument");
        }
        y1050::require(!config.empty(), "--config is required; paths resolve relative to the configuration file");
        auto options = y1050::Options::load(config);
        if (!precision.empty()) {
            y1050::require(precision == "fp32" || precision == "int8", "--precision must be fp32 or int8");
            options.precision = precision;
        }
        y1050::require(!build_only || !view_detections, "--view-detections is a live display option, not an engine build option");
        if (view_detections) {
            options.overlay = true; options.stage_timing = true;
            options.controls_enabled = false; options.auto_shoot = false;
        }
        // Normal priority; no busy-spin CUDA waiting or multiple inference contexts.
        y1050::require(SetPriorityClass(GetCurrentProcess(), NORMAL_PRIORITY_CLASS), "Normal process priority");
        y1050::cuda_check(cudaSetDevice(options.gpu), "Select inference GPU");
        y1050::cuda_check(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync), "Blocking CUDA scheduling");
        auto artifact = y1050::engine_artifact(options, build_only);
        std::cout << "Engine: " << artifact.path.string() << '\n';
        if (!build_only) y1050::run(options, artifact);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "yolo1050: " << error.what() << "\nStopped. No resolution reduction, tracking substitution, or alternate inference backend.\n";
        return 1;
    }
}
