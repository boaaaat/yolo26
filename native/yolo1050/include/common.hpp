#pragma once
#include <windows.h>
#include <bcrypt.h>
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace y1050 {
namespace fs = std::filesystem;
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
inline double seconds(Clock::duration d) { return std::chrono::duration<double>(d).count(); }
inline double milliseconds(Clock::duration d) { return seconds(d) * 1000.0; }
inline void require(bool ok, const std::string& message) { if (!ok) throw std::runtime_error(message); }
inline void cuda_check(cudaError_t code, const char* operation) {
    if (code != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(code));
}
inline void hr_check(HRESULT code, const char* operation) {
    if (FAILED(code)) throw std::runtime_error(std::string(operation) + " failed, HRESULT " + std::to_string(code));
}
inline std::vector<char> read_bytes(const fs::path& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    require(bool(in), "Cannot read " + path.string());
    auto size = in.tellg();
    require(size >= 0, "Cannot size " + path.string());
    std::vector<char> bytes(static_cast<size_t>(size));
    in.seekg(0); if (!bytes.empty()) in.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    require(bool(in), "Incomplete read: " + path.string()); return bytes;
}
inline Json read_json(const fs::path& path) {
    auto bytes = read_bytes(path); return Json::parse(bytes.begin(), bytes.end());
}
inline void write_bytes(const fs::path& path, const void* data, size_t size) {
    fs::create_directories(path.parent_path());
    fs::path temporary = path; temporary += ".tmp";
    std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
    require(bool(out), "Cannot write " + temporary.string());
    out.write(static_cast<const char*>(data), static_cast<std::streamsize>(size)); out.close();
    require(bool(out), "Incomplete write: " + temporary.string());
    require(MoveFileExW(temporary.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH),
            "Cannot publish " + path.string());
}
inline void write_json(const fs::path& path, const Json& value) {
    auto text = value.dump(2); write_bytes(path, text.data(), text.size());
}
inline std::string hash_bytes(const void* data, size_t size) {
    BCRYPT_ALG_HANDLE algorithm = nullptr; BCRYPT_HASH_HANDLE hash = nullptr;
    require(BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0) >= 0, "SHA256 provider");
    DWORD object_size = 0, returned = 0;
    if (BCryptGetProperty(algorithm, BCRYPT_OBJECT_LENGTH, reinterpret_cast<PUCHAR>(&object_size),
                          sizeof(object_size), &returned, 0) < 0) {
        BCryptCloseAlgorithmProvider(algorithm, 0); throw std::runtime_error("SHA256 object length");
    }
    std::vector<UCHAR> object(object_size); std::array<UCHAR, 32> digest{};
    bool ok = BCryptCreateHash(algorithm, &hash, object.data(), object_size, nullptr, 0, 0) >= 0;
    auto input = static_cast<const UCHAR*>(data);
    while (ok && size) {
        auto chunk = static_cast<ULONG>(std::min<size_t>(size, 1024 * 1024));
        ok = BCryptHashData(hash, const_cast<PUCHAR>(input), chunk, 0) >= 0;
        input += chunk; size -= chunk;
    }
    if (ok) ok = BCryptFinishHash(hash, digest.data(), static_cast<ULONG>(digest.size()), 0) >= 0;
    if (hash) BCryptDestroyHash(hash); BCryptCloseAlgorithmProvider(algorithm, 0);
    require(ok, "SHA256 hashing failed");
    const char* hex = "0123456789abcdef"; std::string result;
    for (auto byte : digest) { result += hex[byte >> 4]; result += hex[byte & 15]; } return result;
}
inline std::string hash_file(const fs::path& path) {
    auto bytes = read_bytes(path); return hash_bytes(bytes.data(), bytes.size());
}
inline fs::path resolve_path(const fs::path& root, const std::string& value) {
    fs::path path = fs::u8path(value); return fs::absolute(path.is_absolute() ? path : root / path).lexically_normal();
}
struct Geometry {
    int screen_w = 0, screen_h = 0, width = 0, height = 0, resized_w = 0, resized_h = 0, left = 0, top = 0;
    float scale = 0;
    static Geometry make(int w, int h, int edge) {
        require(w > 0 && h > 0 && edge == 1024, "Full-screen 1024 long edge is required");
        Geometry g; g.screen_w = w; g.screen_h = h; g.scale = float(edge) / std::max(w, h);
        // Match Python's round-to-even geometry for odd display dimensions.
        g.resized_w = int(std::nearbyint(w * double(edge) / std::max(w, h)));
        g.resized_h = int(std::nearbyint(h * double(edge) / std::max(w, h)));
        g.width = (g.resized_w + 31) / 32 * 32; g.height = (g.resized_h + 31) / 32 * 32;
        g.left = (g.width - g.resized_w) / 2; g.top = (g.height - g.resized_h) / 2; return g;
    }
};
struct ResizeEntry { int first, second, a0, a1; };
inline std::vector<ResizeEntry> resize_axis(int source, int destination) {
    std::vector<ResizeEntry> entries; entries.reserve(destination);
    for (int i = 0; i < destination; ++i) {
        double position = (i + .5) * source / destination - .5;
        int first = int(std::floor(position)); double fraction = position - first;
        if (first < 0) { first = 0; fraction = 0; }
        if (first >= source - 1) { first = source - 1; fraction = 0; }
        int a1 = int(std::nearbyint(fraction * 2048));
        entries.push_back({first, std::min(first + 1, source - 1), 2048 - a1, a1});
    }
    return entries;
}
struct Options {
    fs::path root, export_manifest, calibration_manifest, cache, calibration;
    std::optional<fs::path> engine;
    int gpu = 0, capture_adapter = -1, capture_output = -1, edge = 1024, workspace_mib = 256;
    int inference_hz = 60, capture_hz = 60, mouse_hz = 240, target_lost_frames = 4;
    int max_targets = 100, max_display = 100, start_key = 0xBB, stop_key = 0xBD;
    double confidence = 0.5, max_age = .1, report_seconds = 5, hotkey_seconds = .003;
    double shot_interval = .10, shot_hold = .09, aim_height = .9, aim_time = .03, max_step = 70;
    double match_iou = .10, match_distance = .65, match_area = 4;
    bool controls_enabled = true, auto_shoot = true, instant = false, overlay = false, graph = true, stage_timing = false;
    std::string precision = "int8", capture_backend = "auto", enemy_name = "enemy";
    std::vector<std::string> fp32_layers;
    Json qualification;
    static Options load(const fs::path& path) {
        auto j = read_json(path); Options o; o.root = fs::absolute(path).parent_path();
        o.export_manifest = resolve_path(o.root, j.at("export_manifest").get<std::string>());
        o.calibration_manifest = resolve_path(o.root, j.value("calibration_manifest", "calibration.json"));
        o.cache = resolve_path(o.root, j.value("cache_dir", "cache"));
        o.calibration = resolve_path(o.root, j.value("mouse_calibration", "../../mouse_calibration.json"));
        if (j.contains("engine") && !j["engine"].is_null()) o.engine = resolve_path(o.root, j["engine"].get<std::string>());
        o.gpu = j.value("gpu_index", 0); o.capture_adapter = j.value("capture_adapter", -1);
        o.capture_output = j.value("capture_output", -1); o.edge = j.value("long_edge", 1024);
        o.workspace_mib = j.value("workspace_mib", 256); o.inference_hz = j.value("inference_hz", 60);
        o.capture_hz = j.value("capture_hz", 60); o.precision = j.value("precision", "int8");
        o.capture_backend = j.value("capture_backend", "auto"); o.enemy_name = j.value("enemy_class", "enemy");
        o.confidence = j.value("confidence", .5); o.max_age = j.value("max_result_age_seconds", .1);
        o.report_seconds = j.value("report_seconds", 5.0); o.overlay = j.value("overlay", false);
        o.graph = j.value("cuda_graph", true); o.stage_timing = j.value("stage_timing", false);
        o.max_targets = j.value("max_target_detections", 100); o.max_display = j.value("max_display_detections", 100);
        o.fp32_layers = j.value("fp32_layer_patterns", std::vector<std::string>{});
        o.qualification = j.value("qualification", Json());
        auto c = j.value("controls", Json::object());
        o.controls_enabled = c.value("enabled", true);
        o.auto_shoot = c.value("auto_shoot", true); o.instant = c.value("instant_mouse", false);
        o.mouse_hz = c.value("mouse_update_hz", 240); o.aim_height = c.value("aim_height_from_bottom", .9);
        o.aim_time = c.value("aim_time_constant_seconds", .03); o.max_step = c.value("max_mouse_step_pixels", 70.0);
        o.target_lost_frames = c.value("target_lost_frames", 4);
        o.shot_interval = c.value("shoot_interval_seconds", .10); o.shot_hold = c.value("shoot_hold_seconds", .09);
        o.match_iou = c.value("target_match_min_iou", .10);
        o.match_distance = c.value("target_match_max_center_distance", .65);
        o.match_area = c.value("target_match_max_area_ratio", 4.0);
        o.start_key = c.value("start_key", 0xBB); o.stop_key = c.value("stop_key", 0xBD);
        o.hotkey_seconds = c.value("hotkey_poll_seconds", .003);
        require(o.edge == 1024 && o.inference_hz > 0 && o.inference_hz <= 60 && o.capture_hz > 0 &&
                o.capture_hz <= 120 && o.workspace_mib > 0 && o.workspace_mib <= 512 &&
                o.mouse_hz > 0 && o.mouse_hz <= 1000 && o.max_targets > 0 && o.max_targets <= 300 &&
                o.max_display > 0 && o.max_display <= 300 && o.gpu >= 0 &&
                o.capture_adapter >= -1 && o.capture_output >= -1, "Invalid size/rate/memory options");
        require(o.precision == "int8" || o.precision == "fp32", "precision must be int8 or fp32");
        require(o.capture_backend == "auto" || o.capture_backend == "cpu" || o.capture_backend == "cuda",
                "capture_backend must be auto, cpu, or cuda");
        require(o.confidence >= 0 && o.confidence <= 1 && o.max_age > 0 && o.max_age <= .1 &&
                o.report_seconds >= 1 && o.hotkey_seconds >= .001 && o.hotkey_seconds <= 1 &&
                o.aim_time > 0 && o.max_step > 0 && o.max_step <= 32767 &&
                o.aim_height >= 0 && o.aim_height <= 1 && o.shot_interval > 0 && o.shot_hold > 0 &&
                o.match_area >= 1 && o.match_iou >= 0 && o.match_iou <= 1 && o.match_distance >= 0 &&
                o.target_lost_frames > 0 && o.start_key > 0 && o.start_key < 256 &&
                o.stop_key > 0 && o.stop_key < 256 && o.start_key != o.stop_key,
                "Invalid confidence/control/deadline options");
        require(!j.value("collection", false) && !j.value("recording", false),
                "Collection/recording belong in the existing Python tools, not this runtime");
        return o;
    }
};
struct Detection { float x1, y1, x2, y2, score; int cls; };
struct State {
    std::mutex mutex;
    std::atomic<bool> running{false}, shutdown{false};
    uint64_t arm_generation = 0, result_generation = 0;
    Clock::time_point captured{};
    std::optional<Detection> target, visible;
    std::vector<Detection> display;
    int missed = 0, locked_x = 0, locked_y = 0;
    void clear_locked() {
        target.reset(); visible.reset(); display.clear(); missed = 0; captured = {}; ++result_generation;
    }
    void pause() {
        std::lock_guard<std::mutex> lock(mutex); running = false; ++arm_generation; clear_locked();
    }
    void arm() {
        std::lock_guard<std::mutex> lock(mutex); ++arm_generation; clear_locked(); running = true;
    }
};
class Deadline {
    HANDLE timer_ = nullptr;
public:
    Deadline() {
        timer_ = CreateWaitableTimerExW(nullptr, nullptr, 0x2, TIMER_ALL_ACCESS);
        if (!timer_) timer_ = CreateWaitableTimerW(nullptr, FALSE, nullptr);
        require(timer_ != nullptr, "Cannot create pacing timer");
    }
    ~Deadline() { CloseHandle(timer_); }
    void wait(Clock::time_point deadline, const std::atomic<bool>& stop, const std::atomic<bool>* armed = nullptr) {
        while (!stop && (!armed || *armed)) {
            auto remaining = deadline - Clock::now(); if (remaining <= Clock::duration::zero()) return;
            auto ticks = std::chrono::duration_cast<std::chrono::nanoseconds>(remaining).count() / 100;
            LARGE_INTEGER due; due.QuadPart = -std::max<int64_t>(1, std::min<int64_t>(ticks, 20'000));
            require(SetWaitableTimer(timer_, &due, 0, nullptr, nullptr, FALSE), "Cannot set pacing timer");
            WaitForSingleObject(timer_, INFINITE);
        }
    }
};
} // namespace y1050
