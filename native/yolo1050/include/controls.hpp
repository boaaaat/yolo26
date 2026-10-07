#pragma once
#include "common.hpp"
namespace y1050 {
void load_calibration(State&, const Options&, const Geometry&);
struct DetectionSummary {
    std::array<uint64_t, 3> above_threshold{};
    std::array<float, 3> max_score{};
    uint64_t malformed = 0, invalid_boxes = 0;
    bool target_selected = false;
};
bool publish(State&, const Options&, const Geometry&, const float*, int candidates, int enemy,
             Clock::time_point captured, uint64_t generation, DetectionSummary&);
class Controls {
    State& state_;
    const Options& options_;
    std::thread worker_;
    std::mutex mutex_;
    std::exception_ptr error_;
    void run() noexcept;
public:
    Controls(State& s, const Options& o) : state_(s), options_(o) {}
    ~Controls() { state_.shutdown = true; if (worker_.joinable()) worker_.join(); }
    void start() { worker_ = std::thread(&Controls::run, this); }
    void check() { std::lock_guard<std::mutex> lock(mutex_); if (error_) std::rethrow_exception(error_); }
};
class Overlay {
    State& state_;
    const Options& options_;
    const Geometry geometry_;
    const std::vector<std::string> names_;
    std::thread worker_;
    std::mutex mutex_;
    std::exception_ptr error_;
    void run() noexcept;
public:
    Overlay(State& s, const Options& o, Geometry g, std::vector<std::string> n)
        : state_(s), options_(o), geometry_(g), names_(std::move(n)) {}
    ~Overlay() { state_.shutdown = true; if (worker_.joinable()) worker_.join(); }
    void start() { worker_ = std::thread(&Overlay::run, this); }
    void check() { std::lock_guard<std::mutex> lock(mutex_); if (error_) std::rethrow_exception(error_); }
};
}
