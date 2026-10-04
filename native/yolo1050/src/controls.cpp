#include "controls.hpp"
#include <cstdio>
namespace y1050 {
namespace {
double center_x(const Detection& d) { return (d.x1 + d.x2) * .5; }
double center_y(const Detection& d) { return (d.y1 + d.y2) * .5; }
double area(const Detection& d) { return (d.x2 - d.x1) * (d.y2 - d.y1); }
double diagonal(const Detection& d) { return std::hypot(d.x2 - d.x1, d.y2 - d.y1); }
double overlap(const Detection& a, const Detection& b) {
    double intersection = std::max(0.0f, std::min(a.x2, b.x2) - std::max(a.x1, b.x1)) *
                          std::max(0.0f, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
    double combined = area(a) + area(b) - intersection; return combined > 0 ? intersection / combined : 0;
}
std::optional<Detection> match(const Detection& previous, const std::vector<Detection>& targets, const Options& o) {
    std::optional<Detection> overlapping, nearby;
    double best_iou = -1, best_iou_distance = 1e30, best_distance = 1e30;
    for (const auto& box : targets) {
        double ratio = area(previous) > 0 ? area(box) / area(previous) : 0;
        if (ratio < 1 / o.match_area || ratio > o.match_area) continue;
        double distance = std::hypot(center_x(box) - center_x(previous), center_y(box) - center_y(previous));
        double iou = overlap(previous, box);
        if (iou >= o.match_iou) {
            if (iou > best_iou || (iou == best_iou && distance < best_iou_distance)) {
                overlapping = box; best_iou = iou; best_iou_distance = distance;
            }
        } else if (distance <= o.match_distance * std::max(diagonal(previous), diagonal(box)) && distance < best_distance) {
            nearby = box; best_distance = distance;
        }
    }
    return overlapping ? overlapping : nearby;
}
void button(bool down) { mouse_event(down ? MOUSEEVENTF_LEFTDOWN : MOUSEEVENTF_LEFTUP, 0, 0, 0, 0); }
std::pair<int, int> move(double dx, double dy, const Options& o) {
    double distance = std::hypot(dx, dy); if (distance < .5) return {0, 0};
    double scale = o.instant ? 1 : std::min(1.0, o.max_step / distance);
    int x = int(std::nearbyint(dx * scale)), y = int(std::nearbyint(dy * scale));
    if (!x && std::abs(dx) >= .5) x = dx > 0 ? 1 : -1;
    if (!y && std::abs(dy) >= .5) y = dy > 0 ? 1 : -1;
    mouse_event(MOUSEEVENTF_MOVE, static_cast<DWORD>(x), static_cast<DWORD>(y), 0, 0); return {x, y};
}
bool key_down(int code) { return (GetAsyncKeyState(code) & 0x8000) != 0; }
LRESULT CALLBACK overlay_proc(HWND window, UINT message, WPARAM wparam, LPARAM lparam) {
    if (message == WM_NCHITTEST) return HTTRANSPARENT;
    if (message == WM_ERASEBKGND) return 1;
    if (message == WM_PAINT) { PAINTSTRUCT paint{}; BeginPaint(window, &paint); EndPaint(window, &paint); return 0; }
    return DefWindowProcW(window, message, wparam, lparam);
}
}
void load_calibration(State& state, const Options& o, const Geometry& g) {
    auto j = read_json(o.calibration);
    require(j.at("schema_version") == 1 && j.at("screen_width") == g.screen_w &&
            j.at("screen_height") == g.screen_h && j.at("locked_x").is_number_integer() &&
            j.at("locked_y").is_number_integer(), "Mouse calibration does not match this display; run calibrate.py here");
    state.locked_x = j.at("locked_x").get<int>(); state.locked_y = j.at("locked_y").get<int>();
    require(state.locked_x >= 0 && state.locked_x < g.screen_w && state.locked_y >= 0 &&
            state.locked_y < g.screen_h, "Invalid locked cursor position");
}
bool publish(State& state, const Options& o, const Geometry& g, const float* output, int count, int enemy,
             Clock::time_point captured, uint64_t generation) {
    std::vector<Detection> targets, display; targets.reserve(o.max_targets); display.reserve(o.max_display);
    for (int i = 0; i < count; ++i) {
        auto row = output + i * 6; bool finite = true;
        for (int j = 0; j < 6; ++j) finite = finite && std::isfinite(row[j]);
        if (!finite || row[4] < o.confidence || row[4] > 1 || row[5] < 0 || row[5] > 2 ||
            row[5] != std::nearbyint(row[5])) continue;
        Detection d{std::clamp((row[0] - g.left) / g.scale, 0.0f, float(g.screen_w)),
                    std::clamp((row[1] - g.top) / g.scale, 0.0f, float(g.screen_h)),
                    std::clamp((row[2] - g.left) / g.scale, 0.0f, float(g.screen_w)),
                    std::clamp((row[3] - g.top) / g.scale, 0.0f, float(g.screen_h)), row[4], int(row[5])};
        if (d.x2 <= d.x1 || d.y2 <= d.y1) continue;
        if (d.cls == enemy && targets.size() < size_t(o.max_targets)) targets.push_back(d);
        if (o.overlay && display.size() < size_t(o.max_display)) display.push_back(d);
    }
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.running || state.arm_generation != generation) return false;
    if (seconds(Clock::now() - captured) > o.max_age) { state.clear_locked(); return false; }
    auto chosen = state.target ? match(*state.target, targets, o) : std::optional<Detection>{};
    if (state.target && !chosen) {
        if (++state.missed >= o.target_lost_frames) { state.target.reset(); state.missed = 0; }
    } else if (chosen) state.missed = 0;
    if (!state.target && !targets.empty()) {
        chosen = *std::min_element(targets.begin(), targets.end(), [&](const auto& a, const auto& b) {
            return std::hypot(center_x(a) - state.locked_x, center_y(a) - state.locked_y) <
                   std::hypot(center_x(b) - state.locked_x, center_y(b) - state.locked_y);
        });
    }
    state.visible = chosen; if (chosen) state.target = chosen;
    state.display = std::move(display); state.captured = captured; ++state.result_generation; return true;
}
void Controls::run() noexcept {
    bool mouse_down = false;
    try {
        Deadline timer; auto now = Clock::now(), next_mouse = now, previous_mouse = now, next_hotkey = now;
        auto next_shot = Clock::time_point{}, release_at = Clock::time_point{};
        auto period = std::chrono::nanoseconds(1'000'000'000 / options_.mouse_hz);
        auto hotkey_period = std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(options_.hotkey_seconds));
        bool start_was = key_down(options_.start_key), stop_was = key_down(options_.stop_key);
        uint64_t last_generation = UINT64_MAX; double remaining_x = 0, remaining_y = 0;
        while (!state_.shutdown) {
            now = Clock::now();
            if (now >= next_hotkey) {
                bool start = key_down(options_.start_key), stop = key_down(options_.stop_key);
                if (stop && !stop_was) { state_.pause(); std::cout << "Paused. Press = to arm.\n"; }
                else if (start && !start_was) { state_.arm(); std::cout << "Armed. Press - to pause.\n"; }
                start_was = start; stop_was = stop; next_hotkey = now + hotkey_period;
            }
            if (now >= next_mouse) {
                double dt = std::min(seconds(now - previous_mouse), .05); previous_mouse = now;
                next_mouse = std::max(next_mouse + period, now + std::chrono::nanoseconds(1));
                if (mouse_down && now >= release_at) { button(false); mouse_down = false; }
                std::lock_guard<std::mutex> lock(state_.mutex);
                if (state_.captured != Clock::time_point{} && seconds(now - state_.captured) > options_.max_age)
                    state_.clear_locked();
                if (!state_.running || !state_.visible) {
                    if (mouse_down) { button(false); mouse_down = false; }
                    remaining_x = remaining_y = 0; last_generation = state_.result_generation;
                } else {
                    const auto& box = *state_.visible;
                    if (last_generation != state_.result_generation) {
                        last_generation = state_.result_generation;
                        remaining_x = center_x(box) - state_.locked_x;
                        remaining_y = box.y2 - options_.aim_height * (box.y2 - box.y1) - state_.locked_y;
                    }
                    double gain = options_.instant ? 1 : 1 - std::exp(-dt / options_.aim_time);
                    double dx = remaining_x * gain, dy = remaining_y * gain;
                    if (std::abs(remaining_x) >= .5 && std::abs(dx) < .5) dx = std::copysign(.5, remaining_x);
                    if (std::abs(remaining_y) >= .5 && std::abs(dy) < .5) dy = std::copysign(.5, remaining_y);
                    auto movement = move(dx, dy, options_);
                    if (options_.instant) remaining_x = remaining_y = 0;
                    else { remaining_x -= movement.first; remaining_y -= movement.second; }
                    if (options_.auto_shoot && !mouse_down && now >= next_shot) {
                        POINT cursor{};
                        if (GetCursorPos(&cursor) && cursor.x >= box.x1 && cursor.x <= box.x2 &&
                            cursor.y >= box.y1 && cursor.y <= box.y2) {
                            button(true); mouse_down = true;
                            release_at = now + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(options_.shot_hold));
                            next_shot = now + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(options_.shot_interval));
                        }
                    }
                }
            }
            timer.wait(std::min(next_mouse, next_hotkey), state_.shutdown);
        }
    } catch (...) {
        { std::lock_guard<std::mutex> lock(mutex_); error_ = std::current_exception(); }
        state_.pause(); state_.shutdown = true;
    }
    if (mouse_down) button(false);
}
void Overlay::run() noexcept {
    HWND window = nullptr; HDC dc = nullptr, buffer = nullptr; HBITMAP bitmap = nullptr;
    HGDIOBJ previous_bitmap = nullptr, previous_font = nullptr; HFONT font = nullptr;
    HBRUSH background = nullptr; std::array<HPEN, 3> pens{};
    try {
        auto instance = GetModuleHandleW(nullptr); WNDCLASSW wc{};
        wc.hInstance = instance; wc.lpfnWndProc = overlay_proc; wc.lpszClassName = L"YOLO1050Overlay";
        require(RegisterClassW(&wc) || GetLastError() == ERROR_CLASS_ALREADY_EXISTS, "Overlay window class");
        window = CreateWindowExW(WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE | WS_EX_TOPMOST,
            wc.lpszClassName, L"YOLO1050", WS_POPUP, 0, 0, geometry_.screen_w, geometry_.screen_h,
            nullptr, nullptr, instance, nullptr);
        require(window != nullptr, "Overlay window creation");
        require(SetLayeredWindowAttributes(window, RGB(1, 1, 1), 255, LWA_COLORKEY), "Overlay transparency");
        if (!SetWindowDisplayAffinity(window, 0x11)) {
            std::cerr << "Overlay disabled: capture exclusion unavailable.\n"; DestroyWindow(window); return;
        }
        dc = GetDC(window); buffer = CreateCompatibleDC(dc);
        bitmap = CreateCompatibleBitmap(dc, geometry_.screen_w, geometry_.screen_h);
        require(dc && buffer && bitmap, "Overlay drawing buffer");
        previous_bitmap = SelectObject(buffer, bitmap);
        font = CreateFontW(-16, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
                          OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, DEFAULT_QUALITY, DEFAULT_PITCH, L"Consolas");
        require(font != nullptr, "Overlay font"); previous_font = SelectObject(buffer, font);
        background = CreateSolidBrush(RGB(1, 1, 1)); require(background != nullptr, "Overlay background");
        std::array<COLORREF, 3> colors{RGB(180, 180, 180), RGB(255, 65, 65), RGB(65, 200, 255)};
        for (size_t i = 0; i < pens.size(); ++i) { pens[i] = CreatePen(PS_SOLID, 2, colors[i]); require(pens[i] != nullptr, "Overlay pen"); }
        SetBkMode(buffer, TRANSPARENT); ShowWindow(window, SW_SHOWNOACTIVATE);
        Deadline timer; auto next = Clock::now(); uint64_t last = UINT64_MAX; bool was_visible = false;
        while (!state_.shutdown) {
            MSG message{}; while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE)) { TranslateMessage(&message); DispatchMessageW(&message); }
            std::vector<Detection> boxes; uint64_t generation; bool visible;
            {
                std::lock_guard<std::mutex> lock(state_.mutex);
                visible = state_.running && state_.captured != Clock::time_point{} &&
                          seconds(Clock::now() - state_.captured) <= options_.max_age;
                generation = state_.result_generation; if (visible) boxes = state_.display;
            }
            if (generation != last || visible != was_visible) {
                RECT extent{0, 0, geometry_.screen_w, geometry_.screen_h}; FillRect(buffer, &extent, background);
                auto brush = SelectObject(buffer, GetStockObject(HOLLOW_BRUSH));
                for (const auto& box : boxes) {
                    auto pen = SelectObject(buffer, pens[box.cls]); SetTextColor(buffer, colors[box.cls]);
                    Rectangle(buffer, int(box.x1), int(box.y1), int(box.x2), int(box.y2));
                    char label[128]; std::snprintf(label, sizeof(label), "%s %.2f", names_[box.cls].c_str(), box.score);
                    TextOutA(buffer, int(box.x1), std::max(0, int(box.y1) - 18), label, int(std::strlen(label)));
                    SelectObject(buffer, pen);
                }
                SelectObject(buffer, brush);
                BitBlt(dc, 0, 0, geometry_.screen_w, geometry_.screen_h, buffer, 0, 0, SRCCOPY);
                last = generation; was_visible = visible;
            }
            next = std::max(next + std::chrono::nanoseconds(1'000'000'000 / 60), Clock::now());
            timer.wait(next, state_.shutdown);
        }
    } catch (...) {
        { std::lock_guard<std::mutex> lock(mutex_); error_ = std::current_exception(); }
        state_.pause(); state_.shutdown = true;
    }
    if (buffer && previous_font) SelectObject(buffer, previous_font);
    if (buffer && previous_bitmap) SelectObject(buffer, previous_bitmap);
    for (auto pen : pens) if (pen) DeleteObject(pen);
    if (background) DeleteObject(background); if (font) DeleteObject(font); if (bitmap) DeleteObject(bitmap);
    if (buffer) DeleteDC(buffer); if (dc && window) ReleaseDC(window, dc); if (window) DestroyWindow(window);
}
}
