"""Live, browser-based training charts built from Ultralytics results.csv."""

import base64
import csv
import html
import json
import math
import os
import statistics
import webbrowser
from datetime import datetime, timedelta
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from dataset_utils import (atomic_write)


BACKGROUND = "#0b1220"
PANEL = "#142033"
TEXT = "#f1f5fb"
MUTED = "#9aacbf"
GRID = "#34445b"
COLORS = ("#5dd8ff", "#b699ff", "#54d7ad")


def _read_history(path: Path) -> dict[int, dict[str, float]]:
    """Ignore incomplete CSV rows while an epoch is being written."""
    if not path.is_file():
        return {}
    history = {}
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            try:
                values = {key.strip(): float(value) for key, value in row.items()
                          if key and value and math.isfinite(float(value))}
                epoch = values.pop("epoch")
                if not epoch.is_integer() or epoch < 1:
                    continue
                history[int(epoch)] = values
            except (TypeError, ValueError, KeyError):
                continue
    return history


def _metric(history: dict[int, dict[str, float]], name: str) -> list[tuple[int, float]]:
    return [(epoch, values[name]) for epoch, values in sorted(history.items()) if name in values]


def _draw_panel(ax, history, title, series, *, fraction=False) -> None:
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=TEXT, fontsize=12, fontweight="bold", loc="left", pad=14)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID, alpha=0.55, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_xlabel("Epoch", color=MUTED, fontsize=9)
    plotted = False
    for index, (key, label) in enumerate(series):
        points = _metric(history, key)
        if not points:
            continue
        epochs, values = zip(*points)
        color = COLORS[index % len(COLORS)]
        ax.plot(epochs, values, color=color, linewidth=2.2, label=f"{label}  {values[-1]:.3f}")
        ax.scatter(epochs[-1], values[-1], color=color, s=22, zorder=3)
        plotted = True
    if plotted:
        ax.legend(loc="best", frameon=False, fontsize=8, labelcolor=TEXT)
        if fraction:
            ax.set_ylim(0, 1.02)
    else:
        ax.text(0.5, 0.5, "Waiting for metrics", transform=ax.transAxes,
                ha="center", va="center", color=MUTED, fontsize=11)


def _render_chart(path: Path, run_name: str, history: dict[int, dict[str, float]]) -> dict:
    figure = Figure(figsize=(15, 8.6), dpi=140, facecolor=BACKGROUND)
    canvas = FigureCanvasAgg(figure)
    axes = figure.subplots(2, 3)
    figure.subplots_adjust(left=0.055, right=0.975, top=0.78, bottom=0.10,
                           wspace=0.27, hspace=0.42)

    l1_name = "l1" if any("train/l1_loss" in row for row in history.values()) else "dfl"
    panels = (
        ("Detection quality", (("metrics/mAP50-95(B)", "mAP 50–95"),
                                ("metrics/mAP50(B)", "mAP 50")), True),
        ("Precision and recall", (("metrics/precision(B)", "Precision"),
                                  ("metrics/recall(B)", "Recall")), True),
        ("Box loss", (("train/box_loss", "Train"), ("val/box_loss", "Validation")), False),
        ("Class loss", (("train/cls_loss", "Train"), ("val/cls_loss", "Validation")), False),
        (f"{l1_name.upper()} loss", ((f"train/{l1_name}_loss", "Train"),
                                    (f"val/{l1_name}_loss", "Validation")), False),
        ("Learning rate", (("lr/pg0", "Group 0"), ("lr/pg1", "Group 1"),
                           ("lr/pg2", "Group 2")), False),
    )
    for ax, (title, series, fraction) in zip(axes.flat, panels):
        _draw_panel(ax, history, title, series, fraction=fraction)

    epochs = sorted(history)
    scores = _metric(history, "metrics/mAP50-95(B)")
    best_epoch, best_score = max(scores, key=lambda item: item[1]) if scores else (None, None)
    figure.text(0.055, 0.945, "YOLO26  /  TRAINING", color=COLORS[0],
                fontsize=12, fontweight="bold")
    figure.text(0.055, 0.886, run_name, color=TEXT, fontsize=25, fontweight="bold")
    figure.text(0.055, 0.835,
                f"{len(epochs)} completed epochs  ·  Latest epoch {epochs[-1] if epochs else '—'}",
                color=MUTED, fontsize=11)
    figure.text(0.61, 0.902, "BEST mAP 50–95", color=MUTED, fontsize=10)
    figure.text(0.61, 0.850, f"{best_score:.3f}" if best_score is not None else "—",
                color=COLORS[1], fontsize=23, fontweight="bold")
    figure.text(0.80, 0.902, "BEST EPOCH", color=MUTED, fontsize=10)
    figure.text(0.80, 0.850, str(best_epoch) if best_epoch is not None else "—",
                color=COLORS[2], fontsize=23, fontweight="bold")
    figure.text(0.055, 0.035, "Higher is better for detection metrics; lower is better for losses.",
                color=MUTED, fontsize=9)
    canvas.draw()
    width, height = canvas.get_width_height()
    hover_points = []
    for ax, (title, series, _fraction) in zip(axes.flat, panels):
        for key, label in series:
            for epoch, value in _metric(history, key):
                x, y = ax.transData.transform((epoch, value))
                if math.isfinite(x) and math.isfinite(y):
                    hover_points.append({"panel": title, "series": label, "epoch": epoch,
                                         "value": value, "x": round(x, 2), "y": round(height - y, 2)})
    figure.savefig(path, format="png", facecolor=BACKGROUND)
    figure.clear()
    return {"width": width, "height": height, "points": hover_points}


def _training_eta(history: dict[int, dict[str, float]], total_epochs: int | None) -> tuple[str, str, str]:
    if not history or total_epochs is None or total_epochs <= 0:
        return "—", "Waiting for completed epochs", "—"
    latest_epoch = max(history)
    if latest_epoch >= total_epochs:
        return "Complete", f"{latest_epoch} / {total_epochs} epochs", "—"
    timed = [(epoch, values["time"]) for epoch, values in sorted(history.items()) if "time" in values]
    durations = [
        (current_time - previous_time) / (current_epoch - previous_epoch)
        for (previous_epoch, previous_time), (current_epoch, current_time) in zip(timed, timed[1:])
        if current_epoch > previous_epoch and current_time > previous_time
    ]
    if durations:
        recent = durations[-8:]
        seconds_per_epoch = statistics.median(recent)
        basis = f"Median of {len(recent)} recent epoch{'s' if len(recent) != 1 else ''}"
    elif timed and timed[0][0] == 1 and timed[0][1] > 0:
        seconds_per_epoch = timed[0][1]
        basis = "Based on the first epoch"
    else:
        return "—", f"{latest_epoch} / {total_epochs} epochs · waiting for timing", "—"
    remaining = (total_epochs - latest_epoch) * seconds_per_epoch
    hours, remainder = divmod(round(remaining), 3600)
    minutes = round(remainder / 60)
    if minutes == 60:
        hours += 1
        minutes = 0
    duration_text = f"{hours}h {minutes}m" if hours else f"{minutes}m"
    finish_at = (datetime.now().astimezone() + timedelta(seconds=remaining)).strftime("%b %d, %I:%M %p %Z")
    return duration_text, f"{latest_epoch} / {total_epochs} epochs · {basis}", finish_at


def _write_html(path: Path, image_path: Path, csv_path: Path, run_name: str,
                refresh_seconds: int, chart: dict, history: dict[int, dict[str, float]],
                total_epochs: int | None, *, complete: bool = False) -> None:
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    chart_json = json.dumps(chart, ensure_ascii=True, separators=(",", ":"))
    updated = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    eta, eta_basis, finish_at = _training_eta(history, total_epochs)
    if complete:
        eta = "Finished"
        eta_basis = f"{max(history) if history else 0} completed epochs"
        finish_at = "—"
    hover_script = """
const chart = JSON.parse(document.getElementById('chart-data').textContent);
const overlay = document.getElementById('chart-overlay');
const marker = document.getElementById('chart-marker');
const tooltip = document.getElementById('chart-tooltip');
const wrapper = document.getElementById('chart-wrap');
function hidePoint() {
  marker.style.display = 'none';
  tooltip.style.display = 'none';
}
overlay.addEventListener('mousemove', event => {
  const rect = overlay.getBoundingClientRect();
  const x = (event.clientX - rect.left) * chart.width / rect.width;
  const y = (event.clientY - rect.top) * chart.height / rect.height;
  let nearest = null;
  let distance = Infinity;
  for (const point of chart.points) {
    const dx = (point.x - x) * rect.width / chart.width;
    const dy = (point.y - y) * rect.height / chart.height;
    const candidate = dx * dx + dy * dy;
    if (candidate < distance) {
      distance = candidate;
      nearest = point;
    }
  }
  if (!nearest || distance > 18 * 18) {
    hidePoint();
    return;
  }
  marker.setAttribute('cx', nearest.x);
  marker.setAttribute('cy', nearest.y);
  marker.style.display = '';
  tooltip.textContent = `${nearest.panel} · ${nearest.series}\nEpoch ${nearest.epoch}: ${Number(nearest.value).toPrecision(5)}`;
  tooltip.style.display = 'block';
  const bounds = wrapper.getBoundingClientRect();
  const wantedLeft = event.clientX - bounds.left + 14;
  const wantedTop = event.clientY - bounds.top - tooltip.offsetHeight - 14;
  tooltip.style.left = `${Math.max(8, Math.min(wantedLeft, wrapper.clientWidth - tooltip.offsetWidth - 8))}px`;
  tooltip.style.top = `${wantedTop >= 8 ? wantedTop : event.clientY - bounds.top + 14}px`;
});
overlay.addEventListener('mouseleave', hidePoint);
"""
    content = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
{'' if complete else f'<meta http-equiv="refresh" content="{refresh_seconds}">'}
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(run_name)} · Training metrics</title>
<style>
body {{ margin: 0; background: {BACKGROUND}; color: {TEXT}; font: 15px/1.5 system-ui, sans-serif; }}
main {{ max-width: 1600px; margin: 0 auto; padding: 28px; }}
header {{ display: flex; justify-content: space-between; gap: 20px; align-items: center; flex-wrap: wrap; }}
h1 {{ margin: 0; font-size: 21px; }}
.sub {{ color: {MUTED}; margin: 4px 0 0; }}
.live {{ color: {COLORS[2]}; background: #143b34; border: 1px solid #286455; border-radius: 999px; padding: 7px 13px; }}
.stats {{ display: flex; flex-wrap: wrap; gap: 12px; margin: 22px 0 0; }}
.stat {{ min-width: 220px; padding: 14px 18px; border: 1px solid #26354b; border-radius: 12px; background: {PANEL}; }}
.stat span, .stat small {{ display: block; color: {MUTED}; }}
.stat strong {{ display: block; margin: 3px 0; font-size: 22px; }}
.chart-wrap {{ position: relative; margin: 22px 0; }}
.chart {{ display: block; width: 100%; height: auto; box-sizing: border-box; border: 1px solid #26354b; border-radius: 16px; box-shadow: 0 20px 60px #05091288; }}
.chart-overlay {{ position: absolute; left: 1px; top: 1px; width: calc(100% - 2px); height: calc(100% - 2px); cursor: crosshair; }}
.chart-tooltip {{ display: none; position: absolute; z-index: 2; padding: 9px 12px; border: 1px solid #5b7597; border-radius: 8px; background: #091422f2; color: {TEXT}; white-space: pre-line; pointer-events: none; box-shadow: 0 8px 24px #0008; }}
a {{ color: {COLORS[0]}; text-decoration: none; margin-right: 18px; }} a:hover {{ text-decoration: underline; }}
footer {{ color: {MUTED}; font-size: 13px; }}
</style></head><body><main>
<header><div><h1>{html.escape(run_name)} · Training dashboard</h1>
<p class="sub">Updated {html.escape(updated)} · {'Training complete' if complete else f'Refreshes every {refresh_seconds} seconds'}</p></div>
<span class="live">● {'COMPLETE' if complete else 'LIVE METRICS'}</span></header>
<div class="stats"><div class="stat"><span>Estimated time left</span><strong>{html.escape(eta)}</strong><small>{html.escape(eta_basis)}</small></div>
<div class="stat"><span>Estimated finish</span><strong>{html.escape(finish_at)}</strong><small>Based on recent epoch times</small></div></div>
<div class="chart-wrap" id="chart-wrap"><img class="chart" alt="Training metrics charts" src="data:image/png;base64,{encoded}">
<svg class="chart-overlay" id="chart-overlay" viewBox="0 0 {chart['width']} {chart['height']}" preserveAspectRatio="none" aria-label="Hover over a chart point to see its value"><circle id="chart-marker" r="8" fill="none" stroke="#fff" stroke-width="3" style="display:none;pointer-events:none"/></svg>
<div class="chart-tooltip" id="chart-tooltip" role="status"></div></div>
<footer><a href="{html.escape(image_path.name)}" download>Download PNG</a>
<a href="{html.escape(csv_path.name)}">Open results CSV</a>
The chart reads the run's saved CSV, including epochs from an interrupted run.</footer>
<script type="application/json" id="chart-data">{chart_json}</script>
<script>{hover_script}</script>
</main></body></html>"""
    atomic_write(path, content)


class TrainingDashboard:
    """Ultralytics callbacks for a per-run chart that survives resume."""

    def __init__(self, *, open_browser: bool = True, refresh_seconds: int = 15) -> None:
        self.open_browser = open_browser
        self.refresh_seconds = refresh_seconds
        self.history: dict[int, dict[str, float]] = {}
        self.csv_path: Path | None = None
        self.image_path: Path | None = None
        self.html_path: Path | None = None
        self.last_csv_state: tuple[int, int] | None = None
        self.disabled = False

    def start(self, trainer) -> None:
        if self.disabled:
            return
        try:
            self.csv_path = Path(trainer.csv)
            run_dir = Path(trainer.save_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            self.image_path = run_dir / "training_dashboard.png"
            self.html_path = run_dir / "training_dashboard.html"
            self.update(trainer, force=True)
            print(f"Training dashboard: {self.html_path}")
            if self.open_browser:
                webbrowser.open_new_tab(self.html_path.resolve().as_uri())
        except Exception as exc:
            self.disabled = True
            print(f"Training dashboard could not start: {exc}")

    def update(self, trainer, *, force: bool = False, complete: bool = False) -> None:
        if self.disabled or self.csv_path is None or self.image_path is None or self.html_path is None:
            return
        try:
            csv_state = None
            if self.csv_path.is_file():
                stat = self.csv_path.stat()
                csv_state = (stat.st_mtime_ns, stat.st_size)
            if not force and csv_state == self.last_csv_state:
                return
            self.history.update(_read_history(self.csv_path))
            temporary_image = self.image_path.with_name(".training_dashboard.tmp.png")
            try:
                chart = _render_chart(temporary_image, Path(trainer.save_dir).name, self.history)
                os.replace(temporary_image, self.image_path)
            finally:
                temporary_image.unlink(missing_ok=True)
            planned_epochs = getattr(trainer, "epochs", None)
            if planned_epochs is None:
                planned_epochs = getattr(getattr(trainer, "args", None), "epochs", None)
            try:
                planned_epochs = int(planned_epochs) if planned_epochs is not None else None
            except (TypeError, ValueError):
                planned_epochs = None
            _write_html(self.html_path, self.image_path, self.csv_path,
                        Path(trainer.save_dir).name, self.refresh_seconds, chart,
                        self.history, planned_epochs, complete=complete)
            self.last_csv_state = csv_state
        except Exception as exc:
            self.disabled = True
            print(f"Training dashboard stopped updating: {exc}")

    def finish(self, trainer) -> None:
        self.update(trainer, force=True, complete=True)


def attach_dashboard(model, *, open_browser=True, refresh_seconds=15):
    """Register the same per-run dashboard callbacks for training and distillation."""
    dashboard = TrainingDashboard(open_browser=open_browser, refresh_seconds=refresh_seconds)
    model.add_callback("on_train_start", dashboard.start)
    model.add_callback("on_fit_epoch_end", dashboard.update)
    model.add_callback("on_train_end", dashboard.finish)
    return dashboard
