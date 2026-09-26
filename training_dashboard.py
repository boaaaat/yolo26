"""Live, browser-based training charts built from Ultralytics results.csv."""

import base64
import csv
import html
import math
import os
import tempfile
import webbrowser
from datetime import datetime
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


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


def _render_chart(path: Path, run_name: str, history: dict[int, dict[str, float]]) -> None:
    figure = Figure(figsize=(15, 8.6), dpi=140, facecolor=BACKGROUND)
    FigureCanvasAgg(figure)
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
    figure.savefig(path, format="png", facecolor=BACKGROUND)
    figure.clear()


def _write_html(path: Path, image_path: Path, csv_path: Path, run_name: str,
                refresh_seconds: int, *, complete: bool = False) -> None:
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    updated = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
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
.chart {{ display: block; width: 100%; margin: 22px 0; border: 1px solid #26354b; border-radius: 16px; box-shadow: 0 20px 60px #05091288; }}
a {{ color: {COLORS[0]}; text-decoration: none; margin-right: 18px; }} a:hover {{ text-decoration: underline; }}
footer {{ color: {MUTED}; font-size: 13px; }}
</style></head><body><main>
<header><div><h1>{html.escape(run_name)} · Training dashboard</h1>
<p class="sub">Updated {html.escape(updated)} · {'Training complete' if complete else f'Refreshes every {refresh_seconds} seconds'}</p></div>
<span class="live">● {'COMPLETE' if complete else 'LIVE METRICS'}</span></header>
<img class="chart" alt="Training metrics charts" src="data:image/png;base64,{encoded}">
<footer><a href="{html.escape(image_path.name)}" download>Download PNG</a>
<a href="{html.escape(csv_path.name)}">Open results CSV</a>
The chart reads the run's saved CSV, including epochs from an interrupted run.</footer>
</main></body></html>"""
    descriptor, temporary = tempfile.mkstemp(prefix=".dashboard-", suffix=".html", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(content)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


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
                _render_chart(temporary_image, Path(trainer.save_dir).name, self.history)
                os.replace(temporary_image, self.image_path)
            finally:
                temporary_image.unlink(missing_ok=True)
            _write_html(self.html_path, self.image_path, self.csv_path,
                        Path(trainer.save_dir).name, self.refresh_seconds, complete=complete)
            self.last_csv_state = csv_state
        except Exception as exc:
            self.disabled = True
            print(f"Training dashboard stopped updating: {exc}")

    def finish(self, trainer) -> None:
        self.update(trainer, force=True, complete=True)
