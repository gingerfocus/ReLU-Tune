#!/usr/bin/env python3
"""Plot combined staged training metrics saved in per-stage histories."""

import argparse
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))


def _load_stage_metrics(stage_dir):
    metrics_path = stage_dir / "metrics.json"
    if not metrics_path.exists():
        return []
    with metrics_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _collect_series(stage_dirs):
    train_loss_points = []
    eval_loss_points = []
    lr_points = []
    global_offset = 0

    for stage_dir in stage_dirs:
        metrics = _load_stage_metrics(stage_dir)
        stage_max_step = 0

        for item in metrics:
            local_step = item.get("step")
            if local_step is None:
                continue

            local_step = int(local_step)
            stage_max_step = max(stage_max_step, local_step)
            global_step = global_offset + local_step

            if "loss" in item:
                train_loss_points.append((global_step, item["loss"]))
            if "eval_loss" in item:
                eval_loss_points.append((global_step, item["eval_loss"]))
            if "learning_rate" in item:
                lr_points.append((global_step, item["learning_rate"]))

        global_offset += stage_max_step

    return train_loss_points, eval_loss_points, lr_points


def _plot_single_series(points, title, ylabel, label, output_path, marker="o"):
    if not points:
        return

    import matplotlib.pyplot as plt

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys, marker=marker, linewidth=1.5, label=label)
    plt.title(title)
    plt.xlabel("Global Step")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def _plot_loss_overlay(train_points, eval_points, output_path):
    if not train_points and not eval_points:
        return

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))

    if train_points:
        xs = [point[0] for point in train_points]
        ys = [point[1] for point in train_points]
        plt.plot(xs, ys, marker="o", linewidth=1.5, label="train loss")

    if eval_points:
        xs = [point[0] for point in eval_points]
        ys = [point[1] for point in eval_points]
        plt.plot(xs, ys, marker="s", linewidth=1.5, label="validation loss")

    plt.title("Loss Overlay")
    plt.xlabel("Global Step")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot ReLU-Tune run metrics")
    parser.add_argument("--run-dir", required=True, help="Run directory")
    parser.add_argument("--output-dir", default=None, help="Optional output directory")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "plots"
    stage_dirs = sorted(path for path in run_dir.glob("stage_*") if path.is_dir())

    train_loss_points, eval_loss_points, lr_points = _collect_series(stage_dirs)

    try:
        import matplotlib  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plot_run_metrics.py. Install requirements.txt first."
        ) from exc

    _plot_single_series(
        train_loss_points,
        "Train Loss",
        "Loss",
        "train loss",
        output_dir / "train_loss.png",
        marker="o",
    )
    _plot_single_series(
        eval_loss_points,
        "Validation Loss",
        "Loss",
        "validation loss",
        output_dir / "val_loss.png",
        marker="s",
    )
    _plot_loss_overlay(train_loss_points, eval_loss_points, output_dir / "loss_overlay.png")
    _plot_single_series(lr_points, "Learning Rate", "LR", "learning rate", output_dir / "lr.png")
    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    main()
