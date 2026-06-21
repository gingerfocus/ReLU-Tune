from pathlib import Path


def get_stage_bounds(total_steps, stage_size):
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if stage_size <= 0:
        raise ValueError("stage_size must be positive")
    if total_steps % stage_size != 0:
        raise ValueError("total_steps must be divisible by stage_size")

    bounds = []
    stage_count = total_steps // stage_size
    for stage_index in range(stage_count):
        start_step = stage_index * stage_size
        end_step = start_step + stage_size
        bounds.append((stage_index + 1, start_step, end_step))
    return bounds


def get_run_paths(run_dir):
    root = Path(run_dir)
    return {
        "root": root,
        "logs": root / "logs",
        "eval": root / "eval",
        "state": root / "run_state.json",
    }


def get_stage_paths(run_dir, stage_number):
    root = Path(run_dir) / f"stage_{stage_number}"
    return {
        "root": root,
        "adapter": root / "adapter",
        "checkpoint": root / "checkpoint-last",
        "metrics": root / "metrics.json",
        "merged": root / "merged",
    }
