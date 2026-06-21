import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class CompletedStage:
    stage: int
    start_step: int
    end_step: int
    adapter_path: str


@dataclass
class ActiveStage:
    stage: int
    start_step: int
    end_step: int
    latest_checkpoint: Optional[str] = None


@dataclass
class RunState:
    base_model: str
    activation_type: str
    stage_size: int
    total_steps: int
    lora_rank: int
    completed_stages: list[CompletedStage] = field(default_factory=list)
    active_stage: Optional[ActiveStage] = None


def build_initial_run_state(base_model, activation_type, stage_size, total_steps, lora_rank):
    return RunState(
        base_model=base_model,
        activation_type=activation_type,
        stage_size=stage_size,
        total_steps=total_steps,
        lora_rank=lora_rank,
        completed_stages=[],
        active_stage=ActiveStage(stage=1, start_step=0, end_step=stage_size, latest_checkpoint=None),
    )


def save_run_state(state: RunState, path):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_target = target.with_name(f".{target.name}.tmp")
    with tmp_target.open("w", encoding="utf-8") as handle:
        json.dump(asdict(state), handle, indent=2)
    os.replace(tmp_target, target)


def load_run_state(path):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    completed = [CompletedStage(**item) for item in payload.get("completed_stages", [])]
    active_payload = payload.get("active_stage")
    active = ActiveStage(**active_payload) if active_payload else None

    return RunState(
        base_model=payload["base_model"],
        activation_type=payload["activation_type"],
        stage_size=payload["stage_size"],
        total_steps=payload["total_steps"],
        lora_rank=payload["lora_rank"],
        completed_stages=completed,
        active_stage=active,
    )


def mark_checkpoint(state, checkpoint_path):
    if state.active_stage is None:
        raise ValueError("Cannot mark checkpoint without an active stage")
    state.active_stage.latest_checkpoint = str(checkpoint_path)


def complete_active_stage(state, adapter_path):
    if state.active_stage is None:
        raise ValueError("No active stage to complete")

    finished = CompletedStage(
        stage=state.active_stage.stage,
        start_step=state.active_stage.start_step,
        end_step=state.active_stage.end_step,
        adapter_path=str(adapter_path),
    )
    state.completed_stages.append(finished)

    next_start = state.active_stage.end_step
    next_end = min(next_start + state.stage_size, state.total_steps)
    if next_start >= state.total_steps:
        state.active_stage = None
    else:
        state.active_stage = ActiveStage(
            stage=finished.stage + 1,
            start_step=next_start,
            end_step=next_end,
            latest_checkpoint=None,
        )
