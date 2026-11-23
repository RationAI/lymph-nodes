"""Fitting a training run to the GPU it is given.

How large a batch fits depends on the GPU (a whole card, a MIG slice) and on the run
(model, precision, tile size), so ``probe`` measures it: real training steps — forward,
backward and optimizer step — on synthetic batches of growing size, and the peak memory
of each, until one runs out or would leave less than ``headroom`` free. Against the
batch the run asks for, ``plan`` names one of three cases:

fits        the batch fits: train as configured.
accumulate  it does not: micro-batches that fit, their gradients summed over
            ``accumulate`` passes before each optimizer step, which makes the update of
            the full batch (exact for heads without BatchNorm, as ours; Lightning
            divides the loss). Every micro-batch is a forward and a backward pass; only
            the weight update waits for all of them.
room        the batch fits several times over: memory for more runs on the same GPU,
            which ``share`` starts, each capped at its share. Whether that is faster
            depends on compute, not memory (one run of a frozen ViT-H backbone may keep
            the GPU busy on its own), so ``benchmark`` measures the throughput of 1, 2,
            ... runs at once: only a measured gain is worth it.

A bigger batch is not offered for the spare room: it changes the training (the
learning rate, the sampler's draws per step), so it is a decision, not free speed.

    python -m lymph_nodes.gpu_plan probe <overrides>      # the largest batch and the plan
    python -m lymph_nodes.gpu_plan benchmark --runs 1 2 3 [--seconds 60] -- <overrides>
    python -m lymph_nodes.gpu_plan share -- "<command>" "<command>" ...

``<overrides>`` are the training run's Hydra overrides (``+data=... +experiment=...``).
In training, ``gpu_plan.enabled=true`` applies the plan before fitting (lymph_nodes
``__main__``) and logs it with the run's parameters.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from torch import nn

# Set by ``share`` for each run it starts: the share of the GPU's memory it may use.
FRACTION_ENV = "LN_GPU_MEMORY_FRACTION"
# What a process needs besides PyTorch's allocations (CUDA context, cuDNN workspaces),
# when counting how many runs fit side by side.
CONTEXT_BYTES = 768 * 2**20


class BatchTooLargeError(Exception):
    """A trial batch did not fit."""


@dataclass
class Probe:
    """The largest batch that fits within ``budget`` bytes, and the peaks measured."""

    budget: int
    max_batch: int
    peaks: dict[int, int]  # batch size: peak bytes, for every trial that ran

    def peak(self, batch: int) -> float:
        """Peak bytes at ``batch``: a line through the smallest and largest trials."""
        sizes = sorted(self.peaks)
        low, high = sizes[0], sizes[-1]
        if low == high:
            return float(self.peaks[low])
        slope = (self.peaks[high] - self.peaks[low]) / (high - low)
        return self.peaks[low] + slope * (batch - low)


@dataclass
class Plan:
    """What to train with: ``micro_batch`` x ``accumulate`` per optimizer step."""

    case: str  # "fits", "accumulate" or "room"
    target: int
    micro_batch: int
    accumulate: int
    runs: int  # runs of this size that fit side by side in the memory

    def __str__(self) -> str:
        text = {
            "fits": f"batch {self.target} fits: train as configured",
            "accumulate": f"batch {self.target} does not fit: {self.accumulate} x "
            f"{self.micro_batch} (gradient accumulation)",
            "room": f"batch {self.target} fits {self.runs} times over: room for "
            f"{self.runs} runs side by side, if the benchmark shows a gain",
        }[self.case]
        if (
            self.case == "accumulate"
            and self.micro_batch * self.accumulate != self.target
        ):
            text += f", an effective batch of {self.micro_batch * self.accumulate}"
        return text

    def params(self) -> dict[str, Any]:
        return {
            "gpu_plan/case": self.case,
            "gpu_plan/micro_batch": self.micro_batch,
            "gpu_plan/accumulate": self.accumulate,
            "gpu_plan/runs_that_fit": self.runs,
        }


def probe(
    measure: Callable[[int], int], budget: int, start: int = 8, limit: int = 8192
) -> Probe:
    """The largest batch whose ``measure`` (peak bytes, or BatchTooLargeError) is within ``budget``.

    Doubles from ``start`` until a trial fails or ``limit``, then bisects to ~6 %.
    """
    peaks: dict[int, int] = {}

    def fits(batch: int) -> bool:
        try:
            peaks[batch] = measure(batch)
        except BatchTooLargeError:
            return False
        return peaks[batch] <= budget

    if fits(start):
        low, high = start, None
        while high is None and low < limit:
            batch = min(2 * low, limit)
            if fits(batch):
                low = batch
            else:
                high = batch
        if high is None:
            return Probe(budget, low, peaks)
    else:
        low, high = 0, start
    while high - low > max(1, low // 16):
        middle = (low + high) // 2
        if fits(middle):
            low = middle
        else:
            high = middle
    if low == 0:
        raise BatchTooLargeError(
            f"not even a batch of 1 fits in {budget / 2**30:.1f} GiB"
        )
    return Probe(budget, low, peaks)


def plan(target: int, measured: Probe) -> Plan:
    """The case of a run that asks for batches of ``target`` (see the module docs)."""
    if measured.max_batch >= target:
        runs = max(1, int(measured.budget // (measured.peak(target) + CONTEXT_BYTES)))
        return Plan("fits" if runs < 2 else "room", target, target, 1, runs)
    fewest = math.ceil(target / measured.max_batch)
    for accumulate in range(fewest, 4 * fewest + 1):  # an exact split, if one is near
        if target % accumulate == 0 and target // accumulate <= measured.max_batch:
            return Plan("accumulate", target, target // accumulate, accumulate, 1)
    return Plan("accumulate", target, math.ceil(target / fewest), fewest, 1)


def budget_bytes(device: torch.device, headroom: float) -> int:
    """What this process may use: its share (``FRACTION_ENV``) less ``headroom``."""
    total = torch.cuda.get_device_properties(device).total_memory
    fraction = float(os.environ.get(FRACTION_ENV, 1.0))
    return int(total * fraction * (1 - headroom))


def cuda_measure(
    step: Callable[[int], None], device: torch.device
) -> Callable[[int], int]:
    """A probe's ``measure``: the peak memory PyTorch reserves for two ``step``s.

    Two, so the second runs with the optimizer's state in place, as in training.
    """

    def measure(batch: int) -> int:
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        failed = False
        try:
            step(batch)
            step(batch)
            torch.cuda.synchronize(device)
        except (torch.OutOfMemoryError, RuntimeError) as error:
            # On a MIG slice the allocator's out-of-memory path asks NVML for the
            # device's memory, which fails: a RuntimeError ("NVML_SUCCESS == r
            # INTERNAL ASSERT FAILED") instead of an OutOfMemoryError.
            message = str(error)
            if not isinstance(error, torch.OutOfMemoryError) and not (
                "NVML_SUCCESS" in message or "out of memory" in message
            ):
                raise
            failed = True
        if failed:  # outside the except: its traceback no longer holds the tensors
            torch.cuda.empty_cache()
            raise BatchTooLargeError(batch)
        return torch.cuda.max_memory_reserved(device)

    return measure


def segmentation_step(
    model: nn.Module, grid: int, device: torch.device, precision: str, tile: int = 224
) -> Callable[[int], None]:
    """A training step of a SegmentationMetaArch on a synthetic batch of a given size.

    Uint8 tiles (B, tile, tile, 3) and targets (B, grid, grid), as the data gives them;
    forward, loss, backward and an optimizer step, under the run's autocast.
    """
    if not hasattr(model, "_logits"):
        raise NotImplementedError(
            "gpu_plan probes segmentation models (SegmentationMetaArch)"
        )
    model.to(device).train()
    # The run's optimizer state: AdamW over every trainable weight (the head, LoRA's),
    # as configure_optimizers builds it — not that method itself, whose learning-rate
    # schedule needs an attached Trainer (estimated_stepping_batches), which the probe
    # before fitting doesn't have.
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
    dtype = {"bf16-mixed": torch.bfloat16, "16-mixed": torch.float16}.get(
        str(precision)
    )

    def step(batch: int) -> None:
        tiles = torch.randint(
            0, 256, (batch, tile, tile, 3), dtype=torch.uint8, device=device
        )
        grids = torch.rand(batch, grid, grid, device=device)
        with torch.autocast(
            "cuda", dtype=dtype or torch.float32, enabled=dtype is not None
        ):
            loss = model.criterion(model._logits(tiles, grids), grids)  # type: ignore[operator]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return step


def throughput(
    step: Callable[[int], None], batch: int, seconds: float, device: torch.device
) -> float:
    """Training samples per second at ``batch`` (after a few warm-up steps)."""
    for _ in range(3):
        step(batch)
    torch.cuda.synchronize(device)
    steps, start = 0, time.perf_counter()
    while time.perf_counter() - start < seconds:
        step(batch)
        steps += 1
    torch.cuda.synchronize(device)
    return steps * batch / (time.perf_counter() - start)


def apply(config: Any, model: nn.Module, data: Any) -> Plan:
    """Probe and plan before fitting; set the data's batch size to the plan's micro-batch.

    The probe trains the head on synthetic batches, so its weights are restored after.
    """
    settings = config.gpu_plan
    device = torch.device("cuda")
    saved = {name: value.detach().clone() for name, value in model.state_dict().items()}
    grid = int(config.data.get("train", {}).get("grid", 16))
    step = segmentation_step(model, grid, device, config.trainer.get("precision", "32"))
    measured = probe(
        cuda_measure(step, device), budget_bytes(device, settings.headroom)
    )
    del step
    model.load_state_dict(saved)
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    result = plan(int(config.data.batch_size), measured)
    print(f"gpu_plan: largest batch {measured.max_batch}; {result}", flush=True)
    data.batch_size = result.micro_batch
    return result


def _model(overrides: Sequence[str]) -> tuple[Any, nn.Module]:
    """The run's config (fit mode) and its model, from Hydra overrides."""
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    from lymph_nodes.resolvers import register_resolvers

    register_resolvers()
    configs = Path(__file__).resolve().parent.parent / "configs"
    with initialize_config_dir(config_dir=str(configs), version_base=None):
        config = compose("lymph_nodes", ["mode=fit", *overrides])
    return config, instantiate(config.model)


def limit_memory() -> None:
    """Hold this process to its share of the GPU (``share`` sets it)."""
    if FRACTION_ENV in os.environ and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(float(os.environ[FRACTION_ENV]))


def share(commands: Sequence[str], logs: Path) -> int:
    """Runs ``commands`` side by side on this GPU, each with an equal share of its memory."""
    logs.mkdir(parents=True, exist_ok=True)
    fraction = 0.98 / len(commands)  # a little left for the CUDA contexts
    env = {**os.environ, FRACTION_ENV: f"{fraction:.4f}"}
    processes = []
    for index, command in enumerate(commands):
        log = (logs / f"run{index}.log").open("w")
        processes.append(
            subprocess.Popen(
                command, shell=True, env=env, stdout=log, stderr=subprocess.STDOUT
            )
        )
        print(
            f"run {index} ({fraction:.0%} of the memory, log {log.name}): {command}",
            flush=True,
        )
    codes = [process.wait() for process in processes]
    for index, code in enumerate(codes):
        print(f"run {index}: exit {code}", flush=True)
    return max(codes, key=abs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("probe", "throughput"):
        single_run = sub.add_parser(name)
        single_run.add_argument("--headroom", type=float, default=0.1)
        single_run.add_argument("--batch", type=int, default=None)
        single_run.add_argument("--seconds", type=float, default=60)
        single_run.add_argument("overrides", nargs="*")
    bench = sub.add_parser("benchmark")
    bench.add_argument("--runs", type=int, nargs="+", default=[1, 2])
    bench.add_argument("--seconds", type=float, default=60)
    bench.add_argument("--logs", type=Path, default=Path("gpu_plan_logs"))
    bench.add_argument("overrides", nargs="*")
    shared = sub.add_parser("share")
    shared.add_argument("--logs", type=Path, default=Path("gpu_plan_logs"))
    shared.add_argument("commands", nargs="+")
    args = parser.parse_args()

    if args.command == "share":
        sys.exit(share(args.commands, args.logs))
    if args.command == "benchmark":
        base = [
            sys.executable,
            "-m",
            "lymph_nodes.gpu_plan",
            "throughput",
            "--seconds",
            str(args.seconds),
            *args.overrides,
        ]
        single = None
        for runs in args.runs:
            command = " ".join(shlex.quote(part) for part in base)
            share([command] * runs, args.logs / f"{runs}_runs")
            rates: list[float] = []
            for index in range(runs):
                found = re.findall(
                    r"samples/s ([0-9.]+)",
                    (args.logs / f"{runs}_runs" / f"run{index}.log").read_text(),
                )
                rates.append(float(found[-1]) if found else 0.0)
            if not all(rates):  # a run without a rate failed, e.g. its share too small
                print(
                    f"{runs} run(s): failed, see {args.logs / f'{runs}_runs'}",
                    flush=True,
                )
                continue
            total = sum(rates)
            single = single or total
            print(
                f"{runs} run(s): {' + '.join(f'{r:.0f}' for r in rates)} = {total:.0f} samples/s "
                f"(x{total / single:.2f} of one run)",
                flush=True,
            )
        return

    limit_memory()
    config, model = _model(args.overrides)
    device = torch.device("cuda")
    step = segmentation_step(
        model,
        int(config.data.get("train", {}).get("grid", 16)),
        device,
        config.trainer.get("precision", "32"),
    )
    target = args.batch or int(config.data.batch_size)
    if args.command == "throughput":
        print(
            f"samples/s {throughput(step, target, args.seconds, device):.1f}",
            flush=True,
        )
        return
    measured = probe(cuda_measure(step, device), budget_bytes(device, args.headroom))
    gib = 2**30
    print(
        f"budget {measured.budget / gib:.1f} GiB; largest batch {measured.max_batch}; "
        f"peak at the configured {target}: {measured.peak(target) / gib:.1f} GiB",
        flush=True,
    )
    for batch, peak in sorted(measured.peaks.items()):
        print(f"  batch {batch:5d}: {peak / gib:.2f} GiB", flush=True)
    print(plan(target, measured), flush=True)


if __name__ == "__main__":
    main()
