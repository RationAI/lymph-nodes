from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from lightning import LightningModule, Trainer

from lightning import Callback


_GIB = 2**30


class SystemMetrics(Callback):
    """The job's own resource use, logged to its MLflow run every ``interval_s`` seconds.

    MLflow's system metrics (``mlflow.enable_system_metrics_logging``) apply to runs of
    ``mlflow.start_run``, not to the Lightning logger's, and read the CPU and memory with
    psutil, which inside a Kubernetes pod sees the whole node; its NVML GPU monitor can't
    read a MIG slice's utilisation. Here, from the pod's cgroup (v2, else v1) and torch:

    - ``system/cpu_cores_used`` (over the last interval) and ``system/cpu_cores_limit``;
    - ``system/memory_used_gib`` (the cgroup's: what the limit counts),
      ``system/memory_anon_gib`` (the process memory an OOM kill is about),
      ``system/memory_file_gib`` (page cache, reclaimable), ``system/memory_peak_gib``
      and ``system/memory_limit_gib``;
    - ``system/gpu_memory_allocated_gib``, ``..._reserved_gib`` and
      ``..._max_reserved_gib`` (torch's caching allocator, this process; on MIG too).

    Under ``system/`` the MLflow UI shows them in a run's System Metrics. The GPU's
    utilisation (DCGM_FI_PROF_GR_ENGINE_ACTIVE) is in Grafana / Prometheus only. Logged by
    a daemon thread from setup to teardown, on the global-zero rank; a metric the
    machine doesn't offer (e.g. no cgroup on a laptop) is left out.
    """

    def __init__(
        self, interval_s: float = 30.0, cgroup_root: str = "/sys/fs/cgroup"
    ) -> None:
        self.interval_s = interval_s
        self.cgroup_root = Path(cgroup_root)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._step = 0
        self._last_cpu: tuple[float, float] | None = None  # (usage seconds, wall time)

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        logger = trainer.logger
        run_id = getattr(logger, "run_id", None)
        if self._thread is not None or not trainer.is_global_zero or run_id is None:
            return
        client = logger.experiment  # type: ignore[union-attr]
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, args=(client, run_id), daemon=True, name="system-metrics"
        )
        self._thread.start()

    def teardown(
        self, trainer: Trainer, pl_module: LightningModule, stage: str
    ) -> None:
        self._halt()

    def on_exception(
        self, trainer: Trainer, pl_module: LightningModule, exception: BaseException
    ) -> None:
        self._halt()

    def _halt(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=self.interval_s + 10)
        self._thread = None

    def _loop(self, client: Any, run_id: str) -> None:
        from mlflow.entities import Metric

        self.sample()  # primes the CPU counter
        while not self._stop.wait(self.interval_s):
            values = self.sample()
            if not values:
                continue
            now = int(time.time() * 1000)
            try:
                client.log_batch(
                    run_id,
                    metrics=[
                        Metric(f"system/{key}", value, now, self._step)
                        for key, value in values.items()
                    ],
                )
            except Exception as error:  # noqa: BLE001  (monitoring must not stop a run)
                print(f"system metrics not logged: {error}")
            self._step += 1

    def sample(self) -> dict[str, float]:
        """The current values (those this machine offers)."""
        values: dict[str, float] = {}
        cpu = self._cpu_seconds()
        now = time.monotonic()
        if cpu is not None:
            if self._last_cpu is not None and now > self._last_cpu[1]:
                values["cpu_cores_used"] = (cpu - self._last_cpu[0]) / (
                    now - self._last_cpu[1]
                )
            self._last_cpu = (cpu, now)
        limit = self._cpu_limit()
        if limit is not None:
            values["cpu_cores_limit"] = limit
        values.update(self._memory())
        values.update(_gpu_memory())
        return values

    def _read(self, *names: str) -> str | None:
        for name in names:
            path = self.cgroup_root / name
            try:
                return path.read_text().strip()
            except OSError:
                continue
        return None

    def _cpu_seconds(self) -> float | None:
        stat = self._read("cpu.stat")
        if stat is not None:
            for line in stat.splitlines():
                key, _, value = line.partition(" ")
                if key == "usage_usec":
                    return int(value) / 1e6
        usage = self._read("cpuacct/cpuacct.usage", "cpuacct.usage")  # v1, ns
        return None if usage is None else int(usage) / 1e9

    def _cpu_limit(self) -> float | None:
        quota = self._read("cpu.max")
        if quota is not None:
            limit, _, period = quota.partition(" ")
            return None if limit == "max" else int(limit) / int(period)
        v1_quota = self._read("cpu/cpu.cfs_quota_us", "cpu.cfs_quota_us")
        v1_period = self._read("cpu/cpu.cfs_period_us", "cpu.cfs_period_us")
        if v1_quota is None or v1_period is None or int(v1_quota) < 0:
            return None
        return int(v1_quota) / int(v1_period)

    def _memory(self) -> dict[str, float]:
        values: dict[str, float] = {}
        used = self._read("memory.current", "memory/memory.usage_in_bytes")
        if used is not None:
            values["memory_used_gib"] = int(used) / _GIB
        peak = self._read("memory.peak", "memory/memory.max_usage_in_bytes")
        if peak is not None:
            values["memory_peak_gib"] = int(peak) / _GIB
        limit = self._read("memory.max", "memory/memory.limit_in_bytes")
        if limit is not None and limit != "max" and int(limit) < 2**60:
            values["memory_limit_gib"] = int(limit) / _GIB
        stat = self._read("memory.stat", "memory/memory.stat")
        if stat is not None:
            fields = dict(
                line.split(" ", 1) for line in stat.splitlines() if " " in line
            )
            anon = fields.get("anon", fields.get("rss"))
            file = fields.get("file", fields.get("cache"))
            if anon is not None:
                values["memory_anon_gib"] = int(anon) / _GIB
            if file is not None:
                values["memory_file_gib"] = int(file) / _GIB
        return values


def _gpu_memory() -> dict[str, float]:
    """This process's GPU memory, as torch's caching allocator holds it."""
    try:
        import torch

        if not torch.cuda.is_available() or not torch.cuda.is_initialized():
            return {}
        return {
            "gpu_memory_allocated_gib": torch.cuda.memory_allocated() / _GIB,
            "gpu_memory_reserved_gib": torch.cuda.memory_reserved() / _GIB,
            "gpu_memory_max_reserved_gib": torch.cuda.max_memory_reserved() / _GIB,
        }
    except Exception:  # noqa: BLE001
        return {}
