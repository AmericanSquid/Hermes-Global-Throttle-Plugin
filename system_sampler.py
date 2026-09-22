"""Small, dependency-free host resource sampler for Hermes workload control.

The sampler is deliberately observational.  It does not make scheduling
decisions and it never interrupts a running operation.  Linux procfs/cgroup
files are used when available; portable fallbacks keep the plugin harmless on
other platforms.
"""

from __future__ import annotations

import os
import time
from typing import Any

def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()
    except (OSError, UnicodeError):
        return ""


def _read_meminfo() -> dict[str, float]:
    result: dict[str, float] = {}
    for line in _read_text("/proc/meminfo").splitlines():
        parts = line.split()
        if len(parts) >= 2:
            # Linux reports these values in kB.
            result[parts[0].rstrip(":")] = _number(parts[1]) * 1024.0
    return result


def _read_psi(resource: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for line in _read_text(f"/proc/pressure/{resource}").splitlines():
        parts = line.split()
        if not parts:
            continue
        scope = parts[0]
        for item in parts[1:]:
            key, _, value = item.partition("=")
            if key in {"avg10", "avg60", "avg300", "total"}:
                result[f"{scope}_{key}"] = _number(value)
    return result


def _read_cpu_times() -> tuple[float, float] | None:
    line = _read_text("/proc/stat").splitlines()
    if not line:
        return None
    fields = line[0].split()
    if len(fields) < 5 or fields[0] != "cpu":
        return None
    values = [_number(item) for item in fields[1:]]
    idle = values[3] + (values[4] if len(values) > 4 else 0.0)
    return sum(values), idle


def _read_cgroup_cpu() -> dict[str, float]:
    result: dict[str, float] = {}
    for path in ("/sys/fs/cgroup/cpu.stat", "/sys/fs/cgroup/cpu/cpu.stat"):
        text = _read_text(path)
        if not text:
            continue
        for line in text.splitlines():
            key, _, value = line.partition(" ")
            if key in {"nr_throttled", "throttled_usec", "usage_usec"}:
                result[key] = _number(value)
        if result:
            break
    return result


def _read_process(pid: str, previous: dict[str, tuple[float, float]], now: float) -> dict[str, Any] | None:
    stat = _read_text(f"/proc/{pid}/stat")
    if not stat or ")" not in stat:
        return None
    fields = stat.rsplit(")", 1)[1].split()
    if len(fields) <= 19:
        return None
    try:
        # After the comm field, utime/stime are indexes 11/12 in proc stat.
        cpu_ticks = _number(fields[11]) + _number(fields[12])
        start_ticks = _number(fields[19])
    except (IndexError, ValueError):
        return None

    rss = 0.0
    for line in _read_text(f"/proc/{pid}/status").splitlines():
        if line.startswith("VmRSS:"):
            parts = line.split()
            if len(parts) >= 2:
                rss = _number(parts[1]) * 1024.0
            break

    old = previous.get(pid)
    cpu_percent = 0.0
    if old is not None and now > old[1]:
        cpu_percent = max(0.0, (cpu_ticks - old[0]) / (now - old[1]) / max(1, os.cpu_count() or 1) * 100.0 / 100.0)
    previous[pid] = (cpu_ticks, now)
    cmdline = _read_text(f"/proc/{pid}/cmdline").replace("\x00", " ").strip()
    if not cmdline:
        cmdline = _read_text(f"/proc/{pid}/comm").strip()
    return {"pid": int(pid), "command": cmdline[:160], "cpu_percent": round(cpu_percent, 2), "rss_bytes": int(rss)}


class SystemResourceSampler:
    """Collect and persist bounded host telemetry without external packages."""

    _MAX_EVENTS = 240

    def __init__(self, telemetry_state: dict[str, Any], interval_seconds: float = 15.0):
        self._state = telemetry_state
        self._interval = max(1.0, float(interval_seconds))
        self._last_sample_at = 0.0
        self._previous_cpu: tuple[float, float] | None = None
        self._previous_processes: dict[str, tuple[float, float]] = {}
        self._normalize_storage()

    def _normalize_storage(self) -> None:
        for key in ("history", "pressure_events", "recovery_events"):
            value = self._state.get(key)
            if not isinstance(value, list):
                self._state[key] = []
            else:
                del value[:-self._MAX_EVENTS]
        if not isinstance(self._state.get("active_pressure"), dict):
            self._state["active_pressure"] = {}
        if not isinstance(self._state.get("learned_safe_capacity"), dict):
            self._state["learned_safe_capacity"] = {}

    @staticmethod
    def _pressure_signals(
        snapshot: dict[str, Any], previous: dict[str, Any] | None = None
    ) -> dict[str, dict[str, Any]]:
        memory = snapshot.get("memory", {})
        swap = snapshot.get("swap", {})
        cpu = snapshot.get("cpu", {})
        io = snapshot.get("io", {}).get("psi", {})
        throttling = snapshot.get("throttling", {}).get("cgroup_cpu", {})
        signals: dict[str, dict[str, Any]] = {}

        memory_ratio = _number(memory.get("used_ratio"))
        if memory_ratio >= 0.90:
            signals["memory"] = {"severity": round(memory_ratio, 4), "used_ratio": memory_ratio}

        swap_ratio = _number(swap.get("used_ratio"))
        if _number(swap.get("total_bytes")) > 0 and swap_ratio >= 0.25:
            signals["swap"] = {"severity": round(swap_ratio, 4), "used_ratio": swap_ratio}

        cpu_count = max(1.0, _number(cpu.get("count"), 1.0))
        load_ratio = _number(cpu.get("load1")) / cpu_count
        if load_ratio >= 1.5:
            signals["cpu"] = {"severity": round(load_ratio, 4), "load_ratio": round(load_ratio, 4)}

        io_pressure = max(_number(io.get("some_avg10")), _number(io.get("full_avg10")))
        if io_pressure >= 10.0:
            signals["io"] = {"severity": round(io_pressure / 100.0, 4), "psi_avg10": io_pressure}

        previous = previous if isinstance(previous, dict) else {}
        previous_throttle = previous.get("throttling", {}).get("cgroup_cpu", {})
        throttle_delta = _number(throttling.get("nr_throttled")) - _number(previous_throttle.get("nr_throttled"))
        if throttle_delta > 0:
            signals["throttling"] = {"severity": 1.0, "nr_throttled_delta": int(throttle_delta)}
        return signals

    def _update_pressure_events(
        self, snapshot: dict[str, Any], previous: dict[str, Any] | None = None
    ) -> None:
        now = _number(snapshot.get("timestamp"), time.time())
        signals = self._pressure_signals(snapshot, previous)
        active = self._state["active_pressure"]
        pressure_events = self._state["pressure_events"]
        recovery_events = self._state["recovery_events"]

        for kind, details in signals.items():
            current = active.get(kind)
            if not isinstance(current, dict):
                active[kind] = {
                    "started_at": now,
                    "last_seen_at": now,
                    "peak_severity": details.get("severity", 0.0),
                }
                pressure_events.append({
                    "timestamp": now,
                    "kind": kind,
                    "severity": details.get("severity", 0.0),
                    "details": details,
                })
            else:
                current["last_seen_at"] = now
                current["peak_severity"] = max(
                    _number(current.get("peak_severity")), _number(details.get("severity"))
                )

        for kind in list(active):
            if kind in signals:
                continue
            current = active.pop(kind)
            started = _number(current.get("started_at"), now)
            recovery_events.append({
                "timestamp": now,
                "kind": kind,
                "duration_seconds": round(max(0.0, now - started), 3),
                "peak_severity": _number(current.get("peak_severity")),
            })
        del pressure_events[:-self._MAX_EVENTS]
        del recovery_events[:-self._MAX_EVENTS]

    def record_pressure_event(
        self, kind: str, severity: float, details: dict[str, Any] | None = None, timestamp: float | None = None
    ) -> dict[str, Any]:
        event = {
            "timestamp": time.time() if timestamp is None else float(timestamp),
            "kind": str(kind),
            "severity": max(0.0, float(severity)),
            "details": dict(details or {}),
        }
        self._state["pressure_events"].append(event)
        del self._state["pressure_events"][:-self._MAX_EVENTS]
        return event

    def record_recovery_event(
        self, kind: str, duration_seconds: float, peak_severity: float = 0.0, timestamp: float | None = None
    ) -> dict[str, Any]:
        event = {
            "timestamp": time.time() if timestamp is None else float(timestamp),
            "kind": str(kind),
            "duration_seconds": max(0.0, float(duration_seconds)),
            "peak_severity": max(0.0, float(peak_severity)),
        }
        self._state["recovery_events"].append(event)
        del self._state["recovery_events"][:-self._MAX_EVENTS]
        return event

    def record_learned_safe_capacity(
        self,
        state_key: str,
        safe_capacity: float,
        confidence: float = 0.0,
        observations: int = 0,
        timestamp: float | None = None,
    ) -> dict[str, Any]:
        entry = {
            "state_key": str(state_key),
            "safe_capacity": max(0.0, float(safe_capacity)),
            "confidence": max(0.0, min(1.0, float(confidence))),
            "observations": max(0, int(observations)),
            "updated_at": time.time() if timestamp is None else float(timestamp),
        }
        self._state["learned_safe_capacity"][str(state_key)] = entry
        return entry

    def maybe_sample(self, force: bool = False) -> dict[str, Any] | None:
        now = time.time()
        if not force and now - self._last_sample_at < self._interval:
            return None
        snapshot = self.sample(now)
        previous = self._state.get("latest")
        self._update_pressure_events(snapshot, previous if isinstance(previous, dict) else None)
        self._last_sample_at = now
        self._state["latest"] = snapshot
        history = self._state.setdefault("history", [])
        if not isinstance(history, list):
            history = []
            self._state["history"] = history
        history.append(snapshot)
        del history[:-120]
        return snapshot

    def sample(self, now: float | None = None) -> dict[str, Any]:
        now = time.time() if now is None else now
        mem = _read_meminfo()
        total = max(0.0, mem.get("MemTotal", 0.0))
        available = max(0.0, mem.get("MemAvailable", mem.get("MemFree", 0.0)))
        swap_total = max(0.0, mem.get("SwapTotal", 0.0))
        swap_free = max(0.0, mem.get("SwapFree", 0.0))

        cpu = _read_cpu_times()
        cpu_percent = 0.0
        if cpu and self._previous_cpu:
            total_delta = cpu[0] - self._previous_cpu[0]
            idle_delta = cpu[1] - self._previous_cpu[1]
            if total_delta > 0:
                cpu_percent = max(0.0, min(100.0, (1.0 - idle_delta / total_delta) * 100.0))
        self._previous_cpu = cpu
        loads = os.getloadavg() if hasattr(os, "getloadavg") else (0.0, 0.0, 0.0)

        processes: list[dict[str, Any]] = []
        try:
            pids = [name for name in os.listdir("/proc") if name.isdigit()]
            for pid in pids:
                process = _read_process(pid, self._previous_processes, now)
                if process:
                    processes.append(process)
        except OSError:
            pass
        processes.sort(key=lambda item: (item["cpu_percent"], item["rss_bytes"]), reverse=True)

        previous = self._state.get("latest", {})
        previous_cpu = previous.get("cpu", {}) if isinstance(previous, dict) else {}
        previous_memory = previous.get("memory", {}) if isinstance(previous, dict) else {}
        previous_swap = previous.get("swap", {}) if isinstance(previous, dict) else {}

        snapshot = {
            "timestamp": now,
            "memory": {
                "total_bytes": int(total),
                "available_bytes": int(available),
                "used_ratio": round(max(0.0, min(1.0, 1.0 - available / total)) if total else 0.0, 4),
            },
            "swap": {
                "total_bytes": int(swap_total),
                "used_bytes": int(max(0.0, swap_total - swap_free)),
                "used_ratio": round(max(0.0, min(1.0, 1.0 - swap_free / swap_total)) if swap_total else 0.0, 4),
                "psi": _read_psi("memory"),
            },
            "cpu": {
                "percent": round(cpu_percent, 2),
                "load1": round(_number(loads[0]), 3),
                "load5": round(_number(loads[1]), 3),
                "load15": round(_number(loads[2]), 3),
                "count": os.cpu_count() or 1,
            },
            "io": {"psi": _read_psi("io")},
            "throttling": {"cgroup_cpu": _read_cgroup_cpu()},
            "top_processes": processes[:5],
            "trends": {
                "cpu_percent_delta": round(cpu_percent - _number(previous_cpu.get("percent")), 2),
                "load1_delta": round(_number(loads[0]) - _number(previous_cpu.get("load1")), 3),
                "memory_used_ratio_delta": round(
                    (max(0.0, min(1.0, 1.0 - available / total)) if total else 0.0)
                    - _number(previous_memory.get("used_ratio")),
                    4,
                ),
                "swap_used_ratio_delta": round(
                    (max(0.0, min(1.0, 1.0 - swap_free / swap_total)) if swap_total else 0.0)
                    - _number(previous_swap.get("used_ratio")),
                    4,
                ),
            },
        }
        return snapshot
