"""Conservative adaptive learning for the global weighted capacity pool."""

from __future__ import annotations

import hashlib
import os
import time
from typing import Any


GLOBAL_CAPACITY_KEY = "global"
_MIN_CAPACITY = 1.0
_PRESSURE_THRESHOLD = 0.65
_STABLE_THRESHOLD = 0.45
_WORSENING_DELTA = 0.015
_WORSENING_SAMPLES = 3
_STABLE_SAMPLES = 8
_REDUCTION_FACTOR = 0.80
_RECOVERY_STEP = 0.02
_REDUCTION_COOLDOWN = 30.0
_RECOVERY_COOLDOWN = 120.0
_MAX_EVENTS = 240
_MAX_PROFILES = 64
_PROFILE_MIN_OBSERVATIONS = 3


def _number(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
        return parsed if parsed == parsed else default
    except (TypeError, ValueError):
        return default


class AdaptiveCapacityController:
    """Learn a safe capacity from worsening pressure and sustained stability."""

    def __init__(self, resource_state: dict[str, Any]):
        self._state = resource_state
        learned = self._state.get("learned_safe_capacity")
        if not isinstance(learned, dict):
            learned = {}
            self._state["learned_safe_capacity"] = learned
        self._learned = learned
        profiles = self._state.get("capacity_profiles")
        if not isinstance(profiles, dict):
            profiles = {}
            self._state["capacity_profiles"] = profiles
        self._profiles = profiles

    @staticmethod
    def _band(value: float, cutoffs: tuple[float, ...], labels: tuple[str, ...]) -> str:
        for cutoff, label in zip(cutoffs, labels):
            if value < cutoff:
                return label
        return labels[-1]

    @classmethod
    def resource_pattern(cls, snapshot: dict[str, Any]) -> dict[str, Any]:
        """Return a coarse, privacy-safe resource/process state fingerprint."""
        memory = snapshot.get("memory", {})
        swap = snapshot.get("swap", {})
        cpu = snapshot.get("cpu", {})
        io = snapshot.get("io", {}).get("psi", {})
        cpu_count = max(1.0, _number(cpu.get("count"), 1.0))
        load_ratio = _number(cpu.get("load1")) / cpu_count
        io_pressure = max(_number(io.get("some_avg10")), _number(io.get("full_avg10")))

        executables: list[str] = []
        for process in snapshot.get("top_processes", [])[:3]:
            if not isinstance(process, dict):
                continue
            command = str(process.get("command") or "").strip()
            if not command:
                continue
            executable = os.path.basename(command.split()[0]).lower()[:64]
            if executable:
                executables.append(executable)
        process_names = sorted(set(executables)) or ["none"]
        process_fingerprint = hashlib.sha256(
            "\x00".join(process_names).encode("utf-8")
        ).hexdigest()[:12]

        features = {
            "memory": cls._band(_number(memory.get("used_ratio")), (0.55, 0.75, 0.90), ("low", "moderate", "elevated", "high")),
            "swap": cls._band(_number(swap.get("used_ratio")), (0.01, 0.25, 0.60), ("idle", "light", "moderate", "high")),
            "cpu_load": cls._band(load_ratio, (0.50, 1.00, 1.50), ("light", "busy", "saturated", "overloaded")),
            "io": cls._band(io_pressure, (1.0, 10.0, 25.0), ("idle", "light", "busy", "saturated")),
            "top_processes": process_names,
            "process_fingerprint": process_fingerprint,
        }
        key = (
            f"mem:{features['memory']}|swap:{features['swap']}|"
            f"cpu:{features['cpu_load']}|io:{features['io']}|proc:{process_fingerprint}"
        )
        return {"key": key, "features": features}

    def _profile(self, pattern: dict[str, Any], configured_ceiling: float, now: float) -> dict[str, Any]:
        key = str(pattern["key"])
        profile = self._profiles.get(key)
        if not isinstance(profile, dict):
            profile = {}
            self._profiles[key] = profile
        ceiling = max(_MIN_CAPACITY, float(configured_ceiling))
        profile["pattern_key"] = key
        profile["features"] = dict(pattern["features"])
        profile["safe_capacity"] = round(
            max(_MIN_CAPACITY, min(ceiling, _number(profile.get("safe_capacity"), ceiling))), 3
        )
        profile["observations"] = max(0, int(_number(profile.get("observations"), 0)))
        profile["confidence"] = max(0.0, min(1.0, _number(profile.get("confidence"), 0.0)))
        profile["last_seen_at"] = now
        return profile

    def _prune_profiles(self) -> None:
        if len(self._profiles) <= _MAX_PROFILES:
            return
        candidates = sorted(
            self._profiles.items(), key=lambda item: _number(item[1].get("last_seen_at"))
            if isinstance(item[1], dict) else 0.0,
        )
        for key, _ in candidates[:len(self._profiles) - _MAX_PROFILES]:
            self._profiles.pop(key, None)

    @staticmethod
    def pressure_score(snapshot: dict[str, Any]) -> float:
        memory = snapshot.get("memory", {})
        swap = snapshot.get("swap", {})
        cpu = snapshot.get("cpu", {})
        io = snapshot.get("io", {}).get("psi", {})

        cpu_count = max(1.0, _number(cpu.get("count"), 1.0))
        load_ratio = _number(cpu.get("load1")) / cpu_count
        components = (
            _number(memory.get("used_ratio")),
            min(1.0, _number(swap.get("used_ratio")) * 2.0),
            min(1.0, _number(cpu.get("percent")) / 100.0),
            min(1.0, load_ratio / 1.5),
            min(1.0, _number(io.get("some_avg10")) / 20.0),
            min(1.0, _number(io.get("full_avg10")) / 10.0),
        )
        return round(max(0.0, *components), 4)

    def _entry(self, configured_ceiling: float) -> dict[str, Any]:
        ceiling = max(_MIN_CAPACITY, float(configured_ceiling))
        entry = self._learned.get(GLOBAL_CAPACITY_KEY)
        if not isinstance(entry, dict):
            entry = {}
            self._learned[GLOBAL_CAPACITY_KEY] = entry
        current = _number(entry.get("safe_capacity"), ceiling)
        entry["state_key"] = GLOBAL_CAPACITY_KEY
        entry["safe_capacity"] = round(max(_MIN_CAPACITY, min(ceiling, current)), 3)
        entry["configured_ceiling"] = ceiling
        entry["observations"] = max(0, int(_number(entry.get("observations"), 0)))
        entry["confidence"] = max(0.0, min(1.0, _number(entry.get("confidence"), 0.0)))
        entry["worsening_streak"] = max(0, int(_number(entry.get("worsening_streak"), 0)))
        entry["stable_streak"] = max(0, int(_number(entry.get("stable_streak"), 0)))
        entry["reductions"] = max(0, int(_number(entry.get("reductions"), 0)))
        entry["increases"] = max(0, int(_number(entry.get("increases"), 0)))
        entry["last_adjustment_at"] = max(0.0, _number(entry.get("last_adjustment_at"), 0.0))
        return entry

    def current_capacity(self, configured_ceiling: float) -> float:
        return float(self._entry(configured_ceiling)["safe_capacity"])

    def _record_adjustment(
        self,
        *,
        direction: str,
        now: float,
        old_capacity: float,
        new_capacity: float,
        pressure_score: float,
        stable_samples: int,
        pattern_key: str,
    ) -> None:
        key = "pressure_events" if direction == "down" else "recovery_events"
        events = self._state.setdefault(key, [])
        if not isinstance(events, list):
            events = []
            self._state[key] = events
        events.append({
            "timestamp": now,
            "kind": "capacity_reduction" if direction == "down" else "capacity_increase",
            "old_capacity": old_capacity,
            "new_capacity": new_capacity,
            "pressure_score": pressure_score,
            "stable_samples": stable_samples,
            "pattern_key": pattern_key,
        })
        del events[:-_MAX_EVENTS]

    def observe(
        self,
        snapshot: dict[str, Any],
        configured_ceiling: float,
        now: float | None = None,
    ) -> float:
        entry = self._entry(configured_ceiling)
        ceiling = float(entry["configured_ceiling"])
        now = _number(snapshot.get("timestamp"), time.time()) if now is None else float(now)
        pattern = self.resource_pattern(snapshot)
        pattern_key = str(pattern["key"])
        profile = self._profile(pattern, ceiling, now)
        previous_pattern = entry.get("last_pattern_key")
        previous_features = entry.get("last_pattern_features")
        if not isinstance(previous_features, dict):
            previous_features = {}
        if previous_pattern != pattern_key:
            # Re-entering a known state begins from that state's learned safe
            # capacity, rather than the unrelated state used most recently.
            if profile["observations"] >= _PROFILE_MIN_OBSERVATIONS:
                entry["safe_capacity"] = profile["safe_capacity"]
                entry["worsening_streak"] = 0
                entry["stable_streak"] = 0
                entry.pop("last_pressure_score", None)
            # Resource bands naturally change while pressure builds.  Only a
            # new dominant process signature resets the trend detector when
            # there is no learned profile to restore.
            elif (
                previous_features.get("process_fingerprint")
                and previous_features.get("process_fingerprint")
                != pattern["features"]["process_fingerprint"]
            ):
                entry["worsening_streak"] = 0
                entry["stable_streak"] = 0
                entry.pop("last_pressure_score", None)
        entry["last_pattern_key"] = pattern_key
        entry["last_pattern_features"] = dict(pattern["features"])
        current = float(entry["safe_capacity"])
        score = self.pressure_score(snapshot)
        previous_score = entry.get("last_pressure_score")
        previous_score_num = _number(previous_score, score)

        worsening = (
            previous_score is not None
            and score >= _PRESSURE_THRESHOLD
            and score >= previous_score_num + _WORSENING_DELTA
        )
        stable = score <= _STABLE_THRESHOLD

        if worsening:
            entry["worsening_streak"] += 1
        else:
            entry["worsening_streak"] = 0
        if stable:
            entry["stable_streak"] += 1
        else:
            entry["stable_streak"] = 0

        entry["observations"] += 1
        entry["confidence"] = round(min(1.0, entry["observations"] / 20.0), 3)
        entry["last_pressure_score"] = score
        entry["updated_at"] = now

        adjustment_count = entry["reductions"] + entry["increases"]
        since_adjustment = now - entry["last_adjustment_at"]
        if (
            entry["worsening_streak"] >= _WORSENING_SAMPLES
            and (adjustment_count == 0 or since_adjustment >= _REDUCTION_COOLDOWN)
        ):
            reduced = max(_MIN_CAPACITY, min(current - 1.0, current * _REDUCTION_FACTOR))
            reduced = round(reduced, 3)
            if reduced < current:
                entry["safe_capacity"] = reduced
                entry["reductions"] += 1
                entry["last_adjustment_at"] = now
                entry["last_direction"] = "down"
                self._record_adjustment(
                    direction="down",
                    now=now,
                    old_capacity=current,
                    new_capacity=reduced,
                    pressure_score=score,
                    stable_samples=0,
                    pattern_key=pattern_key,
                )
                current = reduced
            entry["worsening_streak"] = 0
            entry["stable_streak"] = 0
        elif (
            entry["stable_streak"] >= _STABLE_SAMPLES
            and current < ceiling
            and (adjustment_count == 0 or since_adjustment >= _RECOVERY_COOLDOWN)
        ):
            increased = round(min(ceiling, current + max(1.0, current * _RECOVERY_STEP)), 3)
            if increased > current:
                stable_samples = entry["stable_streak"]
                entry["safe_capacity"] = increased
                entry["increases"] += 1
                entry["last_adjustment_at"] = now
                entry["last_direction"] = "up"
                self._record_adjustment(
                    direction="up",
                    now=now,
                    old_capacity=current,
                    new_capacity=increased,
                    pressure_score=score,
                    stable_samples=stable_samples,
                    pattern_key=pattern_key,
                )
                current = increased
            entry["stable_streak"] = 0

        profile["safe_capacity"] = float(entry["safe_capacity"])
        profile["observations"] += 1
        profile["confidence"] = round(min(1.0, profile["observations"] / 20.0), 3)
        profile["updated_at"] = now
        self._prune_profiles()
        return float(entry["safe_capacity"])
