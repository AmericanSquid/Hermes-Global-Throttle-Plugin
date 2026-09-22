from __future__ import annotations

import copy
import math
import random
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable
from urllib.parse import urlparse

try:
    from .system_sampler import SystemResourceSampler
except (ImportError, ModuleNotFoundError):
    from system_sampler import SystemResourceSampler

try:
    from .capacity_pool import DEFAULT_WORKLOAD_WEIGHTS, WeightedCapacityPool
except (ImportError, ModuleNotFoundError):
    from capacity_pool import DEFAULT_WORKLOAD_WEIGHTS, WeightedCapacityPool

try:
    from .adaptive_capacity import AdaptiveCapacityController
except (ImportError, ModuleNotFoundError):
    from adaptive_capacity import AdaptiveCapacityController


PLUGIN_STATE_KEY = "global_throttle_state"
STATE_VERSION = 1
CONFIDENCE_THRESHOLD = 0.70
_NUM_PARTICLES = 30

# Intentionally not user-facing knobs in v1.
_EWMA_ALPHA = 0.20
_INITIAL_AVG_TOTAL_TOKENS = 2000.0
_INITIAL_AVG_OUTPUT_TOKENS = 512.0
_429_MULTIPLIER = 0.80
_MIN_ADAPTIVE_FACTOR = 0.50
_RECOVERY_EVERY_SUCCESSES = 25
_RECOVERY_STEP = 0.02


@dataclass(frozen=True)
class Settings:
    enabled: bool
    rpm: int
    tpm: int
    rpd: int
    quota_hours: float
    max_rpd_wait: float


@dataclass
class PendingRequest:
    provider: str
    model: str
    approx_input_tokens: int
    estimated_total_tokens: int
    created_at: float = field(default_factory=time.time)


RPD_WINDOW_SECONDS = 86400.0  # 24 hours rolling window for daily rate limits
DEFAULT_GEMINI_RPD = 500


def is_gemini_scope(provider: str = "", model: str = "", scope: str = "") -> bool:
    p = (provider or "").strip().lower()
    m = (model or "").strip().lower()
    s = (scope or "").strip().lower()
    if p in ("google", "gemini"):
        return True
    if "gemini" in m:
        return True
    if "gemini" in s or s == "google" or s.startswith("google::") or s.startswith("gemini::"):
        return True
    return False


def get_scope_key(provider: str = "", model: str = "") -> str:
    prov = (provider or "").strip().lower()
    mod = (model or "").strip()
    if prov and prov != "unknown" and mod and mod != "unknown":
        return f"{prov}::{mod}"
    if prov and prov != "unknown":
        return prov
    if mod and mod != "unknown":
        return mod
    return "global"


def is_local_model_execution(
    provider: str = "", base_url: str = "", model: str = ""
) -> bool:
    """Return whether resource-capacity control must leave this model alone."""
    provider_key = (provider or "").strip().lower().replace("_", "-")
    if provider_key in {"ollama", "local", "llama.cpp", "llamacpp", "lm-studio", "lmstudio"}:
        return True
    model_key = (model or "").strip().lower()
    if model_key.startswith(("ollama/", "local/", "llama.cpp/", "llamacpp/")):
        return True
    try:
        hostname = (urlparse(base_url).hostname or "").lower()
    except (TypeError, ValueError):
        hostname = ""
    return hostname in {"localhost", "127.0.0.1", "::1"}


def classify_tool_workload(tool_name: str = "") -> str | None:
    """Map Hermes tools to rough global-capacity workload classes."""
    name = (tool_name or "").strip().lower()
    if name in {"cronjob", "cronjob_manage", "cron", "schedule", "scheduled_task"}:
        return None
    if name in {"delegate_task", "delegate", "subagent", "spawn_subagent"}:
        return "subagent"
    if any(part in name for part in ("api", "http", "web", "fetch", "mcp")):
        return "api_request"
    return "tool_call"


def _num(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    try:
        val = float(value)
        if math.isnan(val) or math.isinf(val):
            return float(default)
        return val
    except (TypeError, ValueError):
        return float(default)


def _positive_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
        return parsed if parsed > 0 else default
    except (TypeError, ValueError):
        return default


def _token_total(usage: Any) -> tuple[int, int]:
    if usage is None:
        return 0, 0

    def get_int(obj: Any, *names: str) -> int:
        for name in names:
            val = None
            if isinstance(obj, dict):
                val = obj.get(name)
            elif hasattr(obj, name):
                val = getattr(obj, name, None)
            if val is not None:
                try:
                    parsed = int(val)
                    if parsed >= 0:
                        return parsed
                except (TypeError, ValueError):
                    pass
        return 0

    input_tokens = get_int(usage, "input_tokens", "prompt_tokens")
    output_tokens = get_int(usage, "output_tokens", "completion_tokens")
    total_tokens = get_int(usage, "total_tokens") or (input_tokens + output_tokens)
    return total_tokens, output_tokens


def _extract_headers(kwargs: dict[str, Any]) -> dict[str, str]:
    raw_headers = kwargs.get("headers")
    if raw_headers is None:
        raw_headers = kwargs.get("response_headers")
    if raw_headers is None:
        resp = kwargs.get("response")
        if isinstance(resp, dict):
            raw_headers = resp.get("headers")
        elif hasattr(resp, "headers"):
            raw_headers = getattr(resp, "headers")
    if raw_headers is None:
        err = kwargs.get("error")
        if hasattr(err, "headers"):
            raw_headers = getattr(err, "headers")
        elif hasattr(err, "response"):
            resp = getattr(err, "response")
            if isinstance(resp, dict):
                raw_headers = resp.get("headers")
            elif hasattr(resp, "headers"):
                raw_headers = getattr(resp, "headers")

    headers_map: dict[str, str] = {}
    if raw_headers is None:
        return headers_map

    if isinstance(raw_headers, dict) or hasattr(raw_headers, "items"):
        for k, v in raw_headers.items():
            headers_map[str(k).lower()] = str(v)
    elif isinstance(raw_headers, (list, tuple)):
        for item in raw_headers:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                headers_map[str(item[0]).lower()] = str(item[1])
    return headers_map


_TPM_PATTERN = re.compile(
    r"\b(tpm|tokens?\s+limit|tokens?\s+per\s+minute|tokens?\s+exceeded|rate_limit_exceeded_tokens)\b",
    re.IGNORECASE,
)
_RPM_PATTERN = re.compile(
    r"\b(rpm|requests?\s+limit|requests?\s+per\s+minute|requests?\s+exceeded|rate_limit_exceeded_requests)\b",
    re.IGNORECASE,
)


def _detect_429_target(
    headers: dict[str, str], error_str: str = "", reason_str: str = ""
) -> str | None:
    if headers:
        for k in (
            "x-ratelimit-type",
            "ratelimit-type",
            "x-ratelimit-resource",
            "ratelimit-resource",
        ):
            val = headers.get(k, "").lower()
            if "token" in val or "tpm" in val:
                return "tpm"
            if "request" in val or "rpm" in val:
                return "rpm"

        rem_tokens: float | None = None
        for k in (
            "x-ratelimit-remaining-tokens",
            "anthropic-ratelimit-tokens-remaining",
            "ratelimit-remaining-tokens",
            "x-ratelimit-tokens-remaining",
            "ratelimit-tokens-remaining",
        ):
            if k in headers:
                try:
                    rem_tokens = float(headers[k])
                    break
                except (ValueError, TypeError):
                    pass

        rem_requests: float | None = None
        for k in (
            "x-ratelimit-remaining-requests",
            "anthropic-ratelimit-requests-remaining",
            "ratelimit-remaining-requests",
            "x-ratelimit-requests-remaining",
            "ratelimit-requests-remaining",
        ):
            if k in headers:
                try:
                    rem_requests = float(headers[k])
                    break
                except (ValueError, TypeError):
                    pass

        if rem_tokens is not None and rem_requests is not None:
            if rem_tokens <= 0 and rem_requests > 0:
                return "tpm"
            if rem_requests <= 0 and rem_tokens > 0:
                return "rpm"
        elif rem_tokens is not None and rem_tokens <= 0:
            return "tpm"
        elif rem_requests is not None and rem_requests <= 0:
            return "rpm"

        reset_tokens = None
        for k in (
            "x-ratelimit-reset-tokens",
            "anthropic-ratelimit-tokens-reset",
            "ratelimit-reset-tokens",
        ):
            if k in headers:
                reset_tokens = headers[k].strip().lower()
                break

        reset_requests = None
        for k in (
            "x-ratelimit-reset-requests",
            "anthropic-ratelimit-requests-reset",
            "ratelimit-reset-requests",
        ):
            if k in headers:
                reset_requests = headers[k].strip().lower()
                break

        zero_resets = {"0", "0s", "0ms", "0.0s", "0.0ms", "0m0s", ""}
        if reset_tokens is not None and reset_requests is not None:
            if reset_requests in zero_resets and reset_tokens not in zero_resets:
                return "tpm"
            if reset_tokens in zero_resets and reset_requests not in zero_resets:
                return "rpm"
        elif reset_tokens is not None and reset_tokens not in zero_resets:
            return "tpm"
        elif reset_requests is not None and reset_requests not in zero_resets:
            return "rpm"

    combined_msg = f"{error_str} {reason_str}"
    has_tpm = bool(_TPM_PATTERN.search(combined_msg))
    has_rpm = bool(_RPM_PATTERN.search(combined_msg))

    if has_tpm and not has_rpm:
        return "tpm"
    if has_rpm and not has_tpm:
        return "rpm"

    return None



def compute_delays(
    *,
    rpm: int,
    tpm: int,
    estimated_tokens: float,
    remaining_requests: int,
    remaining_usage_seconds: float,
    max_rpd_wait: float,
    adaptive_factor: float = 1.0,
) -> dict[str, float]:
    """Pure math used by the scheduler and unit tests."""
    factor = min(1.0, max(_MIN_ADAPTIVE_FACTOR, _num(adaptive_factor, 1.0)))

    rpm_val = _num(rpm, 0.0)
    tpm_val = _num(tpm, 0.0)

    if rpm_val > 0:
        effective_rpm = max(1e-9, rpm_val * factor)
        rpm_delay = 60.0 / effective_rpm
    else:
        rpm_delay = 0.0

    est_tok = max(0.0, _num(estimated_tokens, 0.0))
    if tpm_val > 0 and est_tok > 0:
        effective_tpm = max(1e-9, tpm_val * factor)
        tpm_delay = 60.0 * est_tok / effective_tpm
    else:
        tpm_delay = 0.0

    rem_req = max(0, int(_num(remaining_requests, 0)))
    rem_sec = max(0.0, _num(remaining_usage_seconds, 0.0))
    max_wait = max(0.0, _num(max_rpd_wait, 0.0))

    if rem_req > 0 and rem_sec > 0:
        raw_rpd_delay = rem_sec / rem_req
        rpd_delay = min(raw_rpd_delay, max_wait)
    else:
        raw_rpd_delay = 0.0
        rpd_delay = 0.0

    return {
        "rpm": rpm_delay,
        "tpm": tpm_delay,
        "rpd_raw": raw_rpd_delay,
        "rpd": rpd_delay,
        "final": max(rpm_delay, tpm_delay, rpd_delay),
    }



class BayesianRateLimitLearner:
    """
    Maintains bounded particle distributions for RPM and TPM for each provider/model.
    Derives safe limits and confidence, and serializes directly to/from dicts in ctx.state.
    """

    def __init__(self, state_store: dict[str, Any]):
        self._store = state_store

    @staticmethod
    def _weighted_percentile(particles: list[float], weights: list[float], q: float = 0.10) -> float:
        if not particles:
            return 0.0
        total_w = sum(weights)
        if total_w <= 0:
            return min(particles)
        pairs = sorted(zip(particles, weights), key=lambda x: x[0])
        cum = 0.0
        prev_p = pairs[0][0]
        prev_cum = 0.0
        for p, w in pairs:
            norm_w = w / total_w
            cum += norm_w
            if cum >= q:
                if cum <= prev_cum:
                    return round(p, 2)
                t = (q - prev_cum) / (cum - prev_cum)
                return round(prev_p + t * (p - prev_p), 2)
            prev_p = p
            prev_cum = cum
        return round(pairs[-1][0], 2)

    @staticmethod
    def _init_distribution(prior_val: float, min_val: float, max_val: float, n_particles: int = _NUM_PARTICLES) -> dict[str, Any]:
        prior_val = max(min_val, min(max_val, float(prior_val)))
        particles = [float(prior_val)] * n_particles
        weights = [1.0 / n_particles] * n_particles
        mean, std = BayesianRateLimitLearner._compute_stats(particles, weights)
        safe = max(min_val, min(max_val, prior_val))
        return {
            "particles": particles,
            "weights": weights,
            "min_val": min_val,
            "max_val": max_val,
            "mean": mean,
            "std": std,
            "safe_limit": safe,
            "confidence": 0.0,
        }

    @staticmethod
    def _compute_stats(particles: list[float], weights: list[float]) -> tuple[float, float]:
        total_w = sum(weights)
        if total_w <= 0:
            n = len(particles)
            weights = [1.0 / n] * n
            total_w = 1.0
        mean = sum(p * (w / total_w) for p, w in zip(particles, weights))
        var = sum(((p - mean) ** 2) * (w / total_w) for p, w in zip(particles, weights))
        std = math.sqrt(max(0.0, var))
        return round(mean, 2), round(std, 2)

    def get_or_create_entry(self, provider: str, model: str, default_rpm: int, default_tpm: int) -> dict[str, Any]:
        key = f"{provider}::{model}"
        if key not in self._store or not isinstance(self._store[key], dict):
            now = time.time()
            rpm_dist = self._init_distribution(
                prior_val=default_rpm,
                min_val=1.0,
                max_val=max(1.0, float(default_rpm)),
            )
            tpm_dist = self._init_distribution(
                prior_val=default_tpm,
                min_val=100.0,
                max_val=max(100.0, float(default_tpm)),
            )
            self._store[key] = {
                "provider": provider,
                "model": model,
                "rpm": rpm_dist,
                "tpm": tpm_dist,
                "safe_rpm": int(round(rpm_dist["safe_limit"])),
                "safe_tpm": int(round(tpm_dist["safe_limit"])),
                "safe_ceiling_established": False,
                "established_safe_rpm": None,
                "established_safe_tpm": None,
                "established_confidence": 0.0,
                "established_observation_count": 0,
                "safe_ceiling_established_at": None,
                "confidence": 0.0,
                "observation_count": 0,
                "success_count": 0,
                "successes_since_429": 0,
                "error_429_count": 0,
                "slowdown_until": 0.0,
                "created_at": now,
                "last_updated_at": now,
                "last_429_at": None,
            }
        else:
            entry = self._store[key]
            legacy_establishment_state = "safe_ceiling_established" not in entry
            entry["rpm"]["max_val"] = max(1.0, float(default_rpm))
            entry["tpm"]["max_val"] = max(100.0, float(default_tpm))
            entry["safe_rpm"] = min(int(_positive_int(entry.get("safe_rpm"), default_rpm)), int(default_rpm))
            entry["safe_tpm"] = min(int(_positive_int(entry.get("safe_tpm"), default_tpm)), int(default_tpm))
            entry.setdefault("safe_ceiling_established", False)
            entry.setdefault("established_safe_rpm", None)
            entry.setdefault("established_safe_tpm", None)
            entry.setdefault("established_confidence", 0.0)
            entry.setdefault("established_observation_count", 0)
            entry.setdefault("safe_ceiling_established_at", None)
            if entry["safe_ceiling_established"]:
                entry["established_safe_rpm"] = int(
                    _positive_int(entry.get("established_safe_rpm"), entry["safe_rpm"])
                )
                entry["established_safe_tpm"] = int(
                    _positive_int(entry.get("established_safe_tpm"), entry["safe_tpm"])
                )
            elif legacy_establishment_state:
                self._establish_safe_ceiling(entry, time.time())
        return self._store[key]

    @staticmethod
    def _establish_safe_ceiling(entry: dict[str, Any], now: float) -> None:
        if entry.get("safe_ceiling_established"):
            return
        confidence = _num(entry.get("confidence"), 0.0)
        if confidence < CONFIDENCE_THRESHOLD:
            return
        entry["safe_ceiling_established"] = True
        entry["established_safe_rpm"] = int(entry["safe_rpm"])
        entry["established_safe_tpm"] = int(entry["safe_tpm"])
        entry["established_confidence"] = round(confidence, 3)
        entry["established_observation_count"] = int(entry["observation_count"])
        entry["safe_ceiling_established_at"] = now

    def get_effective_limits(
        self, provider: str, model: str, default_rpm: int, default_tpm: int
    ) -> tuple[int, int]:
        if not provider or provider == "unknown" or not model or model == "unknown":
            return default_rpm, default_tpm
        key = f"{provider}::{model}"
        entry = self._store.get(key)
        if entry is None or not isinstance(entry, dict):
            return default_rpm, default_tpm
        if entry.get("safe_ceiling_established"):
            ceiling_rpm = int(_positive_int(entry.get("established_safe_rpm"), default_rpm))
            ceiling_tpm = int(_positive_int(entry.get("established_safe_tpm"), default_tpm))
            safe_rpm = int(_positive_int(entry.get("safe_rpm"), ceiling_rpm))
            safe_tpm = int(_positive_int(entry.get("safe_tpm"), ceiling_tpm))
            return (
                min(safe_rpm, ceiling_rpm, default_rpm),
                min(safe_tpm, ceiling_tpm, default_tpm),
            )
        if int(_num(entry.get("error_429_count"), 0)) > 0:
            safe_rpm = int(_positive_int(entry.get("safe_rpm"), default_rpm))
            safe_tpm = int(_positive_int(entry.get("safe_tpm"), default_tpm))
            return min(safe_rpm, default_rpm), min(safe_tpm, default_tpm)
        conf = _num(entry.get("confidence"), 0.0)
        if conf >= CONFIDENCE_THRESHOLD:
            safe_rpm = int(_positive_int(entry.get("safe_rpm"), default_rpm))
            safe_tpm = int(_positive_int(entry.get("safe_tpm"), default_tpm))
            return min(safe_rpm, default_rpm), min(safe_tpm, default_tpm)
        return default_rpm, default_tpm

    def allows_upward_exploration(
        self, provider: str, model: str, pending_success: bool = False
    ) -> bool:
        if not provider or provider == "unknown" or not model or model == "unknown":
            return True
        entry = self._store.get(f"{provider}::{model}")
        if not isinstance(entry, dict):
            return True
        if entry.get("safe_ceiling_established"):
            return False
        successes = int(_num(entry.get("successes_since_429"), 0))
        return successes + int(pending_success) >= _RECOVERY_EVERY_SUCCESSES

    def get_slowdown_wait(self, provider: str, model: str, now: float) -> float:
        key = f"{provider}::{model}"
        entry = self._store.get(key)
        if entry and isinstance(entry, dict):
            slowdown = _num(entry.get("slowdown_until"), 0.0)
            return max(0.0, slowdown - now)
        return 0.0

    @staticmethod
    def _update_distribution(
        dist: dict[str, Any],
        observed_load: float,
        is_429: bool,
        obs_count: int,
    ) -> None:
        particles = dist.get("particles", [])
        min_val = _num(dist.get("min_val"), 1.0)
        max_val = _num(dist.get("max_val"), 100000.0)
        n = len(particles)
        if n == 0:
            return

        if not is_429:
            # Upward calibration is deliberately handled by on_success, where
            # it is gated by a complete successful-observation window.
            return

        # Requirement 5, 6, 7:
        # Keep 429 responses as the main signal that the current learned limit is too high and should move downward.
        # A 429 always lowers the active learned limit. Established models do
        # not automatically explore upward again; explicit reset is required.
        prev_safe = min(float(dist.get("safe_limit", max_val)), float(dist.get("mean", max_val)), max_val)
        failure_point = min(prev_safe, observed_load) if observed_load > 0 else prev_safe
        new_safe = max(min_val, failure_point * 0.80)
        if prev_safe > min_val:
            new_safe = min(new_safe, max(min_val, prev_safe - 1.0))

        step = (new_safe - max(min_val, new_safe * 0.85)) / max(1, n - 1) if n > 1 else 0.0
        low = max(min_val, new_safe * 0.85)
        particles = [round(low + i * step, 2) for i in range(n)] if new_safe > low else [round(new_safe, 2)] * n
        weights = [1.0 / n] * n

        dist["particles"] = particles
        dist["weights"] = weights
        dist["max_val"] = new_safe
        mean, std = BayesianRateLimitLearner._compute_stats(particles, weights)
        dist["mean"] = mean
        dist["std"] = std
        dist["safe_limit"] = max(min_val, BayesianRateLimitLearner._weighted_percentile(particles, weights, q=0.10))
        dist["confidence"] = 0.0

    @staticmethod
    def _advance_calibrating_distribution(
        dist: dict[str, Any], configured_ceiling: float
    ) -> None:
        ceiling = max(_num(dist.get("min_val"), 1.0), float(configured_ceiling))
        current = min(ceiling, _num(dist.get("safe_limit"), ceiling))
        if current >= ceiling:
            return

        advanced = min(ceiling, current + max(1.0, current * _RECOVERY_STEP))
        delta = advanced - current
        particles = [
            min(ceiling, float(particle) + delta)
            for particle in dist.get("particles", [])
        ]
        if not particles:
            return
        weights = dist.get("weights", [])
        if len(weights) != len(particles):
            weights = [1.0 / len(particles)] * len(particles)

        dist["particles"] = particles
        dist["weights"] = weights
        dist["max_val"] = ceiling
        mean, std = BayesianRateLimitLearner._compute_stats(particles, weights)
        dist["mean"] = mean
        dist["std"] = std
        dist["safe_limit"] = min(
            ceiling,
            BayesianRateLimitLearner._weighted_percentile(particles, weights, q=0.10),
        )

    def on_success(
        self,
        provider: str,
        model: str,
        actual_tokens: int,
        rolling_requests: int,
        rolling_tokens: int,
        now: float,
        default_rpm: int,
        default_tpm: int,
    ) -> None:
        if not provider or provider == "unknown" or not model or model == "unknown":
            return
        entry = self.get_or_create_entry(provider, model, default_rpm, default_tpm)
        obs = entry["observation_count"] + 1
        entry["observation_count"] = obs
        entry["success_count"] = entry.get("success_count", 0) + 1
        entry["successes_since_429"] = entry.get("successes_since_429", 0) + 1
        entry["last_updated_at"] = now

        # Successful traffic establishes confidence. Calibrating models may
        # take one small step per full success window; established models do
        # not explore upward until reset-learning is requested.
        successes = entry["successes_since_429"]
        obs_factor = min(1.0, successes / 15.0)

        entry["rpm"]["confidence"] = round(obs_factor, 3)
        entry["tpm"]["confidence"] = round(obs_factor, 3)
        entry["confidence"] = round(obs_factor, 3)

        needs_exploration = (
            _num(entry["rpm"].get("safe_limit"), default_rpm) < default_rpm
            or _num(entry["tpm"].get("safe_limit"), default_tpm) < default_tpm
        )
        if (
            not entry.get("safe_ceiling_established")
            and needs_exploration
            and successes >= _RECOVERY_EVERY_SUCCESSES
        ):
            self._advance_calibrating_distribution(entry["rpm"], default_rpm)
            self._advance_calibrating_distribution(entry["tpm"], default_tpm)
            entry["successes_since_429"] = 0
            entry["rpm"]["confidence"] = 0.0
            entry["tpm"]["confidence"] = 0.0
            entry["confidence"] = 0.0

        entry["safe_rpm"] = min(int(round(entry["rpm"]["safe_limit"])), default_rpm)
        entry["safe_tpm"] = min(int(round(entry["tpm"]["safe_limit"])), default_tpm)
        if entry["safe_rpm"] >= default_rpm and entry["safe_tpm"] >= default_tpm:
            self._establish_safe_ceiling(entry, now)

    def on_429(
        self,
        provider: str,
        model: str,
        estimated_tokens: int,
        rolling_requests: int,
        rolling_tokens: int,
        now: float,
        default_rpm: int,
        default_tpm: int,
        target_dimension: str | None = None,
    ) -> str:
        if not provider or provider == "unknown" or not model or model == "unknown":
            return ""
        entry = self.get_or_create_entry(provider, model, default_rpm, default_tpm)
        obs = entry["observation_count"] + 1
        entry["observation_count"] = obs
        entry["error_429_count"] = entry.get("error_429_count", 0) + 1
        entry["last_updated_at"] = now
        entry["last_429_at"] = now
        entry["successes_since_429"] = 0

        observed_rpm = max(1.0, float(rolling_requests))
        observed_tpm = max(float(estimated_tokens), float(rolling_tokens))

        target = target_dimension
        if target not in {"rpm", "tpm"}:
            boundary_rpm = max(1.0, float(entry["rpm"].get("safe_limit") or entry["rpm"].get("mean") or default_rpm))
            boundary_tpm = max(1.0, float(entry["tpm"].get("safe_limit") or entry["tpm"].get("mean") or default_tpm))
            proximity_rpm = observed_rpm / boundary_rpm
            proximity_tpm = observed_tpm / boundary_tpm

            target = "rpm" if proximity_rpm >= proximity_tpm else "tpm"

        if target == "rpm":
            self._update_distribution(entry["rpm"], observed_rpm, is_429=True, obs_count=obs)
        else:
            self._update_distribution(entry["tpm"], observed_tpm, is_429=True, obs_count=obs)

        safe_rpm = min(int(round(entry["rpm"]["safe_limit"])), default_rpm)
        entry["safe_rpm"] = safe_rpm
        entry["safe_tpm"] = min(int(round(entry["tpm"]["safe_limit"])), default_tpm)
        entry["confidence"] = round(min(entry["rpm"]["confidence"], entry["tpm"]["confidence"]), 3)

        entry["slowdown_until"] = now + max(5.0, 60.0 / max(1.0, float(safe_rpm)))
        return target

    def reset_learning(self, provider: str, model: str) -> bool:
        key = f"{provider}::{model}"
        if key in self._store:
            del self._store[key]
            return True
        return False

    def recalibrate_learning(self, provider: str, model: str, now: float | None = None) -> bool:
        key = f"{provider}::{model}"
        entry = self._store.get(key)
        if not isinstance(entry, dict):
            return False
        entry["safe_ceiling_established"] = False
        entry["established_safe_rpm"] = None
        entry["established_safe_tpm"] = None
        entry["established_confidence"] = 0.0
        entry["established_observation_count"] = 0
        entry["safe_ceiling_established_at"] = None
        entry["successes_since_429"] = 0
        entry["rpm"]["confidence"] = 0.0
        entry["tpm"]["confidence"] = 0.0
        entry["confidence"] = 0.0
        entry["last_updated_at"] = time.time() if now is None else float(now)
        return True



class ThrottleBucket:
    """
    FIFO leaky bucket for a specific scope (global, provider, or provider::model).
    Each configured scope maintains its own RPM pacing, TPM ledger, in-flight
    reservations, and daily quota/cooldown state.
    """

    def __init__(
        self,
        key: str,
        scope_type: str,
        state_dict: dict[str, Any],
        controller: GlobalThrottle,
        cv: threading.Condition | None = None,
    ):
        self.key = key
        self.scope_type = scope_type  # "global", "provider", or "model"
        self._state = state_dict
        self._controller = controller
        self._cv = cv if cv is not None else threading.Condition(threading.RLock())

        self._next_ticket = 0
        self._serving_ticket = 0
        self._cancelled_tickets: set[int] = set()

        self._last_dispatch_mono = 0.0
        self._last_dispatch_wall = 0.0
        self._last_wait_seconds = 0.0
        self._last_delay_reason = "startup"

        self._token_ledger: list[tuple[float, int]] = []
        self._reservations: dict[str, int] = {}
        self._burst_credit: dict[str, tuple[float, float]] = {}

    @property
    def token_ledger(self) -> list[tuple[float, int]]:
        with self._cv:
            return list(self._token_ledger)

    @property
    def reservations(self) -> dict[str, int]:
        with self._cv:
            return dict(self._reservations)

    @property
    def adaptive_factor(self) -> float:
        with self._cv:
            return _num(self._state.get("adaptive", {}).get("factor"), 1.0)

    @property
    def cooldown_until(self) -> float:
        with self._cv:
            return _num(self._state.get("adaptive", {}).get("cooldown_until"), 0.0)

    @property
    def error_streak(self) -> int:
        with self._cv:
            return int(_num(self._state.get("adaptive", {}).get("error_streak"), 0))

    @property
    def request_count(self) -> int:
        with self._cv:
            return int(_num(self._state.get("day", {}).get("requests"), 0))

    @property
    def token_count(self) -> int:
        with self._cv:
            return int(_num(self._state.get("day", {}).get("tokens"), 0))

    def claim_ticket(self) -> int:
        with self._cv:
            ticket = self._next_ticket
            self._next_ticket += 1
            return ticket

    def wait_for_ticket(self, ticket: int, timeout: float = 1.0) -> bool:
        with self._cv:
            while ticket != self._serving_ticket:
                self._cv.wait(timeout=timeout)
            return True

    def advance_ticket(self) -> None:
        with self._cv:
            self._serving_ticket += 1
            while self._serving_ticket in self._cancelled_tickets:
                self._cancelled_tickets.remove(self._serving_ticket)
                self._serving_ticket += 1
            self._cv.notify_all()

    def cancel_ticket(self, ticket: int) -> None:
        with self._cv:
            if ticket == self._serving_ticket:
                self._serving_ticket += 1
                while self._serving_ticket in self._cancelled_tickets:
                    self._cancelled_tickets.remove(self._serving_ticket)
                    self._serving_ticket += 1
                self._cv.notify_all()
            elif ticket > self._serving_ticket:
                self._cancelled_tickets.add(ticket)

    def prune_ledger_locked(self, now: float) -> None:
        cutoff = now - 60.0
        self._token_ledger = [
            (ts, tok) for ts, tok in self._token_ledger if ts > cutoff
        ]

    def release_reservation_locked(self, request_id: str) -> None:
        if request_id:
            self._reservations.pop(request_id, None)
        elif len(self._reservations) == 1:
            self._reservations.clear()
        self._reservations.pop("_last", None)

    def release_reservation(self, request_id: str = "") -> None:
        with self._cv:
            self.release_reservation_locked(request_id)
            self._cv.notify_all()

    def ensure_current_day_locked(self) -> None:
        today = datetime.now().date().isoformat()
        day = self._state.setdefault("day", {})
        if day.get("date") != today:
            day.clear()
            day.update({
                "date": today,
                "requests": 0,
                "tokens": 0,
                "first_request_at": None,
            })

    def _check_wait_locked(
        self,
        request_id: str,
        estimated_tokens: int,
        provider: str,
        model: str,
    ) -> tuple[float, str, Settings]:
        # Sampling and learned capacity are shared with tool admission. This
        # updates only future admissions and never interrupts active work.
        self._controller.refresh_resource_capacity()
        settings = self._controller.resolve_limits(
            provider=provider,
            model=model,
            scope=self.key if self.scope_type != "global" else None,
        )
        if not settings.enabled:
            return 0.0, "disabled", settings

        self.ensure_current_day_locked()
        scope_key = self.key if self.scope_type != "global" else get_scope_key(provider, model)
        now_wall = time.time()
        now_mono = time.monotonic()

        rpd_delay, raw_rpd_delay, rolling_count, rpd_limit = (
            self._controller.compute_rpd_pacing_locked(
                scope_key=scope_key,
                settings=settings,
                now=now_wall,
                provider=provider,
                model=model,
            )
        )

        adaptive = self._state.setdefault("adaptive", {})
        adaptive_factor = _num(adaptive.get("factor"), 1.0)

        effective_rpm, effective_tpm = self._controller.learner.get_effective_limits(
            provider=provider,
            model=model,
            default_rpm=settings.rpm,
            default_tpm=settings.tpm,
        )

        delays = compute_delays(
            rpm=effective_rpm,
            tpm=effective_tpm,
            estimated_tokens=estimated_tokens,
            remaining_requests=max(0, rpd_limit - rolling_count),
            remaining_usage_seconds=0.0,
            max_rpd_wait=settings.max_rpd_wait,
            adaptive_factor=adaptive_factor,
        )
        delays["rpd_raw"] = raw_rpd_delay
        delays["rpd"] = rpd_delay
        delays["final"] = max(delays["rpm"], delays["tpm"], rpd_delay)

        # Base RPM/TPM pacing is from this bucket's last dispatch.
        rpm_tpm_delay = max(delays["rpm"], delays["tpm"])
        if self._last_dispatch_mono > 0 and rpm_tpm_delay > 0:
            rpm_tpm_earliest = self._last_dispatch_mono + rpm_tpm_delay
        else:
            rpm_tpm_earliest = now_mono
        rpm_tpm_wait = max(0.0, rpm_tpm_earliest - now_mono)

        # Accrue at most two requests of idle credit for this provider/model.
        # Commit the debit only when the request actually dispatches.
        burst_key = get_scope_key(provider, model)
        if "::" in burst_key and rpm_tpm_delay > 0:
            credit, updated_at = self._burst_credit.get(burst_key, (1.0, now_mono))
            credit = min(2.0, credit + max(0.0, now_mono - updated_at) / rpm_tpm_delay)
            self._burst_credit[burst_key] = (credit, now_mono)
            if adaptive_factor >= 1.0:
                rpm_tpm_wait = max(0.0, (1.0 - credit) * rpm_tpm_delay)

        # Scoped RPD pacing from scope's own dispatch history
        scope_entry = self._controller._get_rpd_scope_entry_locked(scope_key, provider, model)
        scope_last_mono = self._controller._rpd_last_dispatch_mono.get(scope_key, 0.0)
        if scope_last_mono > 0:
            rpd_wait = max(0.0, scope_last_mono + rpd_delay - now_mono)
        else:
            # After a restart, recover only the unelapsed spacing from the
            # persisted wall clock; monotonic values belong to one process.
            scope_last_wall = _num(scope_entry.get("last_dispatch_wall"), 0.0)
            elapsed = (
                max(0.0, now_wall - scope_last_wall)
                if scope_last_wall > 0 else rpd_delay
            )
            rpd_wait = max(0.0, rpd_delay - elapsed)

        pacing_wait = max(rpm_tpm_wait, rpd_wait)

        cooldown_until = _num(adaptive.get("cooldown_until"), 0.0)
        cooldown_wait = max(0.0, cooldown_until - now_wall)

        self.prune_ledger_locked(now_wall)
        used_tokens = sum(tok for _, tok in self._token_ledger)
        reserved_tokens = sum(
            tok for r_id, tok in self._reservations.items() if r_id != request_id
        )
        total_projected = used_tokens + reserved_tokens + estimated_tokens

        tpm_ceiling_wait = 0.0
        if total_projected > effective_tpm and (used_tokens > 0 or reserved_tokens > 0):
            needed_freed = total_projected - effective_tpm
            accumulated = 0
            expire_wait = 0.0
            for entry_ts, entry_tok in sorted(self._token_ledger, key=lambda x: x[0]):
                accumulated += entry_tok
                remaining = max(0.001, (entry_ts + 60.0) - now_wall)
                expire_wait = max(expire_wait, remaining)
                if accumulated >= needed_freed:
                    break
            tpm_ceiling_wait = expire_wait if expire_wait > 0 else 1.0

        slowdown_wait = self._controller.learner.get_slowdown_wait(provider, model, now_wall)
        wait_for = max(pacing_wait, cooldown_wait, tpm_ceiling_wait, slowdown_wait)

        if wait_for > 0.001:
            if slowdown_wait >= pacing_wait and slowdown_wait >= cooldown_wait and slowdown_wait >= tpm_ceiling_wait and slowdown_wait > 0:
                winner = "learned slowdown"
            elif cooldown_wait >= pacing_wait and cooldown_wait >= tpm_ceiling_wait and cooldown_wait > 0:
                winner = "429 cooldown"
            elif tpm_ceiling_wait >= pacing_wait and tpm_ceiling_wait >= cooldown_wait and tpm_ceiling_wait > 0:
                winner = "TPM ceiling"
            else:
                winner = max(
                    (
                        ("RPM", delays["rpm"]),
                        ("TPM", delays["tpm"]),
                        ("RPD", delays["rpd"]),
                    ),
                    key=lambda x: x[1],
                )[0]
            return wait_for, winner, settings

        return 0.0, "dispatch", settings

    def _record_dispatch_locked(
        self, request_id: str, estimated_tokens: int, provider: str = "", model: str = ""
    ) -> None:
        now_mono = time.monotonic()
        now_wall = time.time()
        self._last_dispatch_mono = now_mono
        self._last_dispatch_wall = now_wall
        burst_key = get_scope_key(provider, model)
        if "::" in burst_key:
            credit, _ = self._burst_credit.get(burst_key, (1.0, now_mono))
            self._burst_credit[burst_key] = (max(0.0, credit - 1.0), now_mono)
        self._last_wait_seconds = 0.0
        self._last_delay_reason = "dispatch"

        self.ensure_current_day_locked()
        day = self._state.setdefault("day", {})
        if day.get("first_request_at") is None:
            day["first_request_at"] = now_wall
        day["requests"] = int(_num(day.get("requests"), 0)) + 1

        scope_key = self.key if self.scope_type != "global" else get_scope_key(provider, model)
        self._controller.record_rpd_dispatch_locked(
            scope_key=scope_key,
            now_wall=now_wall,
            now_mono=now_mono,
            provider=provider,
            model=model,
        )

        res_key = request_id if request_id else "_last"
        self._reservations[res_key] = estimated_tokens

    def _wait_until_dispatch_allowed_locked(
        self,
        request_id: str,
        estimated_tokens: int,
        provider: str,
        model: str,
        before_dispatch: Callable[[], None] | None = None,
    ) -> None:
        while True:
            wait_for, reason, settings = self._check_wait_locked(
                request_id, estimated_tokens, provider, model
            )
            if not settings.enabled:
                self._last_wait_seconds = 0.0
                self._last_delay_reason = "disabled"
                if before_dispatch is not None:
                    before_dispatch()
                return

            if wait_for > 0.001:
                self._last_wait_seconds = wait_for
                self._last_delay_reason = reason
                self._cv.wait(timeout=min(wait_for, 1.0))
                continue

            if before_dispatch is not None:
                before_dispatch()
            self._record_dispatch_locked(request_id, estimated_tokens, provider, model)
            self._controller._persist_locked()
            return

    def on_success(
        self,
        request_id: str,
        total_tokens: int,
        now: float,
        provider: str = "",
        model: str = "",
    ) -> None:
        with self._cv:
            self.release_reservation_locked(request_id)
            if total_tokens > 0:
                self._token_ledger.append((now, total_tokens))
            self.prune_ledger_locked(now)

            day = self._state.setdefault("day", {})
            if total_tokens > 0:
                day["tokens"] = int(_num(day.get("tokens"), 0)) + total_tokens

            adaptive = self._state.setdefault("adaptive", {})
            adaptive["error_streak"] = 0
            successes = int(_num(adaptive.get("successes_since_429"), 0)) + 1
            adaptive["successes_since_429"] = successes
            if successes >= _RECOVERY_EVERY_SUCCESSES:
                if self._controller.learner.allows_upward_exploration(
                    provider, model, pending_success=True
                ):
                    old_factor = _num(adaptive.get("factor"), 1.0)
                    adaptive["factor"] = min(1.0, old_factor + _RECOVERY_STEP)
                adaptive["successes_since_429"] = 0
            self._cv.notify_all()

    def on_error(
        self,
        request_id: str,
        is_429: bool,
        estimated_tokens: int,
        provider: str,
        model: str,
        now: float,
    ) -> None:
        with self._cv:
            self.release_reservation_locked(request_id)
            if is_429:
                adaptive = self._state.setdefault("adaptive", {})
                adaptive["error_streak"] = int(_num(adaptive.get("error_streak"), 0)) + 1
                old_factor = _num(adaptive.get("factor"), 1.0)
                new_factor = max(_MIN_ADAPTIVE_FACTOR, old_factor * _429_MULTIPLIER)
                adaptive["factor"] = new_factor
                adaptive["last_429_at"] = now
                adaptive["successes_since_429"] = 0

                limits = self._controller.resolve_limits(
                    provider=provider,
                    model=model,
                    scope=self.key if self.scope_type != "global" else None,
                )
                base_delay = max(
                    60.0 / max(1, limits.rpm),
                    60.0 * estimated_tokens / max(1, limits.tpm),
                )
                adaptive["cooldown_until"] = max(
                    _num(adaptive.get("cooldown_until"), 0.0),
                    now + max(10.0, base_delay * 2.0),
                )
            self._cv.notify_all()


class GlobalThrottle:
    """
    Global leaky bucket throttle with optional provider and provider::model scoped overrides.

    Resolves limits in hierarchical order: model override -> provider override -> global default.
    Each configured scope maintains its own RPM state, TPM ledger, in-flight reservations, and pacing state.
    When both provider and model limits apply, requests must satisfy both.
    """

    def __init__(self, ctx):
        self.ctx = ctx
        self._cv = threading.Condition(threading.RLock())

        self._pending: dict[str, PendingRequest] = {}
        self._request_buckets: dict[str, list[str]] = {}

        self._last_provider = ""
        self._last_model = ""

        loaded = self._get_state(PLUGIN_STATE_KEY, default=None)
        self._state = self._normalize_state(loaded)
        self._rpd_last_dispatch_mono: dict[str, float] = {}
        self._learner = BayesianRateLimitLearner(self._state.setdefault("learned_limits", {}))
        telemetry = self._state.get("telemetry")
        if not isinstance(telemetry, dict):
            telemetry = {}
            self._state["telemetry"] = telemetry
        resources = telemetry.get("resources")
        if not isinstance(resources, dict):
            resources = {}
            telemetry["resources"] = resources
        self._capacity_control_lock = threading.RLock()
        self._resource_sampler = SystemResourceSampler(resources)
        self._adaptive_capacity = AdaptiveCapacityController(resources)
        initial_snapshot = self._resource_sampler.maybe_sample(force=True)
        initial_capacity = self._adaptive_capacity.observe(
            initial_snapshot or {}, self._capacity_units()
        )
        self._capacity_pool = WeightedCapacityPool(
            capacity=initial_capacity,
            weights=self._workload_weights(),
        )

        # Root global bucket (default when no override exists)
        self._global_bucket = ThrottleBucket(
            key="global",
            scope_type="global",
            state_dict=self._state,
            controller=self,
            cv=self._cv,
        )
        self._scoped_buckets: dict[str, ThrottleBucket] = {}

        # Initialize any existing scoped buckets from state or config
        self._init_configured_buckets_locked()
        self._persist_locked()

    def _init_configured_buckets_locked(self) -> None:
        overrides = self.get_all_overrides()
        for scope in overrides:
            self.get_bucket(scope)
        persisted = self._state.get("scoped_buckets", {})
        if isinstance(persisted, dict):
            for scope in persisted:
                self.get_bucket(scope)

    # ------------------ Backward Compatibility Properties --------------------

    @property
    def learned_limits(self) -> dict[str, Any]:
        with self._cv:
            return copy.deepcopy(self._state.get("learned_limits", {}))

    @property
    def learner(self) -> BayesianRateLimitLearner:
        return self._learner

    @property
    def capacity_pool(self) -> WeightedCapacityPool:
        return self._capacity_pool

    def acquire_workload(self, request_id: str, workload_type: str = "llm_api") -> tuple[str, float]:
        """Wait for and reserve capacity from the one global workload pool."""
        self.refresh_resource_capacity()
        return self._capacity_pool.acquire(request_id, workload_type)

    def release_workload(self, request_id: str) -> bool:
        """Release a previously acquired global workload reservation."""
        return self._capacity_pool.release(request_id)

    def refresh_resource_capacity(self, force_sample: bool = False) -> float:
        """Sample resources when due and apply the persisted learned capacity."""
        with self._capacity_control_lock:
            ceiling = self._capacity_units()
            snapshot = self._resource_sampler.maybe_sample(force=force_sample)
            if snapshot is not None:
                capacity = self._adaptive_capacity.observe(snapshot, ceiling)
                self._persist_locked()
            else:
                capacity = self._adaptive_capacity.current_capacity(ceiling)
            self._capacity_pool.set_capacity(capacity)
            return capacity

    @property
    def resource_telemetry(self) -> dict[str, Any]:
        with self._cv:
            return copy.deepcopy(self._state.get("telemetry", {}).get("resources", {}))

    def record_resource_pressure_event(
        self,
        kind: str,
        severity: float,
        details: dict[str, Any] | None = None,
        timestamp: float | None = None,
    ) -> dict[str, Any]:
        with self._cv:
            event = self._resource_sampler.record_pressure_event(
                kind, severity, details=details, timestamp=timestamp
            )
            self._persist_locked()
            return copy.deepcopy(event)

    def record_resource_recovery_event(
        self,
        kind: str,
        duration_seconds: float,
        peak_severity: float = 0.0,
        timestamp: float | None = None,
    ) -> dict[str, Any]:
        with self._cv:
            event = self._resource_sampler.record_recovery_event(
                kind,
                duration_seconds,
                peak_severity=peak_severity,
                timestamp=timestamp,
            )
            self._persist_locked()
            return copy.deepcopy(event)

    def record_learned_safe_capacity(
        self,
        state_key: str,
        safe_capacity: float,
        confidence: float = 0.0,
        observations: int = 0,
        timestamp: float | None = None,
    ) -> dict[str, Any]:
        with self._cv:
            entry = self._resource_sampler.record_learned_safe_capacity(
                state_key,
                safe_capacity,
                confidence=confidence,
                observations=observations,
                timestamp=timestamp,
            )
            self._persist_locked()
            return copy.deepcopy(entry)

    @property
    def token_ledger(self) -> list[tuple[float, int]]:
        return self._global_bucket.token_ledger

    @property
    def _token_ledger(self) -> list[tuple[float, int]]:
        return self._global_bucket._token_ledger

    @_token_ledger.setter
    def _token_ledger(self, val: list[tuple[float, int]]) -> None:
        self._global_bucket._token_ledger = val

    @property
    def reservations(self) -> dict[str, int]:
        return self._global_bucket.reservations

    @property
    def _reservations(self) -> dict[str, int]:
        return self._global_bucket._reservations

    @_reservations.setter
    def _reservations(self, val: dict[str, int]) -> None:
        self._global_bucket._reservations = val

    @property
    def _last_dispatch_mono(self) -> float:
        return self._global_bucket._last_dispatch_mono

    @_last_dispatch_mono.setter
    def _last_dispatch_mono(self, val: float) -> None:
        self._global_bucket._last_dispatch_mono = val

    @property
    def _last_dispatch_wall(self) -> float:
        return self._global_bucket._last_dispatch_wall

    @_last_dispatch_wall.setter
    def _last_dispatch_wall(self, val: float) -> None:
        self._global_bucket._last_dispatch_wall = val

    @property
    def _last_wait_seconds(self) -> float:
        return self._global_bucket._last_wait_seconds

    @_last_wait_seconds.setter
    def _last_wait_seconds(self, val: float) -> None:
        self._global_bucket._last_wait_seconds = val

    @property
    def _last_delay_reason(self) -> str:
        return self._global_bucket._last_delay_reason

    @_last_delay_reason.setter
    def _last_delay_reason(self, val: str) -> None:
        self._global_bucket._last_delay_reason = val

    @property
    def _next_ticket(self) -> int:
        return self._global_bucket._next_ticket

    @_next_ticket.setter
    def _next_ticket(self, val: int) -> None:
        self._global_bucket._next_ticket = val

    @property
    def _serving_ticket(self) -> int:
        return self._global_bucket._serving_ticket

    @_serving_ticket.setter
    def _serving_ticket(self, val: int) -> None:
        self._global_bucket._serving_ticket = val

    @property
    def adaptive_factor(self) -> float:
        return self._global_bucket.adaptive_factor

    @property
    def cooldown_until(self) -> float:
        return self._global_bucket.cooldown_until

    @property
    def error_streak(self) -> int:
        return self._global_bucket.error_streak

    @property
    def request_count(self) -> int:
        return self._global_bucket.request_count

    @property
    def token_count(self) -> int:
        return self._global_bucket.token_count

    @property
    def ewma_tokens(self) -> float:
        with self._cv:
            return _num(
                self._state.get("token_stats", {}).get("avg_total_tokens"),
                _INITIAL_AVG_TOTAL_TOKENS,
            )

    @property
    def _pending_requests(self) -> dict[str, PendingRequest]:
        return self._pending

    def _prune_ledger_locked(self, now: float) -> None:
        self._global_bucket.prune_ledger_locked(now)

    def _release_reservation_locked(self, request_id: str) -> None:
        self._global_bucket.release_reservation_locked(request_id)
        for b in self._scoped_buckets.values():
            b.release_reservation_locked(request_id)

    def release_reservation(self, request_id: str = "") -> None:
        self._global_bucket.release_reservation(request_id)
        for b in list(self._scoped_buckets.values()):
            b.release_reservation(request_id)

    # ----------------------------- context access ----------------------------

    def _get_config(self, key: str, default: Any = None) -> Any:
        getter = getattr(self.ctx, "get_config", None)
        if callable(getter):
            try:
                val = getter(key, default=default)
                return default if val is None else val
            except Exception:
                return default
        cfg = getattr(self.ctx, "config", None)
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return default

    def _set_config(self, key: str, value: Any) -> None:
        setter = getattr(self.ctx, "set_config", None)
        if callable(setter):
            try:
                setter(key, value)
                return
            except Exception:
                pass
        cfg = getattr(self.ctx, "config", None)
        if isinstance(cfg, dict):
            cfg[key] = value

    def _get_state(self, key: str, default: Any = None) -> Any:
        state_facade = getattr(self.ctx, "state", None)
        if state_facade is not None and hasattr(state_facade, "get"):
            try:
                return state_facade.get(key, default=default)
            except Exception:
                return default
        return default

    def _set_state(self, key: str, value: Any) -> None:
        setter = getattr(self.ctx, "set_state", None)
        if callable(setter):
            try:
                setter(key, value)
                return
            except Exception:
                pass
        state_facade = getattr(self.ctx, "state", None)
        if state_facade is not None and hasattr(state_facade, "set"):
            try:
                state_facade.set(key, value)
                return
            except Exception:
                pass
        if isinstance(state_facade, dict):
            state_facade[key] = value

    def _capacity_units(self) -> float:
        return max(1.0, _num(self._get_config("capacity_units", default=100.0), 100.0))

    def _workload_weights(self) -> dict[str, float]:
        configured = self._get_config("workload_weights", default={})
        if not isinstance(configured, dict):
            return dict(DEFAULT_WORKLOAD_WEIGHTS)
        weights = dict(DEFAULT_WORKLOAD_WEIGHTS)
        for key, value in configured.items():
            parsed = _num(value, 0.0)
            if parsed > 0:
                weights[str(key).strip().lower()] = parsed
        return weights

    # ----------------------------- configuration -----------------------------

    def _settings(self) -> Settings:
        raw_enabled = self._get_config("enabled", default=True)
        if isinstance(raw_enabled, str):
            enabled = raw_enabled.strip().lower() not in {
                "false", "0", "off", "no", "disable", "disabled"
            }
        else:
            enabled = bool(raw_enabled)

        return Settings(
            enabled=enabled,
            rpm=_positive_int(self._get_config("requests_per_minute", default=30), 30),
            tpm=_positive_int(self._get_config("tokens_per_minute", default=60000), 60000),
            rpd=_positive_int(self._get_config("requests_per_day", default=1000), 1000),
            quota_hours=max(
                0.0,
                _num(self._get_config("make_daily_quota_last_hours", default=8.0), 8.0),
            ),
            max_rpd_wait=max(
                0.0,
                _num(self._get_config("longest_wait_between_requests", default=15.0), 15.0),
            ),
        )

    # ------------------------------- state ----------------------------------

    @staticmethod
    def _empty_state() -> dict[str, Any]:
        today = datetime.now().date().isoformat()
        return {
            "version": STATE_VERSION,
            "day": {
                "date": today,
                "requests": 0,
                "tokens": 0,
                "first_request_at": None,
            },
            "token_stats": {
                "avg_total_tokens": _INITIAL_AVG_TOTAL_TOKENS,
                "avg_output_tokens": _INITIAL_AVG_OUTPUT_TOKENS,
                "samples": 0,
                "scopes": {},
            },
            "adaptive": {
                "factor": 1.0,
                "last_429_at": None,
                "cooldown_until": 0.0,
                "successes_since_429": 0,
            },
            "telemetry": {},
            "learned_limits": {},
            "overrides": {},
            "scoped_buckets": {},
            "rpd_scopes": {},
        }

    def _normalize_state(self, value: Any) -> dict[str, Any]:
        base = self._empty_state()
        if not isinstance(value, dict):
            return base

        for section in ("day", "token_stats", "adaptive"):
            incoming = value.get(section)
            if isinstance(incoming, dict):
                base[section].update(incoming)

        telemetry = value.get("telemetry")
        if isinstance(telemetry, dict):
            base["telemetry"] = telemetry

        learned = value.get("learned_limits")
        if isinstance(learned, dict):
            base["learned_limits"] = copy.deepcopy(learned)

        overrides = value.get("overrides")
        if isinstance(overrides, dict):
            base["overrides"] = copy.deepcopy(overrides)

        scoped_buckets = value.get("scoped_buckets")
        if isinstance(scoped_buckets, dict):
            base["scoped_buckets"] = copy.deepcopy(scoped_buckets)

        rpd_scopes = value.get("rpd_scopes")
        if isinstance(rpd_scopes, dict):
            base["rpd_scopes"] = copy.deepcopy(rpd_scopes)
            for entry in base["rpd_scopes"].values():
                if isinstance(entry, dict):
                    entry.pop("last_dispatch_mono", None)

        day = base["day"]
        day["requests"] = max(0, int(_num(day.get("requests"), 0)))
        if day.get("first_request_at") is not None:
            try:
                day["first_request_at"] = float(day["first_request_at"])
            except (TypeError, ValueError):
                day["first_request_at"] = None

        stats = base["token_stats"]
        stats["avg_total_tokens"] = max(
            1.0, _num(stats.get("avg_total_tokens"), _INITIAL_AVG_TOTAL_TOKENS)
        )
        stats["avg_output_tokens"] = max(
            0.0, _num(stats.get("avg_output_tokens"), _INITIAL_AVG_OUTPUT_TOKENS)
        )
        stats["samples"] = max(0, int(_num(stats.get("samples"), 0)))
        scoped_stats = stats.get("scopes")
        if not isinstance(scoped_stats, dict):
            stats["scopes"] = {}
        else:
            for scope, scope_stats in list(scoped_stats.items()):
                if not isinstance(scope_stats, dict):
                    scoped_stats.pop(scope, None)
                    continue
                scope_stats["avg_total_tokens"] = max(
                    1.0,
                    _num(scope_stats.get("avg_total_tokens"), _INITIAL_AVG_TOTAL_TOKENS),
                )
                scope_stats["avg_output_tokens"] = max(
                    0.0,
                    _num(scope_stats.get("avg_output_tokens"), _INITIAL_AVG_OUTPUT_TOKENS),
                )
                scope_stats["samples"] = max(0, int(_num(scope_stats.get("samples"), 0)))

        adaptive = base["adaptive"]
        adaptive["factor"] = min(
            1.0, max(_MIN_ADAPTIVE_FACTOR, _num(adaptive.get("factor"), 1.0))
        )
        adaptive["cooldown_until"] = max(0.0, _num(adaptive.get("cooldown_until"), 0.0))
        adaptive["successes_since_429"] = max(
            0, int(_num(adaptive.get("successes_since_429"), 0))
        )
        if adaptive.get("last_429_at") is not None:
            try:
                adaptive["last_429_at"] = float(adaptive["last_429_at"])
            except (TypeError, ValueError):
                adaptive["last_429_at"] = None

        self._reset_day_if_needed_locked(base)
        for sc_key, sc_val in base["scoped_buckets"].items():
            if isinstance(sc_val, dict):
                self._reset_day_if_needed_locked(sc_val)
        return base

    def _persist_locked(self) -> None:
        self._set_state(PLUGIN_STATE_KEY, copy.deepcopy(self._state))

    @staticmethod
    def _reset_day_if_needed_locked(state: dict[str, Any]) -> None:
        today = datetime.now().date().isoformat()
        day = state.setdefault("day", {})
        if day.get("date") != today:
            day.clear()
            day.update({
                "date": today,
                "requests": 0,
                "tokens": 0,
                "first_request_at": None,
            })

    def _ensure_current_day_locked(self) -> None:
        before = self._state["day"].get("date")
        self._reset_day_if_needed_locked(self._state)
        if self._state["day"].get("date") != before:
            self._persist_locked()

    def _seconds_until_local_midnight(self) -> float:
        now = datetime.now()
        tomorrow = (now + timedelta(days=1)).date()
        midnight = datetime.combine(tomorrow, datetime.min.time())
        return max(0.0, (midnight - now).total_seconds())

    # ----------------------- Scoped RPD & Rolling Window ---------------------

    def _get_rpd_scope_entry_locked(
        self, scope_key: str, provider: str = "", model: str = ""
    ) -> dict[str, Any]:
        scopes = self._state.setdefault("rpd_scopes", {})
        if scope_key not in scopes or not isinstance(scopes[scope_key], dict):
            scopes[scope_key] = {
                "provider": provider,
                "model": model,
                "timestamps": [],
                "first_request_at": None,
                "last_dispatch_wall": 0.0,
            }
        entry = scopes[scope_key]
        # Sanitize legacy or externally restored state before it is persisted.
        entry.pop("last_dispatch_mono", None)
        if "timestamps" not in entry or not isinstance(entry["timestamps"], list):
            entry["timestamps"] = []
        if provider and not entry.get("provider"):
            entry["provider"] = provider
        if model and not entry.get("model"):
            entry["model"] = model
        return entry

    def _prune_rpd_scope_locked(
        self, entry: dict[str, Any], now: float, window_seconds: float = RPD_WINDOW_SECONDS
    ) -> list[float]:
        cutoff = now - window_seconds
        timestamps = entry.get("timestamps", [])
        valid = []
        for ts in timestamps:
            try:
                t = float(ts)
                if t > cutoff:
                    valid.append(t)
            except (TypeError, ValueError):
                pass
        valid.sort()
        entry["timestamps"] = valid
        if not valid:
            entry["first_request_at"] = None
        elif entry.get("first_request_at") is None or entry.get("first_request_at") <= cutoff:
            entry["first_request_at"] = valid[0]
        return valid

    def get_rpd_usage(self, scope_key: str, now: float | None = None) -> tuple[int, float | None]:
        """Returns (rolling_request_count, time_until_oldest_rolls_out)."""
        if now is None:
            now = time.time()
        with self._cv:
            scopes = self._state.get("rpd_scopes", {})
            entry = scopes.get(scope_key)
            if not entry or not isinstance(entry, dict):
                return 0, None
            valid = self._prune_rpd_scope_locked(entry, now)
            count = len(valid)
            if not valid:
                return 0, None
            oldest = valid[0]
            time_until_rolloff = max(0.0, (oldest + RPD_WINDOW_SECONDS) - now)
            return count, time_until_rolloff

    def record_rpd_dispatch_locked(
        self,
        scope_key: str,
        now_wall: float,
        now_mono: float,
        provider: str = "",
        model: str = "",
    ) -> None:
        entry = self._get_rpd_scope_entry_locked(scope_key, provider, model)
        self._prune_rpd_scope_locked(entry, now_wall)
        entry["timestamps"].append(now_wall)
        entry["last_dispatch_wall"] = now_wall
        self._rpd_last_dispatch_mono[scope_key] = now_mono
        if entry.get("first_request_at") is None:
            entry["first_request_at"] = now_wall

    def compute_rpd_pacing_locked(
        self,
        scope_key: str,
        settings: Settings,
        now: float,
        provider: str = "",
        model: str = "",
    ) -> tuple[float, float, int, int]:
        """
        Computes (rpd_delay, raw_rpd_delay, rolling_count, rpd_limit).
        Treats RPD as a pacing signal across the rolling 24h window.
        If approaching or exceeding pacing target, increases spacing rather than stopping requests.
        """
        entry = self._get_rpd_scope_entry_locked(scope_key, provider, model)
        valid = self._prune_rpd_scope_locked(entry, now)
        rolling_count = len(valid)
        limit = settings.rpd
        max_wait = settings.max_rpd_wait

        first_req = entry.get("first_request_at")
        horizon_sec = settings.quota_hours * 3600.0
        if first_req is not None:
            rem_horizon = max(0.0, (float(first_req) + horizon_sec) - now)
        else:
            rem_horizon = horizon_sec

        if valid:
            oldest = valid[0]
            time_until_rolloff = max(0.0, (oldest + RPD_WINDOW_SECONDS) - now)
        else:
            time_until_rolloff = 0.0

        rem_requests = max(0, limit - rolling_count)
        rem_sec = rem_horizon if rem_horizon > 0 else time_until_rolloff

        if rem_requests > 0 and rem_sec > 0:
            raw_rpd_delay = rem_sec / rem_requests
            rpd_delay = min(raw_rpd_delay, max_wait)
        elif rolling_count >= limit:
            # Pacing target reached or exceeded: increase spacing rather than stopping requests
            ratio = rolling_count / max(1, limit)
            raw_rpd_delay = max(max_wait, time_until_rolloff / max(1, limit)) if time_until_rolloff > 0 else max_wait
            rpd_delay = min(30.0, max(max_wait, max_wait * min(2.0, ratio)))
        else:
            raw_rpd_delay = 0.0
            rpd_delay = 0.0

        return rpd_delay, raw_rpd_delay, rolling_count, limit

    # ---------------------- Overrides & Bucket Resolution -------------------

    def get_override(self, scope: str) -> dict[str, Any] | None:
        scope = scope.strip().lower()
        cfg_overrides = self._get_config("overrides")
        if isinstance(cfg_overrides, dict):
            for k, v in cfg_overrides.items():
                if str(k).strip().lower() == scope and isinstance(v, dict):
                    return dict(v)
        state_overrides = self._state.get("overrides", {})
        if isinstance(state_overrides, dict):
            for k, v in state_overrides.items():
                if str(k).strip().lower() == scope and isinstance(v, dict):
                    return dict(v)
        return None

    def has_override(self, scope: str) -> bool:
        return self.get_override(scope) is not None

    def get_all_overrides(self) -> dict[str, dict[str, Any]]:
        combined: dict[str, dict[str, Any]] = {}
        cfg_overrides = self._get_config("overrides")
        if isinstance(cfg_overrides, dict):
            for k, v in cfg_overrides.items():
                if isinstance(v, dict):
                    combined[str(k).strip().lower()] = dict(v)
        state_overrides = self._state.get("overrides", {})
        if isinstance(state_overrides, dict):
            for k, v in state_overrides.items():
                if isinstance(v, dict):
                    key = str(k).strip().lower()
                    if key not in combined:
                        combined[key] = dict(v)
                    else:
                        combined[key].update(v)
        return combined

    def set_override(self, scope: str, **limits) -> None:
        scope = scope.strip().lower()
        with self._cv:
            state_overrides = self._state.setdefault("overrides", {})
            current = state_overrides.setdefault(scope, {})
            for k, v in limits.items():
                if v is not None:
                    current[k] = v
            cfg_overrides = self._get_config("overrides")
            if not isinstance(cfg_overrides, dict):
                cfg_overrides = {}
            cfg_overrides[scope] = copy.deepcopy(current)
            self._set_config("overrides", cfg_overrides)
            self._persist_locked()
            self.get_bucket(scope)
            self._cv.notify_all()

    def remove_override(self, scope: str, param: str | None = None) -> bool:
        scope = scope.strip().lower()
        with self._cv:
            found = False
            state_overrides = self._state.get("overrides", {})
            cfg_overrides = self._get_config("overrides")
            if not isinstance(cfg_overrides, dict):
                cfg_overrides = {}

            if param:
                param = param.strip().lower()
                if scope in state_overrides and param in state_overrides[scope]:
                    del state_overrides[scope][param]
                    found = True
                    if not state_overrides[scope]:
                        del state_overrides[scope]
                if scope in cfg_overrides and param in cfg_overrides[scope]:
                    del cfg_overrides[scope][param]
                    found = True
                    if not cfg_overrides[scope]:
                        del cfg_overrides[scope]
            else:
                if scope in state_overrides:
                    del state_overrides[scope]
                    found = True
                if scope in cfg_overrides:
                    del cfg_overrides[scope]
                    found = True

            if found:
                self._set_config("overrides", cfg_overrides)
                self._persist_locked()
                self._cv.notify_all()
            return found

    def resolve_limits(
        self,
        provider: str = "",
        model: str = "",
        scope: str | None = None,
    ) -> Settings:
        """
        Resolves limits in hierarchical order:
        model override -> provider override -> global default.
        """
        global_settings = self._settings()
        if scope:
            scope = scope.strip().lower()
            if "::" in scope:
                prov, mod = scope.split("::", 1)
            else:
                prov, mod = scope, ""
        else:
            prov = provider.strip().lower() if provider and provider != "unknown" else ""
            mod = model.strip() if model and model != "unknown" else ""

        mod_key = f"{prov}::{mod}" if prov and mod else ""
        mod_override = self.get_override(mod_key) if mod_key else None
        prov_override = self.get_override(prov) if prov else None

        def pick_val(key: str, default: Any, aliases: tuple[str, ...] = ()) -> Any:
            for al in (key, *aliases):
                if mod_override and al in mod_override and mod_override[al] is not None:
                    return mod_override[al]
            for al in (key, *aliases):
                if prov_override and al in prov_override and prov_override[al] is not None:
                    return prov_override[al]
            return default

        rpm = _positive_int(
            pick_val("rpm", global_settings.rpm, ("requests_per_minute",)),
            global_settings.rpm,
        )
        tpm = _positive_int(
            pick_val("tpm", global_settings.tpm, ("tokens_per_minute",)),
            global_settings.tpm,
        )
        default_rpd = DEFAULT_GEMINI_RPD if is_gemini_scope(prov, mod, scope or "") else global_settings.rpd
        rpd = _positive_int(
            pick_val("rpd", default_rpd, ("requests_per_day",)),
            default_rpd,
        )
        quota_hours = max(
            0.0,
            _num(
                pick_val(
                    "quota_hours",
                    global_settings.quota_hours,
                    ("make_daily_quota_last_hours", "hours"),
                ),
                global_settings.quota_hours,
            ),
        )
        max_rpd_wait = max(
            0.0,
            _num(
                pick_val(
                    "max_rpd_wait",
                    global_settings.max_rpd_wait,
                    ("longest_wait_between_requests", "max-wait", "max_wait"),
                ),
                global_settings.max_rpd_wait,
            ),
        )

        return Settings(
            enabled=global_settings.enabled,
            rpm=rpm,
            tpm=tpm,
            rpd=rpd,
            quota_hours=quota_hours,
            max_rpd_wait=max_rpd_wait,
        )

    def get_bucket(self, key: str, scope_type: str | None = None) -> ThrottleBucket:
        key = key.strip().lower()
        if key == "global":
            return self._global_bucket
        with self._cv:
            if key not in self._scoped_buckets:
                if scope_type is None:
                    scope_type = "model" if "::" in key else "provider"
                scoped_state = self._state.setdefault("scoped_buckets", {}).setdefault(key, {})
                bucket = ThrottleBucket(
                    key=key,
                    scope_type=scope_type,
                    state_dict=scoped_state,
                    controller=self,
                )
                self._scoped_buckets[key] = bucket
            return self._scoped_buckets[key]

    def get_applicable_buckets(self, provider: str, model: str) -> list[ThrottleBucket]:
        prov = provider.strip().lower() if provider and provider != "unknown" else ""
        mod = model.strip() if model and model != "unknown" else ""

        has_prov = bool(prov and self.has_override(prov))
        mod_key = f"{prov}::{mod}" if prov and mod else ""
        has_mod = bool(mod_key and self.has_override(mod_key))

        if has_prov and has_mod:
            return [
                self.get_bucket(prov, scope_type="provider"),
                self.get_bucket(mod_key, scope_type="model"),
            ]
        elif has_prov:
            return [self.get_bucket(prov, scope_type="provider")]
        elif has_mod:
            return [self.get_bucket(mod_key, scope_type="model")]
        elif is_gemini_scope(provider, model):
            gemini_key = mod_key if mod_key else (prov if prov else "google")
            scope_type = "model" if "::" in gemini_key else "provider"
            return [self.get_bucket(gemini_key, scope_type=scope_type)]
        else:
            return [self._global_bucket]

    def _get_active_scoped_buckets_locked(self) -> list[ThrottleBucket]:
        active_keys: set[str] = set()
        overrides = self.get_all_overrides()
        for k in overrides:
            active_keys.add(k)
        for k in self._state.get("rpd_scopes", {}):
            if is_gemini_scope(scope=k):
                active_keys.add(k)

        for k, b in self._scoped_buckets.items():
            b_day = b._state.get("day", {})
            b_reqs = int(_num(b_day.get("requests"), 0))
            b_toks = int(_num(b_day.get("tokens"), 0))
            b_adaptive = b._state.get("adaptive", {})
            b_factor = _num(b_adaptive.get("factor"), 1.0)
            b_cooldown = _num(b_adaptive.get("cooldown_until"), 0.0)
            if (
                k in overrides
                or b_reqs > 0
                or b_toks > 0
                or b._token_ledger
                or b._reservations
                or b._next_ticket > b._serving_ticket
                or b_factor < 1.0
                or b_cooldown > time.time()
            ):
                active_keys.add(k)

        persisted = self._state.get("scoped_buckets", {})
        if isinstance(persisted, dict):
            for k, p_state in persisted.items():
                if isinstance(p_state, dict):
                    p_day = p_state.get("day", {})
                    if int(_num(p_day.get("requests"), 0)) > 0 or int(_num(p_day.get("tokens"), 0)) > 0:
                        active_keys.add(k)

        result: list[ThrottleBucket] = []
        for k in sorted(active_keys):
            result.append(self.get_bucket(k))
        return result

    # ------------------------- request token estimate ------------------------

    def on_pre_api_request(self, **kwargs):
        request_id = str(kwargs.get("api_request_id") or kwargs.get("request_id") or "")
        provider = str(kwargs.get("provider") or "unknown")
        model = str(kwargs.get("model") or "unknown")
        approx_input = max(0, _positive_int(kwargs.get("approx_input_tokens"), 0))

        if approx_input == 0 and "request" in kwargs:
            req_obj = kwargs.get("request")
            if isinstance(req_obj, dict):
                msgs = req_obj.get("messages")
                if isinstance(msgs, list):
                    total_chars = sum(
                        len(str(m.get("content", "")))
                        for m in msgs
                        if isinstance(m, dict)
                    )
                    approx_input = max(0, total_chars // 4)
                elif isinstance(req_obj.get("prompt"), str):
                    approx_input = max(0, len(req_obj["prompt"]) // 4)

        with self._cv:
            if len(self._pending) > 500:
                cutoff = time.time() - 300.0
                expired = [
                    k for k, v in self._pending.items()
                    if getattr(v, "created_at", 0.0) < cutoff
                ]
                for k in expired:
                    self._pending.pop(k, None)

            stats = self._state.get("token_stats", {})
            scope_key = get_scope_key(provider, model)
            scope_stats = None
            if "::" in scope_key:
                scoped = stats.get("scopes", {})
                candidate = scoped.get(scope_key) if isinstance(scoped, dict) else None
                if isinstance(candidate, dict) and int(_num(candidate.get("samples"), 0)) > 0:
                    scope_stats = candidate
            if scope_stats is not None:
                stats = scope_stats
            avg_out = _num(stats.get("avg_output_tokens"), _INITIAL_AVG_OUTPUT_TOKENS)
            avg_total = _num(stats.get("avg_total_tokens"), _INITIAL_AVG_TOTAL_TOKENS)

            if approx_input > 0:
                estimate = int(max(1.0, approx_input + avg_out))
            else:
                estimate = int(max(1.0, avg_total))

            pending_item = PendingRequest(
                provider=provider,
                model=model,
                approx_input_tokens=approx_input,
                estimated_total_tokens=estimate,
            )
            key = request_id if request_id else (
                f"req_{int(time.time()*1000)}" if not request_id and "request" in kwargs else "_last"
            )
            self._pending[key] = pending_item

            return {
                "request_id": key,
                "estimated_tokens": estimate,
                "approx_input_tokens": approx_input,
            }

    def _estimated_tokens_locked(self, request_id: str) -> int:
        pending = self._pending.get(request_id) or self._pending.get("_last")
        if pending is None and len(self._pending) == 1:
            pending = next(iter(self._pending.values()))
        if pending is not None:
            return max(1, pending.estimated_total_tokens)
        stats = self._state.get("token_stats", {})
        avg = _num(stats.get("avg_total_tokens"), _INITIAL_AVG_TOTAL_TOKENS)
        return max(1, int(avg))

    # ------------------------------ scheduler -------------------------------

    def wrap_llm_execution(self, *args, **kwargs):
        request = None
        next_call = None
        request_id = ""

        if len(args) == 1 and isinstance(args[0], dict):
            ctx_dict = args[0]
            request = ctx_dict.get("request")
            next_call = ctx_dict.get("next_call")
            request_id = str(ctx_dict.get("api_request_id") or ctx_dict.get("request_id") or "")
        elif len(args) >= 2:
            request = args[0]
            next_call = args[1]

        if request is None and "request" in kwargs:
            request = kwargs.get("request")
        if next_call is None and "next_call" in kwargs:
            next_call = kwargs.get("next_call")
        if not request_id:
            request_id = str(kwargs.get("api_request_id") or kwargs.get("request_id") or "")

        provider = str(kwargs.get("provider") or "")
        model = str(kwargs.get("model") or "")
        base_url = str(kwargs.get("base_url") or "")
        if not provider or not model:
            if len(args) == 1 and isinstance(args[0], dict):
                provider = provider or str(args[0].get("provider") or "")
                model = model or str(args[0].get("model") or "")
                base_url = base_url or str(args[0].get("base_url") or "")
        if not provider or not model:
            pending = self._pending.get(request_id) or self._pending.get("_last")
            if pending is None and len(self._pending) == 1:
                pending = next(iter(self._pending.values()))
            if pending:
                provider = provider or pending.provider
                model = model or pending.model
        if not model and isinstance(request, dict):
            model = str(request.get("model") or "")

        provider = provider.strip()
        model = model.strip()
        workload_type = str(kwargs.get("workload_type") or "").strip().lower()
        if not workload_type and len(args) == 1 and isinstance(args[0], dict):
            workload_type = str(args[0].get("workload_type") or "").strip().lower()
        if not workload_type:
            workload_type = "llm_api"

        with self._cv:
            if not request_id:
                if "_last" in self._pending:
                    request_id = "_last"
                elif len(self._pending) == 1:
                    request_id = next(iter(self._pending.keys()))
                else:
                    request_id = f"req_{int(time.time() * 1000)}"

            estimated_tokens = self._estimated_tokens_locked(request_id)
            buckets = self.get_applicable_buckets(provider, model)
            self._request_buckets[request_id] = [b.key for b in buckets]
            if request_id != "_last":
                self._request_buckets["_last"] = [b.key for b in buckets]

            self._last_provider = provider
            self._last_model = model

        capacity_request_id = f"llm:{request_id}"
        use_capacity_pool = not is_local_model_execution(provider, base_url, model)
        capacity_acquired = False

        def acquire_capacity_at_dispatch() -> None:
            nonlocal capacity_acquired
            if use_capacity_pool and not capacity_acquired:
                self.acquire_workload(capacity_request_id, workload_type)
                capacity_acquired = True

        try:
            if len(buckets) == 1:
                self._dispatch_single_bucket(
                    buckets[0], request_id, estimated_tokens, provider, model,
                    kwargs, before_dispatch=acquire_capacity_at_dispatch,
                )
            else:
                b_prov = next(b for b in buckets if b.scope_type == "provider")
                b_mod = next(b for b in buckets if b.scope_type == "model")
                self._dispatch_two_buckets(
                    b_mod, b_prov, request_id, estimated_tokens, provider, model,
                    kwargs, before_dispatch=acquire_capacity_at_dispatch,
                )
        except BaseException:
            if capacity_acquired:
                self.release_workload(capacity_request_id)
            raise

        if callable(next_call):
            try:
                return next_call(request)
            except BaseException:
                self.release_reservation(request_id)
                raise
            finally:
                if capacity_acquired:
                    self.release_workload(capacity_request_id)
        if capacity_acquired:
            self.release_workload(capacity_request_id)
        return request

    def wrap_tool_execution(self, *args, **kwargs):
        """Gate every non-cron tool at the shared pool immediately before execution."""
        tool_name = str(kwargs.get("tool_name") or "")
        tool_args = kwargs.get("args")
        next_call = kwargs.get("next_call")
        tool_call_id = str(kwargs.get("tool_call_id") or "")

        if len(args) == 1 and isinstance(args[0], dict):
            context = args[0]
            tool_name = tool_name or str(context.get("tool_name") or "")
            tool_args = tool_args if tool_args is not None else context.get("args")
            next_call = next_call or context.get("next_call")
            tool_call_id = tool_call_id or str(context.get("tool_call_id") or "")
        elif len(args) >= 3:
            tool_name = tool_name or str(args[0] or "")
            tool_args = tool_args if tool_args is not None else args[1]
            next_call = next_call or args[2]

        workload_type = classify_tool_workload(tool_name)
        if not callable(next_call):
            return tool_args
        if workload_type is None:
            return next_call(tool_args)

        reservation_id = f"tool:{tool_call_id or uuid.uuid4().hex}"
        self.acquire_workload(reservation_id, workload_type)

        if workload_type == "subagent":
            # This is a start gate. Releasing before delegate_task runs avoids
            # deadlocking the child model/tool work that uses the same pool.
            self.release_workload(reservation_id)
            return next_call(tool_args)

        try:
            return next_call(tool_args)
        finally:
            self.release_workload(reservation_id)

    def _dispatch_single_bucket(
        self,
        bucket: ThrottleBucket,
        request_id: str,
        estimated_tokens: int,
        provider: str,
        model: str,
        context: dict[str, Any],
        before_dispatch: Callable[[], None] | None = None,
    ) -> None:
        ticket = bucket.claim_ticket()
        try:
            with bucket._cv:
                while ticket != bucket._serving_ticket:
                    bucket._cv.wait(timeout=1.0)
                bucket._wait_until_dispatch_allowed_locked(
                    request_id,
                    estimated_tokens,
                    provider,
                    model,
                    before_dispatch=before_dispatch,
                )
        except BaseException:
            bucket.release_reservation(request_id)
            raise
        finally:
            bucket.advance_ticket()

    def _dispatch_two_buckets(
        self,
        b_mod: ThrottleBucket,
        b_prov: ThrottleBucket,
        request_id: str,
        estimated_tokens: int,
        provider: str,
        model: str,
        context: dict[str, Any],
        before_dispatch: Callable[[], None] | None = None,
    ) -> None:
        ticket_mod = b_mod.claim_ticket()
        try:
            with b_mod._cv:
                while ticket_mod != b_mod._serving_ticket:
                    b_mod._cv.wait(timeout=1.0)

                while True:
                    wait_for, reason, settings = b_mod._check_wait_locked(
                        request_id, estimated_tokens, provider, model
                    )
                    if not settings.enabled:
                        b_mod._last_wait_seconds = 0.0
                        b_mod._last_delay_reason = "disabled"
                        break
                    if wait_for > 0.001:
                        b_mod._last_wait_seconds = wait_for
                        b_mod._last_delay_reason = reason
                        b_mod._cv.wait(timeout=min(wait_for, 1.0))
                        continue
                    break

            ticket_prov = b_prov.claim_ticket()
            try:
                with b_prov._cv:
                    while ticket_prov != b_prov._serving_ticket:
                        b_prov._cv.wait(timeout=1.0)

                    while True:
                        wait_for, reason, settings = b_prov._check_wait_locked(
                            request_id, estimated_tokens, provider, model
                        )
                        if not settings.enabled:
                            b_prov._last_wait_seconds = 0.0
                            b_prov._last_delay_reason = "disabled"
                            break
                        if wait_for > 0.001:
                            b_prov._last_wait_seconds = wait_for
                            b_prov._last_delay_reason = reason
                            b_prov._cv.wait(timeout=min(wait_for, 1.0))
                            continue
                        break

                    if before_dispatch is not None:
                        before_dispatch()
                    with b_mod._cv:
                        b_mod._record_dispatch_locked(request_id, estimated_tokens, provider, model)
                    b_prov._record_dispatch_locked(request_id, estimated_tokens, provider, model)
                    self._persist_locked()
            except BaseException:
                b_prov.release_reservation(request_id)
                raise
            finally:
                b_prov.advance_ticket()
        except BaseException:
            b_mod.release_reservation(request_id)
            raise
        finally:
            b_mod.advance_ticket()

    def _wait_until_dispatch_allowed_locked(
        self, request_id: str, context: dict[str, Any]
    ) -> None:
        """Compatibility fallback for tests or internal calls."""
        provider = str(context.get("provider") or "")
        model = str(context.get("model") or "")
        estimated_tokens = self._estimated_tokens_locked(request_id)
        self._global_bucket._wait_until_dispatch_allowed_locked(
            request_id, estimated_tokens, provider, model
        )

    # ------------------------------ telemetry -------------------------------

    def _telemetry_entry_locked(self, provider: str, model: str) -> dict[str, Any]:
        key = f"{provider}:{model}"
        telemetry = self._state.setdefault("telemetry", {})
        entry = telemetry.setdefault(
            key,
            {
                "provider": provider,
                "model": model,
                "requests_seen": 0,
                "successes": 0,
                "errors": 0,
                "rate_limit_429s": 0,
                "avg_total_tokens": 0.0,
                "token_samples": 0,
                "last_success_at": None,
                "last_error_at": None,
                "last_429_at": None,
            },
        )
        return entry

    def on_post_api_request(self, **kwargs):
        request_id = str(kwargs.get("api_request_id") or kwargs.get("request_id") or "")
        provider = str(kwargs.get("provider") or "unknown")
        model = str(kwargs.get("model") or "unknown")
        usage = kwargs.get("usage")
        if usage is None:
            resp = kwargs.get("response")
            if isinstance(resp, dict):
                usage = resp.get("usage")
            elif hasattr(resp, "usage"):
                usage = getattr(resp, "usage")

        total_tokens, output_tokens = _token_total(usage)
        now = time.time()

        pending = self._pending.get(request_id) or self._pending.get("_last")
        if (not provider or provider == "unknown") and pending and pending.provider != "unknown":
            provider = pending.provider
        if (not model or model == "unknown") and pending and pending.model != "unknown":
            model = pending.model

        bucket_keys = self._request_buckets.pop(request_id, None) or self._request_buckets.pop("_last", None)
        if bucket_keys:
            buckets = [self.get_bucket(k) for k in bucket_keys]
        else:
            buckets = self.get_applicable_buckets(provider, model)

        for bucket in buckets:
            bucket.on_success(request_id, total_tokens, now, provider, model)

        with self._cv:
            settings = self._settings()
            primary_bucket = buckets[0]
            rolling_reqs = len(primary_bucket.token_ledger)
            rolling_toks = sum(tok for _, tok in primary_bucket.token_ledger)

            self._learner.on_success(
                provider=provider,
                model=model,
                actual_tokens=total_tokens,
                rolling_requests=rolling_reqs,
                rolling_tokens=rolling_toks,
                now=now,
                default_rpm=settings.rpm,
                default_tpm=settings.tpm,
            )

            if self._global_bucket not in buckets and total_tokens > 0:
                day = self._state.setdefault("day", {})
                day["tokens"] = int(_num(day.get("tokens"), 0)) + total_tokens

            stats = self._state.setdefault("token_stats", {})
            if total_tokens > 0:
                samples = int(_num(stats.get("samples"), 0))
                if samples <= 0:
                    stats["avg_total_tokens"] = float(total_tokens)
                    stats["avg_output_tokens"] = float(output_tokens)
                else:
                    curr_avg_total = _num(stats.get("avg_total_tokens"), _INITIAL_AVG_TOTAL_TOKENS)
                    curr_avg_output = _num(stats.get("avg_output_tokens"), _INITIAL_AVG_OUTPUT_TOKENS)
                    stats["avg_total_tokens"] = (
                        (1.0 - _EWMA_ALPHA) * curr_avg_total
                        + _EWMA_ALPHA * float(total_tokens)
                    )
                    stats["avg_output_tokens"] = (
                        (1.0 - _EWMA_ALPHA) * curr_avg_output
                        + _EWMA_ALPHA * float(output_tokens)
                    )
                stats["samples"] = samples + 1

                scope_key = get_scope_key(provider, model)
                if "::" in scope_key:
                    scoped = stats.setdefault("scopes", {})
                    scope_stats = scoped.setdefault(
                        scope_key,
                        {
                            "avg_total_tokens": _INITIAL_AVG_TOTAL_TOKENS,
                            "avg_output_tokens": _INITIAL_AVG_OUTPUT_TOKENS,
                            "samples": 0,
                        },
                    )
                    scope_samples = int(_num(scope_stats.get("samples"), 0))
                    if scope_samples <= 0:
                        scope_stats["avg_total_tokens"] = float(total_tokens)
                        scope_stats["avg_output_tokens"] = float(output_tokens)
                    else:
                        current_scope_total = _num(
                            scope_stats.get("avg_total_tokens"), _INITIAL_AVG_TOTAL_TOKENS
                        )
                        current_scope_output = _num(
                            scope_stats.get("avg_output_tokens"), _INITIAL_AVG_OUTPUT_TOKENS
                        )
                        scope_stats["avg_total_tokens"] = (
                            (1.0 - _EWMA_ALPHA) * current_scope_total
                            + _EWMA_ALPHA * float(total_tokens)
                        )
                        scope_stats["avg_output_tokens"] = (
                            (1.0 - _EWMA_ALPHA) * current_scope_output
                            + _EWMA_ALPHA * float(output_tokens)
                        )
                    scope_stats["samples"] = scope_samples + 1

            entry = self._telemetry_entry_locked(provider, model)
            entry["requests_seen"] = int(_num(entry.get("requests_seen"), 0)) + 1
            entry["successes"] = int(_num(entry.get("successes"), 0)) + 1
            entry["last_success_at"] = now

            if total_tokens > 0:
                n = int(_num(entry.get("token_samples"), 0))
                old = _num(entry.get("avg_total_tokens"), 0.0)
                entry["avg_total_tokens"] = (
                    float(total_tokens) if n == 0
                    else old + (float(total_tokens) - old) / (n + 1)
                )
                entry["token_samples"] = n + 1

            if request_id:
                self._pending.pop(request_id, None)
            self._pending.pop("_last", None)
            self._persist_locked()
            self._cv.notify_all()

    def on_api_request_error(self, **kwargs):
        request_id = str(kwargs.get("api_request_id") or kwargs.get("request_id") or "")
        provider = str(kwargs.get("provider") or "unknown")
        model = str(kwargs.get("model") or "unknown")
        status_code = kwargs.get("status_code")
        now = time.time()

        is_429 = False
        try:
            if status_code is not None and int(status_code) == 429:
                is_429 = True
        except (TypeError, ValueError):
            pass

        if not is_429:
            reason = str(kwargs.get("reason") or "").lower()
            error_str = str(kwargs.get("error") or "").lower()
            if "rate_limit" in reason or "rate limit" in error_str or "429" in error_str:
                is_429 = True

        pending = self._pending.get(request_id) or self._pending.get("_last")
        if (not provider or provider == "unknown") and pending and pending.provider != "unknown":
            provider = pending.provider
        if (not model or model == "unknown") and pending and pending.model != "unknown":
            model = pending.model

        bucket_keys = self._request_buckets.pop(request_id, None) or self._request_buckets.pop("_last", None)
        if bucket_keys:
            buckets = [self.get_bucket(k) for k in bucket_keys]
        else:
            buckets = self.get_applicable_buckets(provider, model)

        est_tokens = self._estimated_tokens_locked(request_id)
        for bucket in buckets:
            bucket.on_error(request_id, is_429, est_tokens, provider, model, now)

        with self._cv:
            entry = self._telemetry_entry_locked(provider, model)
            entry["requests_seen"] = int(_num(entry.get("requests_seen"), 0)) + 1
            entry["errors"] = int(_num(entry.get("errors"), 0)) + 1
            entry["last_error_at"] = now

            if is_429:
                entry["rate_limit_429s"] = int(_num(entry.get("rate_limit_429s"), 0)) + 1
                entry["last_429_at"] = now

                settings = self._settings()
                headers = _extract_headers(kwargs)
                error_str = str(kwargs.get("error") or "")
                reason_str = str(kwargs.get("reason") or "")
                target_dim = _detect_429_target(headers, error_str, reason_str)

                primary_bucket = buckets[0]
                rolling_reqs = len(primary_bucket.token_ledger) + 1
                rolling_toks = sum(tok for _, tok in primary_bucket.token_ledger)

                self._learner.on_429(
                    provider=provider,
                    model=model,
                    estimated_tokens=est_tokens,
                    rolling_requests=rolling_reqs,
                    rolling_tokens=rolling_toks,
                    now=now,
                    default_rpm=settings.rpm,
                    default_tpm=settings.tpm,
                    target_dimension=target_dim,
                )

            if request_id:
                self._pending.pop(request_id, None)
            self._pending.pop("_last", None)
            self._persist_locked()
            self._cv.notify_all()

    # ---------------------------- slash command -----------------------------

    def handle_command(self, raw_args: Any = "") -> str:
        if isinstance(raw_args, (list, tuple)):
            raw = " ".join(str(x) for x in raw_args).strip()
        else:
            raw = str(raw_args or "").strip()

        if not raw or raw.lower() == "status":
            return self._status_text()
        if raw.lower() == "status verbose":
            return self._status_text(verbose=True)

        parts = raw.split()
        cmd = parts[0].lower()

        if cmd in {"on", "enable", "enabled"}:
            self._set_config("enabled", True)
            with self._cv:
                self._cv.notify_all()
            return "Throttle enabled."

        if cmd in {"off", "disable", "disabled"}:
            self._set_config("enabled", False)
            with self._cv:
                self._cv.notify_all()
            return "Throttle disabled."

        if cmd == "reset":
            with self._cv:
                adaptive = self._state.setdefault("adaptive", {})
                adaptive["factor"] = 1.0
                adaptive["last_429_at"] = None
                adaptive["cooldown_until"] = 0.0
                adaptive["successes_since_429"] = 0
                adaptive["error_streak"] = 0
                for b in self._scoped_buckets.values():
                    b_adaptive = b._state.setdefault("adaptive", {})
                    b_adaptive["factor"] = 1.0
                    b_adaptive["last_429_at"] = None
                    b_adaptive["cooldown_until"] = 0.0
                    b_adaptive["successes_since_429"] = 0
                    b_adaptive["error_streak"] = 0
                self._persist_locked()
                self._cv.notify_all()
            return "Adaptive rate factor and cooldown have been reset."

        setters = {
            "rpm": ("requests_per_minute", int, "RPM"),
            "tpm": ("tokens_per_minute", int, "TPM"),
            "rpd": ("requests_per_day", int, "RPD"),
            "hours": ("make_daily_quota_last_hours", float, "quota-hours"),
            "max-wait": ("longest_wait_between_requests", float, "max-wait"),
            "max_wait": ("longest_wait_between_requests", float, "max-wait"),
        }

        if cmd in setters and len(parts) == 2:
            key, caster, label = setters[cmd]
            try:
                value = caster(parts[1])
            except (TypeError, ValueError):
                return f"Invalid {label} value: {parts[1]}"

            if value <= 0 and cmd not in {"hours", "max-wait", "max_wait"}:
                return f"{label} must be greater than zero."
            if value < 0:
                return f"{label} cannot be negative."

            self._set_config(key, value)
            with self._cv:
                self._cv.notify_all()
            return f"{label} set to {value}."

        if cmd in {"reset-learning", "reset_learning"}:
            if len(parts) < 3:
                if len(parts) == 2 and "::" in parts[1]:
                    prov, mod = parts[1].split("::", 1)
                else:
                    return "Usage: /throttle reset-learning <provider> <model>"
            else:
                prov = parts[1]
                mod = parts[2]

            with self._cv:
                reset = self._learner.reset_learning(prov, mod)
                self._persist_locked()
                self._cv.notify_all()
            if reset:
                return f"Learned limits reset for {prov}::{mod}."
            return f"No learned limits found for {prov}::{mod}."

        if cmd in {"recalibrate", "recalibration"}:
            if len(parts) < 3:
                if len(parts) == 2 and "::" in parts[1]:
                    prov, mod = parts[1].split("::", 1)
                else:
                    return "Usage: /throttle recalibrate <provider> <model>"
            else:
                prov = parts[1]
                mod = parts[2]

            with self._cv:
                recalibrated = self._learner.recalibrate_learning(prov, mod)
                self._persist_locked()
                self._cv.notify_all()
            if recalibrated:
                return f"Recalibration started for {prov}::{mod}."
            return f"No learned limits found for {prov}::{mod}."

        if cmd in {"overrides", "override"}:
            if len(parts) == 1 or (len(parts) == 2 and parts[1].lower() in {"list", "show"}):
                return self._list_overrides_text()
            if len(parts) >= 2 and parts[1].lower() == "set":
                return self._parse_set_override(parts[2:])
            if len(parts) >= 2 and parts[1].lower() in {"remove", "unset", "delete", "clear"}:
                return self._parse_remove_override(parts[2:])
            if len(parts) >= 2 and parts[1].lower() == "get":
                return self._parse_get_override(parts[2:])
            if len(parts) >= 3:
                return self._parse_set_override(parts[1:])
            return self._list_overrides_text()

        if cmd == "set":
            return self._parse_set_override(parts[1:])

        if cmd in {"remove", "unset", "delete", "clear"}:
            return self._parse_remove_override(parts[1:])

        if cmd == "get":
            return self._parse_get_override(parts[1:])

        if cmd == "help":
            return self._help_text()

        return self._help_text()

    def _list_overrides_text(self) -> str:
        overrides = self.get_all_overrides()
        if not overrides:
            return "No provider or model overrides configured."
        lines = ["Configured Overrides:"]
        for scope, limits in sorted(overrides.items()):
            scope_type = "model" if "::" in scope else "provider"
            lim_items = []
            for k in ("rpm", "tpm", "rpd", "quota_hours", "max_rpd_wait"):
                if k in limits:
                    lim_items.append(f"{k.upper()}={limits[k]}")
            lim_str = ", ".join(lim_items) if lim_items else str(limits)
            lines.append(f"  {scope} ({scope_type}): {lim_str}")
        return "\n".join(lines)

    def _parse_get_override(self, parts: list[str]) -> str:
        if not parts:
            return "Usage: /throttle get <provider|provider::model>"
        if "::" in parts[0]:
            scope = parts[0].strip().lower()
        elif len(parts) >= 2:
            scope = f"{parts[0].strip().lower()}::{parts[1].strip().lower()}"
        else:
            scope = parts[0].strip().lower()

        override = self.get_override(scope)
        if not override:
            return f"No override configured for {scope}."
        scope_type = "model" if "::" in scope else "provider"
        lim_items = [f"{k.upper()}={v}" for k, v in sorted(override.items())]
        resolved = self.resolve_limits(scope=scope)
        return (
            f"Override for {scope} ({scope_type}):\n"
            f"  configured: {', '.join(lim_items)}\n"
            f"  resolved limits: {resolved.rpm} RPM | {resolved.tpm} TPM | {resolved.rpd} RPD"
        )

    def _parse_set_override(self, parts: list[str]) -> str:
        if not parts:
            return "Usage: /throttle set <provider|provider::model> [rpm N] [tpm N] [rpd N]"

        valid_names = {"rpm", "tpm", "rpd", "hours", "max-wait", "max_wait"}

        if "::" in parts[0]:
            scope = parts[0].strip().lower()
            param_parts = parts[1:]
        elif (
            len(parts) >= 2
            and parts[1].lower() not in valid_names
            and "=" not in parts[1]
        ):
            scope = f"{parts[0].strip().lower()}::{parts[1].strip().lower()}"
            param_parts = parts[2:]
        else:
            scope = parts[0].strip().lower()
            param_parts = parts[1:]

        if not param_parts:
            return f"Usage: /throttle set {scope} [rpm N] [tpm N] [rpd N]"

        valid_params = {
            "rpm": ("rpm", int),
            "tpm": ("tpm", int),
            "rpd": ("rpd", int),
            "hours": ("quota_hours", float),
            "max-wait": ("max_rpd_wait", float),
            "max_wait": ("max_rpd_wait", float),
        }

        parsed_values: dict[str, Any] = {}
        i = 0
        while i < len(param_parts):
            token = param_parts[i]
            if "=" in token:
                k, v = token.split("=", 1)
                k = k.lower()
                if k not in valid_params:
                    return f"Unknown parameter: {k}. Allowed: rpm, tpm, rpd, hours, max-wait"
                field_name, caster = valid_params[k]
                try:
                    val = caster(v)
                except (TypeError, ValueError):
                    return f"Invalid {k.upper()} value: {v}"
                if val <= 0 and k not in {"hours", "max-wait", "max_wait"}:
                    return f"{k.upper()} must be greater than zero."
                if val < 0:
                    return f"{k.upper()} cannot be negative."
                parsed_values[field_name] = val
                i += 1
            else:
                k = token.lower()
                if k not in valid_params:
                    return f"Unknown parameter: {k}. Allowed: rpm, tpm, rpd, hours, max-wait"
                if i + 1 >= len(param_parts):
                    return f"Missing value for parameter: {k}"
                v = param_parts[i + 1]
                field_name, caster = valid_params[k]
                try:
                    val = caster(v)
                except (TypeError, ValueError):
                    return f"Invalid {k.upper()} value: {v}"
                if val <= 0 and k not in {"hours", "max-wait", "max_wait"}:
                    return f"{k.upper()} must be greater than zero."
                if val < 0:
                    return f"{k.upper()} cannot be negative."
                parsed_values[field_name] = val
                i += 2

        if not parsed_values:
            return f"No parameters specified for {scope}."

        self.set_override(scope, **parsed_values)
        summary = ", ".join(f"{k.upper()}={v}" for k, v in sorted(parsed_values.items()))
        return f"Set override for {scope}: {summary}."

    def _parse_remove_override(self, parts: list[str]) -> str:
        if not parts:
            return "Usage: /throttle remove <provider|provider::model> [rpm|tpm|rpd]"

        valid_names = {"rpm", "tpm", "rpd", "hours", "max-wait", "max_wait"}

        if "::" in parts[0]:
            scope = parts[0].strip().lower()
            rem_parts = parts[1:]
        elif (
            len(parts) >= 2
            and parts[1].lower() not in valid_names
        ):
            scope = f"{parts[0].strip().lower()}::{parts[1].strip().lower()}"
            rem_parts = parts[2:]
        else:
            scope = parts[0].strip().lower()
            rem_parts = parts[1:]

        param = rem_parts[0].lower() if rem_parts else None
        if param:
            valid_map = {
                "rpm": "rpm",
                "tpm": "tpm",
                "rpd": "rpd",
                "hours": "quota_hours",
                "max-wait": "max_rpd_wait",
                "max_wait": "max_rpd_wait",
            }
            if param not in valid_map:
                return f"Unknown parameter to remove: {param}. Allowed: rpm, tpm, rpd, hours, max-wait"
            field_name = valid_map[param]
            removed = self.remove_override(scope, param=field_name)
            if removed:
                return f"Removed {param.upper()} override for {scope}."
            return f"No {param.upper()} override found for {scope}."
        else:
            removed = self.remove_override(scope)
            if removed:
                return f"Override removed for {scope}."
            return f"No override found for {scope}."

    @staticmethod
    def _status_int(value: Any) -> str:
        return f"{int(round(_num(value, 0))):,}"

    @staticmethod
    def _status_decimal(value: Any, places: int = 2, suffix: str = "") -> str:
        number = _num(value, 0.0)
        if abs(number) < 0.005:
            number = 0.0
        rendered = f"{number:,.{places}f}".rstrip("0").rstrip(".")
        return f"{rendered}{suffix}"

    @classmethod
    def _status_duration(cls, seconds: Any) -> str:
        value = max(0.0, _num(seconds, 0.0))
        if value < 1:
            return f"{value:.2f}s"
        minutes, remainder = divmod(int(round(value)), 60)
        hours, minutes = divmod(minutes, 60)
        days, hours = divmod(hours, 24)
        parts = []
        if days:
            parts.append(f"{days}d")
        if hours:
            parts.append(f"{hours}h")
        if minutes:
            parts.append(f"{minutes}m")
        if not parts:
            parts.append(f"{remainder}s")
        return " ".join(parts[:2])

    @classmethod
    def _status_limits(cls, settings: Settings) -> str:
        return (
            f"{cls._status_int(settings.rpm)} RPM / "
            f"{cls._status_int(settings.tpm)} TPM / "
            f"{cls._status_int(settings.rpd)} RPD"
        )

    def _status_text(self, verbose: bool = False) -> str:
        with self._cv:
            self._ensure_current_day_locked()
            settings = self._settings()
            day = self._state.get("day", {})
            stats = self._state.get("token_stats", {})
            adaptive = self._state.get("adaptive", {})

            requests_today = int(_num(day.get("requests"), 0))
            remaining = max(0, settings.rpd - requests_today)
            avg_tokens = max(
                1.0,
                _num(stats.get("avg_total_tokens"), _INITIAL_AVG_TOTAL_TOKENS),
            )

            first_request_at = day.get("first_request_at")
            if first_request_at is not None:
                try:
                    horizon_end = float(first_request_at) + settings.quota_hours * 3600.0
                    remaining_seconds = max(0.0, horizon_end - time.time())
                except (TypeError, ValueError):
                    remaining_seconds = settings.quota_hours * 3600.0
            else:
                remaining_seconds = settings.quota_hours * 3600.0

            delays = compute_delays(
                rpm=settings.rpm,
                tpm=settings.tpm,
                estimated_tokens=avg_tokens,
                remaining_requests=max(1, remaining),
                remaining_usage_seconds=remaining_seconds,
                max_rpd_wait=settings.max_rpd_wait,
                adaptive_factor=_num(adaptive.get("factor"), 1.0),
            )

            queue_depth = max(
                0, self._global_bucket._next_ticket - self._global_bucket._serving_ticket
            )
            factor = _num(adaptive.get("factor"), 1.0)

            learned = self._state.get("learned_limits", {})
            learned_lines = [
                "### Learned safe limits",
                "| Scope | Current safe (RPM / TPM) | Established ceiling (RPM / TPM) | Confidence | Observations | State |",
                "| --- | ---: | ---: | ---: | ---: | --- |",
            ]
            for key, data in sorted(learned.items()):
                prov = str(data.get("provider", "unknown"))
                mod = str(data.get("model", "unknown"))
                scope = f"{prov}::{mod}"
                conf = _num(data.get("confidence"), 0.0)
                safe_rpm = _num(data.get("safe_rpm"), settings.rpm)
                safe_tpm = _num(data.get("safe_tpm"), settings.tpm)
                obs = _num(data.get("observation_count"), 0)
                current = (
                    f"{self._status_int(safe_rpm)} / {self._status_int(safe_tpm)}"
                )
                if data.get("safe_ceiling_established"):
                    ceiling = (
                        f"{self._status_int(_num(data.get('established_safe_rpm'), safe_rpm))} / "
                        f"{self._status_int(_num(data.get('established_safe_tpm'), safe_tpm))}"
                    )
                    state = "established"
                else:
                    ceiling = "—"
                    state = "recovering" if _num(data.get("error_429_count"), 0) > 0 else "calibrating"
                learned_lines.append(
                    f"| `{scope}` | {current} | {ceiling} | {conf * 100:.0f}% | "
                    f"{self._status_int(obs)} | {state} |"
                )
            if not learned:
                learned_lines.append("| _None_ | — | — | — | — | using configured limits |")
            learned_text = "\n".join(learned_lines)

            # Scoped RPD section (rolling 24h window)
            rpd_scopes = self._state.get("rpd_scopes", {})
            now_wall = time.time()
            all_rpd_keys = set(rpd_scopes.keys())
            for bucket in self._get_active_scoped_buckets_locked():
                all_rpd_keys.add(bucket.key)

            scoped_rpd_lines = [
                "### Scoped rolling RPD usage",
                "| Scope | Used / limit | Remaining | Pacing | Reset |",
                "| --- | ---: | ---: | ---: | --- |",
            ]
            rpd_rows = 0
            if all_rpd_keys:
                for sk in sorted(all_rpd_keys):
                    s_entry = self._get_rpd_scope_entry_locked(sk)
                    s_prov = s_entry.get("provider", "")
                    s_mod = s_entry.get("model", "")
                    s_limits = self.resolve_limits(provider=s_prov, model=s_mod, scope=sk)
                    s_valid = self._prune_rpd_scope_locked(s_entry, now_wall)
                    s_count = len(s_valid)
                    s_rem = max(0, s_limits.rpd - s_count)

                    # The compact report is about usage, so omit untouched scopes.
                    if s_count == 0 and not verbose:
                        continue
                    rpd_rows += 1

                    if s_valid:
                        time_to_rolloff = max(0.0, (s_valid[0] + RPD_WINDOW_SECONDS) - now_wall)
                        rolloff_str = self._status_duration(time_to_rolloff)
                    else:
                        rolloff_str = "ready"

                    pacing_delay, _, _, _ = self.compute_rpd_pacing_locked(
                        scope_key=sk,
                        settings=s_limits,
                        now=now_wall,
                        provider=s_prov,
                        model=s_mod,
                    )
                    scoped_rpd_lines.append(
                        f"| `{sk}` | {self._status_int(s_count)} / {self._status_int(s_limits.rpd)} | "
                        f"{self._status_int(s_rem)} | {self._status_duration(pacing_delay)} | {rolloff_str} |"
                    )
            if rpd_rows == 0:
                scoped_rpd_lines.append("| _None_ | — | — | — | — |")

            scoped_rpd_text = "\n".join(scoped_rpd_lines)

            # Active scoped buckets section
            active_scoped = self._get_active_scoped_buckets_locked()
            # Configured scopes are not noteworthy by themselves. Include buckets
            # only when they have traffic/state, queue pressure, or degradation.
            active_scoped = [
                bucket for bucket in active_scoped
                if (
                    bucket._next_ticket > bucket._serving_ticket
                    or bucket._reservations
                    or bucket._token_ledger
                    or int(_num(bucket._state.get("day", {}).get("requests"), 0)) > 0
                    or _num(bucket._state.get("adaptive", {}).get("factor"), 1.0) < 1.0
                    or _num(bucket._state.get("adaptive", {}).get("cooldown_until"), 0.0) > now_wall
                    or bucket._last_delay_reason not in {"startup", "dispatch"}
                )
            ]
            scoped_lines = [
                "### Active scoped buckets",
                "| Scope | Type | Configured limits | Effective limits | Queue | Pacing | Adaptive | Reason |",
                "| --- | --- | --- | --- | ---: | ---: | ---: | --- |",
            ]
            if active_scoped:
                for bucket in active_scoped:
                    b_limits = self.resolve_limits(scope=bucket.key)
                    b_adaptive = bucket._state.get("adaptive", {})
                    b_queue = max(0, bucket._next_ticket - bucket._serving_ticket)
                    b_factor = _num(b_adaptive.get("factor"), 1.0)
                    if "::" in bucket.key:
                        b_provider, b_model = bucket.key.split("::", 1)
                        b_rpm, b_tpm = self._learner.get_effective_limits(
                            b_provider, b_model, b_limits.rpm, b_limits.tpm
                        )
                        b_effective = Settings(
                            enabled=b_limits.enabled,
                            rpm=b_rpm,
                            tpm=b_tpm,
                            rpd=b_limits.rpd,
                            quota_hours=b_limits.quota_hours,
                            max_rpd_wait=b_limits.max_rpd_wait,
                        )
                    else:
                        b_effective = b_limits
                    b_delays = compute_delays(
                        rpm=b_effective.rpm,
                        tpm=b_effective.tpm,
                        estimated_tokens=avg_tokens,
                        remaining_requests=max(1, b_limits.rpd),
                        remaining_usage_seconds=remaining_seconds,
                        max_rpd_wait=b_limits.max_rpd_wait,
                        adaptive_factor=b_factor,
                    )
                    scoped_lines.append(
                        f"| `{bucket.key}` | {bucket.scope_type} | {self._status_limits(b_limits)} | "
                        f"{self._status_limits(b_effective)} | {self._status_int(b_queue)} | "
                        f"{self._status_duration(b_delays['final'])} | {self._status_decimal(b_factor)} | "
                        f"{bucket._last_delay_reason} |"
                    )
            elif verbose:
                scoped_lines.append("| _None_ | — | — | — | — | — | — | — |")

            summary = (
                "## Global Throttle Status\n"
                f"- Enabled: **{'yes' if settings.enabled else 'no'}** | "
                f"Configured global limits: **{self._status_limits(settings)}**\n"
                f"- Current pacing: **{self._status_duration(delays['final'])}** "
                f"(RPM {self._status_duration(delays['rpm'])}, TPM {self._status_duration(delays['tpm'])}, "
                f"RPD {self._status_duration(delays['rpd'])}) | "
                f"Queue: **{self._status_int(queue_depth)}** | "
                f"Last limiter reason: **{self._last_delay_reason}**\n"
                "- Effective global limits: **same as configured**; learned model limits apply at their scoped model buckets."
            )
            if not verbose:
                sections = [summary, learned_text, scoped_rpd_text]
                if active_scoped:
                    sections.append("\n".join(scoped_lines))
                return "\n\n".join(sections)

            diagnostics = (
                "### Diagnostics\n"
                f"- Daily pacing target: {self._status_decimal(settings.quota_hours)} hours\n"
                f"- Max RPD pacing wait: {self._status_duration(settings.max_rpd_wait)}\n"
                f"- Requests today: {self._status_int(requests_today)} / {self._status_int(settings.rpd)}\n"
                f"- Rolling average tokens/request: {self._status_int(avg_tokens)}\n"
                f"- Adaptive rate factor: {self._status_decimal(factor)}"
            )
            return "\n\n".join([summary, learned_text, scoped_rpd_text, diagnostics, "\n".join(scoped_lines)])

    @staticmethod
    def _help_text() -> str:
        return (
            "Usage:\n"
            "  /throttle status\n"
            "  /throttle status verbose\n"
            "  /throttle on | off\n"
            "  /throttle rpm <number>\n"
            "  /throttle tpm <number>\n"
            "  /throttle rpd <number>\n"
            "  /throttle hours <number>\n"
            "  /throttle max-wait <seconds>\n"
            "  /throttle overrides\n"
            "  /throttle get <provider|provider::model>\n"
            "  /throttle set <provider|provider::model> [rpm N] [tpm N] [rpd N]\n"
            "  /throttle remove <provider|provider::model> [rpm|tpm|rpd]\n"
            "  /throttle reset\n"
            "  /throttle reset-learning <provider> <model>\n"
            "  /throttle recalibrate <provider> <model>"
        )


def register(ctx) -> GlobalThrottle:
    """Helper to register GlobalThrottle on a Hermes plugin context."""
    controller = GlobalThrottle(ctx)
    if hasattr(ctx, "register_hook"):
        ctx.register_hook("pre_api_request", controller.on_pre_api_request)
        ctx.register_hook("post_api_request", controller.on_post_api_request)
        ctx.register_hook("api_request_error", controller.on_api_request_error)
    if hasattr(ctx, "register_middleware"):
        ctx.register_middleware("llm_execution", controller.wrap_llm_execution)
        ctx.register_middleware("tool_execution", controller.wrap_tool_execution)
    if hasattr(ctx, "register_command"):
        ctx.register_command(
            "throttle",
            controller.handle_command,
            description="View or change the global API throttle.",
            args_hint="[status|on|off|rpm N|tpm N|rpd N|hours N|max-wait N|overrides|set <scope> ...|remove <scope>|reset|reset-learning <provider> <model>|recalibrate <provider> <model>]",
        )
    return controller
