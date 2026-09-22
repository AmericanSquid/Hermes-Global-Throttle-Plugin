"""Global weighted concurrency capacity for Hermes work."""

from __future__ import annotations

import threading
import time
from typing import Any, Callable


DEFAULT_WORKLOAD_WEIGHTS: dict[str, float] = {
    "interactive": 1.0,
    "tool_call": 3.0,
    "api_request": 4.0,
    "llm_api": 5.0,
    "subagent": 8.0,
    "background": 2.0,
    "other": 3.0,
}


class WeightedCapacityPool:
    """One FIFO pool that delays admission; it never cancels active work."""

    def __init__(
        self,
        capacity: float = 100.0,
        weights: dict[str, float] | None = None,
    ) -> None:
        self._cv = threading.Condition(threading.RLock())
        self._capacity = max(1.0, float(capacity))
        self._weights = dict(DEFAULT_WORKLOAD_WEIGHTS)
        if isinstance(weights, dict):
            for key, value in weights.items():
                try:
                    parsed = float(value)
                except (TypeError, ValueError):
                    continue
                if parsed > 0:
                    self._weights[str(key).strip().lower()] = parsed
        self._next_ticket = 0
        self._serving_ticket = 0
        self._cancelled_tickets: set[int] = set()
        self._active: dict[str, dict[str, Any]] = {}

    @property
    def capacity(self) -> float:
        with self._cv:
            return self._capacity

    @property
    def used(self) -> float:
        with self._cv:
            return sum(item["weight"] for item in self._active.values())

    @property
    def overcommitted(self) -> bool:
        """Whether a reduced capacity is below work already in progress."""
        with self._cv:
            return sum(item["weight"] for item in self._active.values()) > self._capacity

    @property
    def active(self) -> dict[str, dict[str, Any]]:
        with self._cv:
            return {key: dict(value) for key, value in self._active.items()}

    @property
    def weights(self) -> dict[str, float]:
        with self._cv:
            return dict(self._weights)

    def set_capacity(self, capacity: float) -> None:
        with self._cv:
            # Deliberately do not trim or interrupt existing reservations. A
            # lower capacity only prevents later admissions until enough work
            # completes naturally and releases its reservation.
            self._capacity = max(1.0, float(capacity))
            self._cv.notify_all()

    def weight_for(self, workload_type: str | None) -> tuple[str, float]:
        key = str(workload_type or "llm_api").strip().lower()
        with self._cv:
            return key, self._weights.get(key, self._weights["other"])

    def acquire(
        self,
        request_id: str,
        workload_type: str | None = None,
        tool_name: str | None = None,
        session_id: str | None = None,
        on_stall: Callable[[str, float, float, float, str | None, str | None], None] | None = None,
        on_unblock: Callable[[str, float, str | None, str | None], None] | None = None,
        stall_threshold: float = 2.0,
    ) -> tuple[str, float]:
        key, weight = self.weight_for(workload_type)
        start_mono = time.monotonic()
        stall_reported = False
        with self._cv:
            ticket = self._next_ticket
            self._next_ticket += 1
            try:
                while ticket != self._serving_ticket:
                    waited = time.monotonic() - start_mono
                    if waited >= stall_threshold and not stall_reported:
                        stall_reported = True
                        if on_stall is not None:
                            used = sum(item["weight"] for item in self._active.values())
                            on_stall(key, weight, used, self._capacity, tool_name, session_id)
                    wait_slice = max(0.05, min(0.5, stall_threshold - waited if not stall_reported else 0.5))
                    self._cv.wait(timeout=wait_slice)

                # A single work item must always be admissible, even if a
                # future resource policy lowers capacity below its weight.
                required_capacity = max(self._capacity, weight)
                # Existing reservations remain untouched when capacity is
                # constrained. This loop only delays *this* new admission.
                while sum(item["weight"] for item in self._active.values()) + weight > required_capacity:
                    waited = time.monotonic() - start_mono
                    if waited >= stall_threshold and not stall_reported:
                        stall_reported = True
                        if on_stall is not None:
                            used = sum(item["weight"] for item in self._active.values())
                            on_stall(key, weight, used, self._capacity, tool_name, session_id)
                    wait_slice = max(0.05, min(0.5, stall_threshold - waited if not stall_reported else 0.5))
                    self._cv.wait(timeout=wait_slice)

                self._active[request_id] = {
                    "workload_type": key,
                    "weight": weight,
                }
            except BaseException:
                if ticket == self._serving_ticket:
                    self._serving_ticket += 1
                else:
                    self._cancelled_tickets.add(ticket)
                while self._serving_ticket in self._cancelled_tickets:
                    self._cancelled_tickets.remove(self._serving_ticket)
                    self._serving_ticket += 1
                self._cv.notify_all()
                raise
            self._serving_ticket += 1
            while self._serving_ticket in self._cancelled_tickets:
                self._cancelled_tickets.remove(self._serving_ticket)
                self._serving_ticket += 1
            self._cv.notify_all()

        if stall_reported and on_unblock is not None:
            on_unblock(key, weight, tool_name, session_id)
        return key, weight

    def release(self, request_id: str) -> bool:
        with self._cv:
            removed = self._active.pop(request_id, None) is not None
            if removed:
                self._cv.notify_all()
            return removed
