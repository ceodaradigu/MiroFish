"""Keep one ChatGPT plan session stable throughout local model work.

The request owns the first reference to a lease. Background work retains a
second reference before its thread starts and releases it in ``finally``.
Account and model transitions are refused while any lease is live. No tokens,
account identifiers, or task inputs are stored here.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator

from flask import g, has_request_context


class PlanWorkBusyError(RuntimeError):
    """The plan session is being changed or is in use by an active task."""


class PlanWorkLease:
    def __init__(self, guard: "PlanWorkGuard") -> None:
        self._guard = guard
        self._references = 1

    def retain(self) -> "PlanWorkLease":
        with self._guard._lock:
            if self._references <= 0:
                raise RuntimeError("Cannot retain a released plan work lease")
            self._references += 1
            self._guard._references += 1
        return self

    def release(self) -> None:
        with self._guard._lock:
            if self._references <= 0:
                raise RuntimeError("Plan work lease released more than once")
            self._references -= 1
            self._guard._references -= 1


class PlanWorkGuard:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._references = 0
        self._transitioning = False

    def claim(self) -> PlanWorkLease:
        with self._lock:
            if self._transitioning:
                raise PlanWorkBusyError("La conexión de ChatGPT está cambiando; vuelve a intentarlo después.")
            self._references += 1
            return PlanWorkLease(self)

    def has_work(self) -> bool:
        with self._lock:
            return self._references > 0 or self._transitioning

    @contextmanager
    def transition(self) -> Iterator[None]:
        """Reserve the session before token exchange or model selection.

        The mutex is released during network I/O; new work observes the
        transition flag and cannot claim a lease until it completes.
        """

        with self._lock:
            if self._references or self._transitioning:
                raise PlanWorkBusyError("Espera a que termine el trabajo activo antes de cambiar la conexión o el modelo.")
            self._transitioning = True
        try:
            yield
        finally:
            with self._lock:
                self._transitioning = False


plan_work_guard = PlanWorkGuard()


def retain_request_plan_work() -> PlanWorkLease | None:
    """Pass a request's lease to a worker before starting its thread."""

    lease = getattr(g, "plan_work_lease", None) if has_request_context() else None
    return lease.retain() if lease is not None else None


def release_plan_work(lease: PlanWorkLease | None) -> None:
    if lease is not None:
        lease.release()
