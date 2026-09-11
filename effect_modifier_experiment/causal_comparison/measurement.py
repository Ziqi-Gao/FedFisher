"""Wall/CPU observations and logical payload accounting for local simulations.

These helpers never inspect hardware, communicate over a network, or launch
work. CPU seconds cover this process only; wall seconds include an optional,
explicit synchronization callback supplied by the caller.
"""

from contextlib import contextmanager
import json
import math
from pathlib import Path
import time


STAGES = ("data_preparation", "train", "tuning", "aggregation", "evaluation")
TIMING_OUTPUT_ROOTS = (Path("/data/del6500/FedFisher"), Path("/scr/del6500/FedFisher"))


def _nonnegative_finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("%s must be a finite nonnegative number" % name)
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("%s must be a finite nonnegative number" % name)
    return value


def _count(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError("%s must be an integer >= %d" % (name, minimum))
    return value


def _label(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("%s must be a nonempty string" % name)
    return value


def _bucket():
    return {"measured": False, "wall_seconds": 0.0, "cpu_seconds": 0.0,
            "calls": 0, "failed_calls": 0}


class MethodTimer:
    """Accumulate disjoint stages, with optional nested component detail.

    A component must be inside its matching stage and does not contribute a
    second time to stage totals. Unmeasured stages remain explicitly marked.
    Each method needs its own instance. No CUDA calls occur unless the caller
    explicitly passes a synchronization callback.
    """

    def __init__(self, method, *, wall_clock=None, cpu_clock=None, synchronize=None):
        self.method = _label(method, "method")
        self._wall_clock = time.perf_counter if wall_clock is None else wall_clock
        self._cpu_clock = time.process_time if cpu_clock is None else cpu_clock
        if not callable(self._wall_clock) or not callable(self._cpu_clock):
            raise ValueError("clocks must be callable")
        if synchronize is not None and not callable(synchronize):
            raise ValueError("synchronize must be callable or None")
        self._synchronize = synchronize
        self._stages = {stage: dict(_bucket(), components={}) for stage in STAGES}
        self._active_stage = None
        self._active_component = None

    def _stamp(self):
        wall, cpu = self._wall_clock(), self._cpu_clock()
        if not all(isinstance(value, (int, float)) and not isinstance(value, bool)
                   and math.isfinite(value) for value in (wall, cpu)):
            raise ValueError("clock readings must be finite numbers")
        return wall, cpu

    @contextmanager
    def measure(self, stage, component=None):
        if stage not in STAGES:
            raise ValueError("unknown timing stage: %s" % stage)
        if component is None:
            if self._active_stage is not None:
                raise ValueError("top-level timing stages must not overlap")
        else:
            _label(component, "component")
            if self._active_stage != stage or self._active_component is not None:
                raise ValueError("components require their matching stage and cannot overlap")
        # Flush earlier queued device work before starting this observation.
        if self._synchronize is not None:
            self._synchronize()
        start_wall, start_cpu = self._stamp()
        if component is None:
            self._active_stage = stage
        else:
            self._active_component = component
        failed = False
        try:
            yield
        except BaseException:
            failed = True
            raise
        finally:
            try:
                if self._synchronize is not None:
                    self._synchronize()
                end_wall, end_cpu = self._stamp()
                wall = _nonnegative_finite(end_wall - start_wall, "wall_seconds")
                cpu = _nonnegative_finite(end_cpu - start_cpu, "cpu_seconds")
                bucket = self._stages[stage]
                if component is not None:
                    bucket = bucket["components"].setdefault(component, _bucket())
                wall_total = _nonnegative_finite(bucket["wall_seconds"] + wall, "wall_seconds")
                cpu_total = _nonnegative_finite(bucket["cpu_seconds"] + cpu, "cpu_seconds")
                bucket.update(measured=True, wall_seconds=wall_total, cpu_seconds=cpu_total,
                              calls=bucket["calls"] + 1,
                              failed_calls=bucket["failed_calls"] + int(failed))
            finally:
                if component is None:
                    self._active_stage = None
                else:
                    self._active_component = None

    def to_dict(self):
        if self._active_stage is not None:
            raise ValueError("finish active timing stages before taking a snapshot")
        # JSON also returns a detached snapshot and rejects accidental nonfinite values.
        stages = json.loads(json.dumps(self._stages, allow_nan=False))
        return {
            "schema_version": 1,
            "method": self.method,
            "execution_model": "single_process_simulation",
            "actual_network_measured": False,
            "clock": {
                "wall": "perf_counter" if self._wall_clock is time.perf_counter else "injected",
                "cpu": "process_time" if self._cpu_clock is time.process_time else "injected",
                "cpu_scope": "current_process_only; excludes GPU execution time",
                "explicit_synchronization": self._synchronize is not None,
            },
            "stages": stages,
            "measured_stage_sum": {
                key: _nonnegative_finite(sum(stages[stage][key] for stage in STAGES), key)
                for key in ("wall_seconds", "cpu_seconds")
            },
            "scope": "instrumented stages only; not an end-to-end or deployed parallel runtime",
        }


class CommunicationLedger:
    """Count declared numeric payloads; this is not a network benchmark.

    ``payload_bytes`` is the total for an event, not a per-message size.
    Repeated ``round_id`` values count as one logical phase, including when
    upload and download events belong to the same phase. A missing round ID
    does not imply zero rounds: it marks incomplete round accounting.
    """

    def __init__(self):
        self._events = []

    def add(self, direction, payload_bytes, *, messages=1, round_id=None, label="payload"):
        if direction not in ("upload", "download"):
            raise ValueError("direction must be upload or download")
        _count(payload_bytes, "payload_bytes")
        _count(messages, "messages", minimum=1)
        _label(label, "label")
        if round_id is not None:
            _label(round_id, "round_id")
        self._events.append({"direction": direction, "payload_bytes": payload_bytes,
                             "messages": messages, "round_id": round_id, "label": label})

    def to_dict(self):
        events = [dict(event) for event in self._events]
        return {
            "execution_model": "single_process_simulation",
            "actual_network_measured": False,
            "counting_basis": "declared logical numeric payload; excludes transport/serialization overhead",
            "accounted": bool(events),
            "upload_bytes": sum(e["payload_bytes"] for e in events if e["direction"] == "upload"),
            "download_bytes": sum(e["payload_bytes"] for e in events if e["direction"] == "download"),
            "messages": sum(e["messages"] for e in events),
            "identified_logical_rounds": len({e["round_id"] for e in events if e["round_id"] is not None}),
            "round_accounting_complete": bool(events) and all(e["round_id"] is not None for e in events),
            "events": events,
        }


def timed_call(timer, stage, function, *args, **kwargs):
    """Call unchanged code with optional stage observation and no extra work."""
    if timer is None:
        return function(*args, **kwargs)
    with timer.measure(stage):
        return function(*args, **kwargs)


def validate_timing_output_path(path):
    """Preflight a new timing sidecar without creating or changing anything."""
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("timing output must use an absolute path without '..'")
    if not any(path != root and root in path.parents for root in TIMING_OUTPUT_ROOTS):
        raise ValueError("timing output must remain below the FedFisher data or scratch root")
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("timing output path must not contain symlinks")
    if path.exists():
        raise FileExistsError("refusing to overwrite timing output: %s" % path)
    return path


def write_timing_json(path, record):
    """Write one new sidecar; leave parent-directory lifecycle to the caller."""
    text = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path = validate_timing_output_path(path)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)
    return path
