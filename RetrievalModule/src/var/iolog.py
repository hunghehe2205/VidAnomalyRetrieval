"""iolog — simple tee stdout/stderr to a file, plus timestamped prefix."""
from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, TextIO


class _Tee:
    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        count = 0
        for s in self._streams:
            count = s.write(data)
            s.flush()
        return count

    def flush(self) -> None:
        for s in self._streams:
            s.flush()


@contextmanager
def tee_to_file(log_path: Path) -> Iterator[Path]:
    """Mirror stdout and stderr into `log_path` for the duration of the context."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_path, "a", encoding="utf-8", buffering=1)
    prev_out, prev_err = sys.stdout, sys.stderr
    sys.stdout = _Tee(prev_out, f)
    sys.stderr = _Tee(prev_err, f)
    try:
        yield log_path
    finally:
        sys.stdout = prev_out
        sys.stderr = prev_err
        f.close()


def ts() -> str:
    return time.strftime("%H:%M:%S")


def log(scope: str, msg: str) -> None:
    """Print a timestamped, scoped line. Gets teed to file automatically
    when called inside `tee_to_file`."""
    print(f"[{ts()}] [{scope}] {msg}", flush=True)


def new_log_filename(prefix: str) -> str:
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}.log"
