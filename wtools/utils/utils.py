#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from IPython import get_ipython

from collections import defaultdict
from typing import Any, Dict, List, Optional
from tabulate import tabulate
import os
import time
import psutil

#  ref: https://github.com/ppwwyyxx/RAM-multiprocess-dataloader/blob/main/common.py
def get_mem_info(pid: int) -> Dict[str, int]:
  """Collect detailed memory usage statistics for a given process.

    Aggregates ``rss`` (resident set size), ``pss`` (proportional set size),
    ``uss`` (unique set size), ``shared``, and ``shared_file`` (shared memory
    backed by a file) across all memory-mapped regions of the process.

    Args:
        pid: The process ID to query.

    Returns:
        A dictionary mapping memory metric names (``"rss"``, ``"pss"``,
        ``"uss"``, ``"shared"``, ``"shared_file"``) to their values in
        bytes.

    Examples:
        >>> info = get_mem_info(12345)
        >>> info["rss"]
        123456789
    """
  # Aggregate memory metrics in one pass over the memory-mapped regions.
  # Local accumulators are used (instead of defaultdict item access) because
  # local-variable lookups are faster than dict __getitem__/__setitem__ in
  # tight loops, and the set of keys is known and fixed.
  rss = pss = uss = shared = shared_file = 0
  try:
    memory_maps = psutil.Process(pid).memory_maps()
  except psutil.NoSuchProcess:
    raise psutil.NoSuchProcess(
      f"Process with pid={pid} does not exist or has already terminated."
    )
  for mmap in memory_maps:
    rss += mmap.rss
    pss += mmap.pss
    uss += mmap.private_clean + mmap.private_dirty
    shared_clean = mmap.shared_clean
    shared_dirty = mmap.shared_dirty
    shared_total = shared_clean + shared_dirty
    shared += shared_total
    # Only file-backed mappings (path starting with '/') contribute to
    # shared_file, so we skip the string check for anonymous mappings.
    if shared_total and mmap.path.startswith('/'):
      shared_file += shared_total
  return {
      'rss': rss,
      'pss': pss,
      'uss': uss,
      'shared': shared,
      'shared_file': shared_file,
  }


class MemoryMonitor():
  """Monitor memory usage of one or more processes.

    This class wraps :func:`get_mem_info` to periodically collect and display
    memory statistics (RSS, PSS, USS, shared, shared_file) for a set of
    process IDs. It is useful for profiling memory consumption of
    multi-process data loaders or training pipelines.

    Attributes:
        pids: A list of process IDs being monitored.

    Args:
        pids: A list of process IDs to monitor. If ``None``, defaults to
            the current process (``os.getpid()``).

    Examples:
        >>> monitor = MemoryMonitor()
        >>> monitor.add_pid(12345)
        >>> print(monitor.table())
        >>> print(monitor.str())
    """

  def __init__(self, pids: Optional[List[int]] = None):
    if pids is None:
      pids = [os.getpid()]
    self.pids = pids

  def add_pid(self, pid: int) -> None:
    """Add a process ID to the monitored set.

        Args:
            pid: The process ID to add. Must not already be in ``self.pids``.

        Raises:
            AssertionError: If ``pid`` is already in ``self.pids``.
        """
    assert pid not in self.pids
    self.pids.append(pid)

  def _refresh(self) -> Dict[int, Dict[str, int]]:
    self.data = {pid: get_mem_info(pid) for pid in self.pids}
    return self.data

  def table(self) -> "str":
    """Return a formatted tabular string of current memory usage.

        Refreshes the collected data and formats it as a table with columns
        for time, PID, and each memory metric. Sizes are human-readable
        (e.g. ``"1.2G"``).

        Returns:
            A string containing the formatted table (produced by
            ``tabulate``).
        """
    self._refresh()
    table = []
    keys = list(list(self.data.values())[0].keys())
    now = str(int(time.perf_counter() % 1e5))
    for pid, data in self.data.items():
      table.append((now, str(pid)) + tuple(self.format(data[k]) for k in keys))
    return str(tabulate(table, headers=["time", "PID"] + keys))

  def str(self) -> "str":
    """Return a one-line-per-process summary of current memory usage.

        Refreshes the collected data and formats each process as
        ``PID=<pid>, rss=<val>, pss=<val>, ...``.

        Returns:
            A string with one line per monitored process.
        """
    self._refresh()
    keys = list(list(self.data.values())[0].keys())
    res = []
    for pid in self.pids:
      s = f"PID={pid}"
      for k in keys:
        v = self.format(self.data[pid][k])
        s += f", {k}={v}"
      res.append(s)
    return "\n".join(res)

  @staticmethod
  def format(size: float) -> "str":  # type: ignore[valid-type]
    """Format a byte count into a human-readable string.

        Args:
            size: A size in bytes.

        Returns:
            A string like ``"1.5M"`` or ``"2.0G"``.
        """
    for unit in ('', 'K', 'M', 'G'):
      if size < 1024:
        break
      size /= 1024.0
    return "%.1f%s" % (size, unit)

# 判断当前执行环境是否是notebook
def isnotebook() -> bool:
    """Determine whether the current execution environment is a Jupyter notebook.

    Detects the IPython shell class to distinguish between a Jupyter notebook
    / QtConsole, a terminal IPython session, and a standard Python
    interpreter.

    Returns:
        ``True`` if running inside a Jupyter notebook or QtConsole,
        ``False`` otherwise.

    Examples:
        >>> if isnotebook():
        ...     from tqdm.notebook import tqdm
        ... else:
        ...     from tqdm import tqdm
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter
