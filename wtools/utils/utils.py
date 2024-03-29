#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from IPython import get_ipython

from collections import defaultdict
from typing import Any, Dict, List
from tabulate import tabulate
import os
import time
import psutil

#  ref: https://github.com/ppwwyyxx/RAM-multiprocess-dataloader/blob/main/common.py 
def get_mem_info(pid: int) -> Dict[str, int]:
  res = defaultdict(int)
  for mmap in psutil.Process(pid).memory_maps():
    res['rss'] += mmap.rss
    res['pss'] += mmap.pss
    res['uss'] += mmap.private_clean + mmap.private_dirty
    res['shared'] += mmap.shared_clean + mmap.shared_dirty
    if mmap.path.startswith('/'):
      res['shared_file'] += mmap.shared_clean + mmap.shared_dirty
  return res


class MemoryMonitor():
  def __init__(self, pids: List[int] = None):
    if pids is None:
      pids = [os.getpid()]
    self.pids = pids

  def add_pid(self, pid: int):
    assert pid not in self.pids
    self.pids.append(pid)

  def _refresh(self):
    self.data = {pid: get_mem_info(pid) for pid in self.pids}
    return self.data

  def table(self) -> str:
    self._refresh()
    table = []
    keys = list(list(self.data.values())[0].keys())
    now = str(int(time.perf_counter() % 1e5))
    for pid, data in self.data.items():
      table.append((now, str(pid)) + tuple(self.format(data[k]) for k in keys))
    return tabulate(table, headers=["time", "PID"] + keys)

  def str(self):
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
  def format(size: int) -> str:
    for unit in ('', 'K', 'M', 'G'):
      if size < 1024:
        break
      size /= 1024.0
    return "%.1f%s" % (size, unit)

# 判断当前执行环境是否是notebook
def isnotebook() -> bool:
    """判断当前执行环境是否是notebook

    Returns:
        bool -- 如果在 notebook 则返回 True， 否则 False
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
