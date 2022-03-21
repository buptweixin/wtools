#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from IPython import get_ipython

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
