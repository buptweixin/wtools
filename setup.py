#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Setup shim - all configuration lives in pyproject.toml.

This file is kept for backwards compatibility so that ``python setup.py ...``
still works.  The version and all package metadata are declared in
``pyproject.toml`` (PEP 621).  The version itself is read dynamically from
``wtools/__version__.py``.
"""

from setuptools import setup

if __name__ == "__main__":
    setup()
