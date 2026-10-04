#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Make ``src`` importable for the test suite without an editable install."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
