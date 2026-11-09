# SPDX-License-Identifier: Apache-2.0
"""Load real plugin modules against the explicit module-tree fixture."""

import importlib.util
import sys

from tests.support.source import ROOT


def load_patch(relative):
    spec = importlib.util.spec_from_file_location("_capability_patch", ROOT / relative)
    mod = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(spec.name)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        if previous is None:
            sys.modules.pop(spec.name, None)
        else:
            sys.modules[spec.name] = previous
    return mod
