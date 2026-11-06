#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Observe the installed test runtime without claiming host/BOM verification."""

import json
import os
import platform
from pathlib import Path


def main():
    import torch
    from vllm.platforms import current_platform

    assert current_platform.is_ppu(), "The active platform is not PPU"
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.cuda.synchronize()
    sdk = Path(os.environ.get("PPU_SDK", "/usr/local/PPU_SDK"))
    result = {
        "schema": 1,
        "image": os.environ["OBSERVED_IMAGE"],
        "python": platform.python_version(),
        "os": platform.platform(),
        "torch": torch.__version__,
        "sdk": {"root": str(sdk), "envsetup_present": (sdk / "envsetup.sh").is_file()},
        "gpu": {
            "count": torch.cuda.device_count(),
            "capability": list(torch.cuda.get_device_capability(0)),
        },
        "host_contract": {"kmd": "not_measured", "firmware": "not_measured"},
        "quality_scope": "minimum-regression",
        "verified_bom": False,
    }
    Path("/results/runtime.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="ascii"
    )
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
