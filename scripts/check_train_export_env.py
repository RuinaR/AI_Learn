from __future__ import annotations

import importlib
import platform
import sys


def main() -> int:
    print(f"python={sys.version.split()[0]}")
    print(f"platform={platform.platform()}")

    required = ["torch", "ultralytics", "onnx", "onnxruntime", "tensorrt"]
    failed = False
    for name in required:
        try:
            module = importlib.import_module(name)
            print(f"{name}=OK version={getattr(module, '__version__', 'unknown')}")
        except Exception as exc:
            failed = True
            print(f"{name}=FAIL error={exc!r}")

    try:
        import torch

        print(f"cuda_available={torch.cuda.is_available()}")
        print(f"cuda_device_count={torch.cuda.device_count()}")
        if torch.cuda.is_available():
            print(f"cuda_device_name={torch.cuda.get_device_name(0)}")
    except Exception as exc:
        failed = True
        print(f"torch_cuda_check=FAIL error={exc!r}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
