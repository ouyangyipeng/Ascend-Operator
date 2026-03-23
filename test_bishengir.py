#!/usr/bin/env python3
"""Test bishengir-compile directly"""
import os
import subprocess
import tempfile

# Test bishengir-compile directly
ttadapter_path = "/root/.triton/cache/th3GGLR_s5Yt5MhXZlsA38N9c1ZrW-xf38SzJgBnjyE/add_kernel.ttadapter"

with tempfile.TemporaryDirectory() as tmpdir:
    output_path = os.path.join(tmpdir, "kernel.o")
    cmd = [
        "bishengir-compile",
        "--target=Ascend910B4",
        "--enable-hfusion-compile=true",
        "--enable-triton-kernel-compile=true",
        "-o", output_path,
        ttadapter_path
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(f"Return code: {result.returncode}")
    print(f"stdout: {result.stdout}")
    print(f"stderr: {result.stderr}")
    print(f"Output files: {os.listdir(tmpdir)}")