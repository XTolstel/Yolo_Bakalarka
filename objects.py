#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
objects.py for Jetson Nano + DeepStream.

This script does NOT run YOLO through Ultralytics/PyTorch.
It starts DeepStream:
    deepstream-app -c ~/DeepStream-Yolo/deepstream_app_config.txt

It also:
- reads FPS from DeepStream output;
- monitors CPU load with psutil;
- logs tegrastats output;
- estimates latency from DeepStream FPS;
- estimates energy per frame from Jetson tegrastats power values;
- saves simple plots.

Before running:
    cd ~/DeepStream-Yolo
    deepstream-app -c deepstream_app_config.txt

Run the first time manually to create TensorRT .engine.
Use this script for the second and next runs.
"""

import os
import re
import time
import shutil
import signal
import subprocess
import threading
from pathlib import Path

import psutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================
# USER SETTINGS
# =========================

HOME = Path.home()

# Folder where DeepStream-Yolo is located
DEEPSTREAM_DIR = HOME / "DeepStream-Yolo"

# Main DeepStream config
DEEPSTREAM_CONFIG = DEEPSTREAM_DIR / "deepstream_app_config.txt"

# Where to save logs and plots
OUTPUT_DIR = Path.cwd() / "jetson_deepstream_results"

# tegrastats interval in ms
TEGRATS_INTERVAL_MS = 500

# If True, script will start tegrastats automatically
ENABLE_TEGRASTATS = True

# Jetson Nano INA3221 power sensor paths
JETSON_POWER_DIR = Path("/sys/devices/50000000.host1x/546c0000.i2c/i2c-6/6-0040/iio:device0")

# According to your current readings:
# in_power0_input = total board power
# in_power1_input = GPU/SOC power
# in_power2_input = CPU power
TOTAL_POWER_PATH = JETSON_POWER_DIR / "in_power0_input"
GPU_POWER_PATH = JETSON_POWER_DIR / "in_power1_input"
CPU_POWER_PATH = JETSON_POWER_DIR / "in_power2_input"

# =========================
# HELPERS
# =========================

def get_video_path_from_deepstream_config(config_path: Path):
    """
    Reads video path from line like:
        uri=file:///home/jetson/Video/test.mp4
    """
    if not config_path.exists():
        return None

    with config_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith("uri=file://"):
                uri = line.split("=", 1)[1].strip()
                return uri.replace("file://", "")

    return None


def count_video_frames(video_path):
    """
    Counts frames using OpenCV if available.
    If OpenCV is missing, returns 0.
    """
    if not video_path or not os.path.exists(video_path):
        return 0

    try:
        import cv2
    except Exception:
        return 0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frames


def extract_fps_from_deepstream_line(line: str):
    """
    Parses DeepStream performance lines.

    Possible examples:
        **PERF:  12.34 (12.01)
        **PERF:  FPS 0 (Avg) 12.34
    """
    if "PERF" not in line:
        return None

    if "PERF:" in line:
        line = line.split("PERF:", 1)[1]

    numbers = re.findall(r"\d+(?:\.\d+)?", line)
    values = []

    for n in numbers:
        value = float(n)
        # Remove source id 0 and unrealistic values
        if 0.1 <= value <= 1000:
            values.append(value)

    if not values:
        return None

    return values[0]


def start_tegrastats(log_path: Path):
    """
    Starts tegrastats and writes its output to file.
    """
    if not ENABLE_TEGRASTATS:
        return None, None

    if shutil.which("tegrastats") is None:
        print("tegrastats not found. Power/GPU monitoring will be limited.")
        return None, None

    log_file = log_path.open("w", encoding="utf-8")

    process = subprocess.Popen(
        ["tegrastats", "--interval", str(TEGRATS_INTERVAL_MS)],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        preexec_fn=os.setsid
    )

    return process, log_file


def stop_tegrastats(process, log_file):
    """
    Stops tegrastats process safely.
    """
    if process is not None:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=3)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass

    if log_file is not None:
        log_file.close()


def parse_tegrastats_log(log_path: Path):
    """
    Extracts average values from tegrastats log.

    Jetson Nano usually has fields:
        GR3D_FREQ 45%@...
        POM_5V_IN 3500/3500
        POM_5V_GPU 800/800
        POM_5V_CPU 900/900

    Power values are usually in mW.
    """
    gpu_usage = []
    total_power_w = []
    gpu_power_w = []
    cpu_power_w = []
    ram_used_mb = []

    if not log_path.exists():
        return {
            "gpu_usage_percent": 0.0,
            "total_power_w": 0.0,
            "gpu_power_w": 0.0,
            "cpu_power_w": 0.0,
            "ram_used_mb": 0.0,
        }

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # GPU usage
            m = re.search(r"GR3D_FREQ\s+(\d+)%", line)
            if m:
                gpu_usage.append(float(m.group(1)))

            # RAM used
            m = re.search(r"RAM\s+(\d+)/", line)
            if m:
                ram_used_mb.append(float(m.group(1)))

            # Jetson Nano power names
            m = re.search(r"POM_5V_IN\s+(\d+)/", line)
            if m:
                total_power_w.append(float(m.group(1)) / 1000.0)

            m = re.search(r"POM_5V_GPU\s+(\d+)/", line)
            if m:
                gpu_power_w.append(float(m.group(1)) / 1000.0)

            m = re.search(r"POM_5V_CPU\s+(\d+)/", line)
            if m:
                cpu_power_w.append(float(m.group(1)) / 1000.0)

            # Some Jetson versions use VDD_IN / VDD_GPU / VDD_CPU
            m = re.search(r"VDD_IN\s+(\d+)mW", line)
            if m:
                total_power_w.append(float(m.group(1)) / 1000.0)

            m = re.search(r"VDD_GPU\s+(\d+)mW", line)
            if m:
                gpu_power_w.append(float(m.group(1)) / 1000.0)

            m = re.search(r"VDD_CPU\s+(\d+)mW", line)
            if m:
                cpu_power_w.append(float(m.group(1)) / 1000.0)

    def avg(values):
        return float(sum(values) / len(values)) if values else 0.0

    return {
        "gpu_usage_percent": avg(gpu_usage),
        "total_power_w": avg(total_power_w),
        "gpu_power_w": avg(gpu_power_w),
        "cpu_power_w": avg(cpu_power_w),
        "ram_used_mb": avg(ram_used_mb),
    }


def save_plot(x, y, filename, xlabel, ylabel, title):
    if not x or not y:
        return

    plt.figure(figsize=(8, 6))
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(str(filename))
    plt.close()


def monitor_cpu(stop_event, cpu_usage, time_graph, start_time):
    """
    Monitors CPU load while DeepStream is running.
    """
    while not stop_event.is_set():
        cpu_usage.append(psutil.cpu_percent(interval=0.2))
        time_graph.append(time.time() - start_time)

def read_power_w(path: Path):
    """
    Reads Jetson Nano power sensor value.

    On your Jetson Nano values are in milliwatts:
        2329 = 2.329 W
    """
    try:
        value_mw = float(path.read_text().strip())
        return value_mw / 1000.0
    except Exception:
        return 0.0


def monitor_power(stop_event, gpu_power_values, cpu_power_values):
    """
    Monitors Jetson GPU and CPU power while DeepStream is running.
    """
    while not stop_event.is_set():
        gpu_power_values.append(read_power_w(GPU_POWER_PATH))
        cpu_power_values.append(read_power_w(CPU_POWER_PATH))
        time.sleep(0.2)

def run_deepstream():
    """
    Starts DeepStream and returns detection/performance metrics.
    """
    if not DEEPSTREAM_DIR.exists():
        raise FileNotFoundError(f"DeepStream-Yolo folder not found: {DEEPSTREAM_DIR}")

    if not DEEPSTREAM_CONFIG.exists():
        raise FileNotFoundError(f"DeepStream config not found: {DEEPSTREAM_CONFIG}")

    if shutil.which("deepstream-app") is None:
        raise RuntimeError("deepstream-app not found. Check DeepStream installation.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    deepstream_log = OUTPUT_DIR / "deepstream_output.log"
    tegrastats_log = OUTPUT_DIR / "tegrastats.log"

    video_path = get_video_path_from_deepstream_config(DEEPSTREAM_CONFIG)
    total_frames = count_video_frames(video_path)

    # stdbuf is used to receive DeepStream output line-by-line
    if shutil.which("stdbuf"):
        cmd = ["stdbuf", "-oL", "-eL", "deepstream-app", "-c", str(DEEPSTREAM_CONFIG)]
    else:
        cmd = ["deepstream-app", "-c", str(DEEPSTREAM_CONFIG)]

    print("Starting DeepStream benchmark")
    print(f"DeepStream folder: {DEEPSTREAM_DIR}")
    print(f"Config: {DEEPSTREAM_CONFIG}")
    print(f"Video: {video_path}")
    print(f"Frames in video: {total_frames}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)

    fps_values = []
    start_time = time.time()

    tegrastats_process, tegrastats_file = start_tegrastats(tegrastats_log)

    try:
        with deepstream_log.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                cmd,
                cwd=str(DEEPSTREAM_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )

            for line in process.stdout:
                print(line, end="")
                log_file.write(line)

                fps = extract_fps_from_deepstream_line(line)
                if fps is not None:
                    fps_values.append(fps)

            return_code = process.wait()

    finally:
        stop_tegrastats(tegrastats_process, tegrastats_file)

    end_time = time.time()
    wall_time_s = end_time - start_time

    if return_code != 0:
        raise RuntimeError(
            f"DeepStream finished with error code {return_code}. "
            f"Check log: {deepstream_log}"
        )

    if fps_values:
        avg_fps = float(sum(fps_values) / len(fps_values))
    elif total_frames > 0 and wall_time_s > 0:
        avg_fps = float(total_frames / wall_time_s)
    else:
        avg_fps = 0.0

    avg_time_per_frame_s = float(1.0 / avg_fps) if avg_fps > 0 else 0.0
    latencies_ms = [1000.0 / fps for fps in fps_values if fps > 0]

    if total_frames > 0 and avg_fps > 0:
        processing_time_s = float(total_frames / avg_fps)
    else:
        processing_time_s = float(wall_time_s)

    tegra_metrics = parse_tegrastats_log(tegrastats_log)

    energy_per_frame_j = 0.0
    if total_frames > 0 and tegra_metrics["total_power_w"] > 0:
        energy_per_frame_j = (
            tegra_metrics["total_power_w"] * processing_time_s
        ) / total_frames

    return {
        "avg_time_per_frame_s": avg_time_per_frame_s,
        "total_frames": total_frames,
        "total_processing_time_s": processing_time_s,
        "latencies_ms": latencies_ms,
        "avg_fps": avg_fps,
        "wall_time_s": wall_time_s,
        "energy_per_frame_j": energy_per_frame_j,
        "deepstream_log": str(deepstream_log),
        "tegrastats_log": str(tegrastats_log),
        **tegra_metrics,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cpu_usage = []
    time_graph = []
    gpu_power_values = []
    cpu_power_values = []
    stop_event = threading.Event()
    start_time = time.time()

    cpu_thread = threading.Thread(
        target=monitor_cpu,
        args=(stop_event, cpu_usage, time_graph, start_time)
    )
    
    power_thread = threading.Thread(
    target=monitor_power,
    args=(stop_event, gpu_power_values, cpu_power_values)
    )
    cpu_thread.start()
    power_thread.start()
    try:
        result = run_deepstream()
    finally:
        stop_event.set()
        cpu_thread.join()
        power_thread.join()

    total_time = time.time() - start_time

    avg_cpu_usage = float(sum(cpu_usage) / len(cpu_usage)) if cpu_usage else 0.0

    latencies_ms = result.get("latencies_ms", []) or []
    p95_latency_ms = float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0
    p99_latency_ms = float(np.percentile(latencies_ms, 99)) if latencies_ms else 0.0

    avg_gpu_power = float(sum(gpu_power_values) / len(gpu_power_values)) if gpu_power_values else 0.0
    avg_cpu_power = float(sum(cpu_power_values) / len(cpu_power_values)) if cpu_power_values else 0.0

    processed_frames = result.get("total_frames", 0) or 0
    processing_time_s = result.get("total_processing_time_s", 0) or 0

    energy_per_frame_j = 0.0
    if processed_frames > 0 and processing_time_s > 0:
        energy_per_frame_j = ((avg_gpu_power + avg_cpu_power) * processing_time_s) / processed_frames


    result["gpu_power_w"] = avg_gpu_power
    result["cpu_power_w"] = avg_cpu_power
    result["energy_per_frame_j"] = energy_per_frame_j
    save_plot(
        time_graph,
        cpu_usage,
        OUTPUT_DIR / "plot_cpu_usage.png",
        "Time, s",
        "CPU usage, %",
        "CPU usage during DeepStream run"
    )

    # Save FPS plot if DeepStream printed FPS values
    fps_values = [1000.0 / v for v in latencies_ms if v > 0]
    fps_time = list(range(len(fps_values)))
    save_plot(
        fps_time,
        fps_values,
        OUTPUT_DIR / "plot_deepstream_fps.png",
        "Measurement number",
        "FPS",
        "DeepStream FPS"
    )

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Total script time: {total_time:.2f} s")
    print(f"DeepStream wall time: {result.get('wall_time_s', 0):.2f} s")
    print(f"Processed frames: {result.get('total_frames', 0)}")
    print(f"avg_fps: {result.get('avg_fps', 0):.2f}")
    print(f"Average time per frame: {result.get('avg_time_per_frame_s', 0) * 1000:.2f} ms")
    print(f"p95_latency_ms: {p95_latency_ms:.2f}")
    print(f"p99_latency_ms: {p99_latency_ms:.2f}")
    print(f"Average RAM used: {result.get('ram_used_mb', 0):.2f} MB")
    print(f"Average GPU power: {result.get('gpu_power_w', 0):.2f} W")
    print(f"Average CPU power: {result.get('cpu_power_w', 0):.2f} W")
    print(f"energy_per_frame_j: {result.get('energy_per_frame_j', 0):.6f}")
    print(f"DeepStream log: {result.get('deepstream_log')}")
    print(f"tegrastats log: {result.get('tegrastats_log')}")
    print(f"Results folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()