#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
objects.py for Jetson Nano + DeepStream.

Runs YOLO through DeepStream and saves required graphs:
- CPU usage
- GPU usage
- CPU power
- GPU power
- latency

Important: Jetson power sensor files usually need root permissions.
Run with:
    sudo -E python3 objects.py
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
DEEPSTREAM_DIR = HOME / "DeepStream-Yolo"
DEEPSTREAM_CONFIG = DEEPSTREAM_DIR / "deepstream_app_config.txt"
OUTPUT_DIR = Path.cwd() / "jetson_deepstream_results"
TEGRATS_INTERVAL_MS = 500
ENABLE_TEGRASTATS = True

JETSON_POWER_DIR = Path("/sys/devices/50000000.host1x/546c0000.i2c/i2c-6/6-0040/iio:device0")
GPU_POWER_PATH = JETSON_POWER_DIR / "in_power1_input"
CPU_POWER_PATH = JETSON_POWER_DIR / "in_power2_input"


# =========================
# HELPERS
# =========================

def get_video_path_from_deepstream_config(config_path):
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


def extract_fps_from_deepstream_line(line):
    if "PERF" not in line:
        return None
    if "PERF:" in line:
        line = line.split("PERF:", 1)[1]
    numbers = re.findall(r"\d+(?:\.\d+)?", line)
    values = []
    for n in numbers:
        value = float(n)
        if 0.1 <= value <= 1000:
            values.append(value)
    if not values:
        return None
    return values[0]


def save_plot(x, y, filename, xlabel, ylabel, title):
    if not x or not y:
        return
    n = min(len(x), len(y))
    if n <= 0:
        return
    plt.figure(figsize=(10, 6))
    plt.plot(x[:n], y[:n])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(str(filename))
    plt.close()


def start_tegrastats(log_path):
    if not ENABLE_TEGRASTATS:
        return None, None
    if shutil.which("tegrastats") is None:
        print("WARNING: tegrastats not found. GPU usage graph will not be created.")
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


def parse_gpu_usage_from_tegrastats(log_path):
    gpu_usage = []
    if not log_path.exists():
        return gpu_usage
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = re.search(r"GR3D_FREQ\s+(\d+)%", line)
            if m:
                gpu_usage.append(float(m.group(1)))
    return gpu_usage


def read_power_w(path):
    """
    Reads Jetson Nano power sensor.
    On your current Jetson readings, values look like milliwatts:
        2329 = 2.329 W
    If you see 0.00 W, run this script with sudo:
        sudo -E python3 objects.py
    """
    try:
        raw = path.read_text().strip()
        if raw == "":
            return 0.0
        return float(raw) / 1000.0
    except Exception:
        return 0.0


def monitor_cpu(stop_event, cpu_time, cpu_usage, start_time):
    while not stop_event.is_set():
        cpu_usage.append(psutil.cpu_percent(interval=0.2))
        cpu_time.append(time.time() - start_time)


def monitor_power(stop_event, power_time, gpu_power_values, cpu_power_values, start_time):
    while not stop_event.is_set():
        gpu_power_values.append(read_power_w(GPU_POWER_PATH))
        cpu_power_values.append(read_power_w(CPU_POWER_PATH))
        power_time.append(time.time() - start_time)
        time.sleep(0.2)


def check_power_permissions():
    for label, path in [("GPU", GPU_POWER_PATH), ("CPU", CPU_POWER_PATH)]:
        if not path.exists():
            print("WARNING: {} power path does not exist: {}".format(label, path))
            continue
        try:
            _ = path.read_text().strip()
        except PermissionError:
            print("WARNING: no permission to read {} power sensor: {}".format(label, path))
            print("         Run this script with: sudo -E python3 objects.py")
        except Exception as e:
            print("WARNING: cannot read {} power sensor {}: {}".format(label, path, e))


def run_deepstream():
    if not DEEPSTREAM_DIR.exists():
        raise FileNotFoundError("DeepStream-Yolo folder not found: {}".format(DEEPSTREAM_DIR))
    if not DEEPSTREAM_CONFIG.exists():
        raise FileNotFoundError("DeepStream config not found: {}".format(DEEPSTREAM_CONFIG))
    if shutil.which("deepstream-app") is None:
        raise RuntimeError("deepstream-app not found. Check DeepStream installation.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    deepstream_log = OUTPUT_DIR / "deepstream_output.log"
    tegrastats_log = OUTPUT_DIR / "tegrastats.log"

    video_path = get_video_path_from_deepstream_config(DEEPSTREAM_CONFIG)
    total_frames = count_video_frames(video_path)

    if shutil.which("stdbuf"):
        cmd = ["stdbuf", "-oL", "-eL", "deepstream-app", "-c", str(DEEPSTREAM_CONFIG)]
    else:
        cmd = ["deepstream-app", "-c", str(DEEPSTREAM_CONFIG)]

    print("Starting DeepStream benchmark")
    print("DeepStream folder: {}".format(DEEPSTREAM_DIR))
    print("Config: {}".format(DEEPSTREAM_CONFIG))
    print("Video: {}".format(video_path))
    print("Frames in video: {}".format(total_frames))
    print("Command: {}".format(" ".join(cmd)))
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

    wall_time_s = time.time() - start_time

    if return_code != 0:
        raise RuntimeError("DeepStream finished with error code {}. Check log: {}".format(return_code, deepstream_log))

    if fps_values:
        avg_fps = float(sum(fps_values) / len(fps_values))
    elif total_frames > 0 and wall_time_s > 0:
        avg_fps = float(total_frames / wall_time_s)
    else:
        avg_fps = 0.0

    avg_time_per_frame_s = float(1.0 / avg_fps) if avg_fps > 0 else 0.0
    latency_ms = [1000.0 / fps for fps in fps_values if fps > 0]

    if total_frames > 0 and avg_fps > 0:
        processing_time_s = float(total_frames / avg_fps)
    else:
        processing_time_s = float(wall_time_s)

    return {
        "avg_time_per_frame_s": avg_time_per_frame_s,
        "total_frames": total_frames,
        "total_processing_time_s": processing_time_s,
        "latency_ms": latency_ms,
        "avg_fps": avg_fps,
        "wall_time_s": wall_time_s,
        "deepstream_log": str(deepstream_log),
        "tegrastats_log": str(tegrastats_log),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    check_power_permissions()

    cpu_time = []
    cpu_usage = []
    power_time = []
    gpu_power_values = []
    cpu_power_values = []

    stop_event = threading.Event()
    start_time = time.time()

    cpu_thread = threading.Thread(target=monitor_cpu, args=(stop_event, cpu_time, cpu_usage, start_time))
    power_thread = threading.Thread(target=monitor_power, args=(stop_event, power_time, gpu_power_values, cpu_power_values, start_time))

    cpu_thread.start()
    power_thread.start()

    try:
        result = run_deepstream()
    finally:
        stop_event.set()
        cpu_thread.join()
        power_thread.join()

    total_script_time = time.time() - start_time

    latencies_ms = result.get("latency_ms", []) or []
    p95_latency_ms = float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0
    p99_latency_ms = float(np.percentile(latencies_ms, 99)) if latencies_ms else 0.0

    processed_frames = result.get("total_frames", 0) or 0
    processing_time_s = result.get("total_processing_time_s", 0) or 0

    avg_cpu_usage = float(sum(cpu_usage) / len(cpu_usage)) if cpu_usage else 0.0
    avg_gpu_power = float(sum(gpu_power_values) / len(gpu_power_values)) if gpu_power_values else 0.0
    avg_cpu_power = float(sum(cpu_power_values) / len(cpu_power_values)) if cpu_power_values else 0.0

    energy_per_frame_j = 0.0
    if processed_frames > 0 and processing_time_s > 0:
        energy_per_frame_j = ((avg_gpu_power + avg_cpu_power) * processing_time_s) / processed_frames

    tegrastats_log = Path(result.get("tegrastats_log"))
    gpu_usage = parse_gpu_usage_from_tegrastats(tegrastats_log)
    gpu_time = [i * (TEGRATS_INTERVAL_MS / 1000.0) for i in range(len(gpu_usage))]
    avg_gpu_usage = float(sum(gpu_usage) / len(gpu_usage)) if gpu_usage else 0.0

    # Required graphs
    save_plot(cpu_time, cpu_usage, OUTPUT_DIR / "plot_cpu_usage.png", "Time, s", "CPU usage, %", "CPU usage during DeepStream run")
    save_plot(gpu_time, gpu_usage, OUTPUT_DIR / "plot_gpu_usage.png", "Time, s", "GPU usage, %", "GPU usage during DeepStream run")
    save_plot(power_time, cpu_power_values, OUTPUT_DIR / "plot_cpu_power.png", "Time, s", "CPU power, W", "CPU power during DeepStream run")
    save_plot(power_time, gpu_power_values, OUTPUT_DIR / "plot_gpu_power.png", "Time, s", "GPU power, W", "GPU power during DeepStream run")
    latency_time = list(range(len(latencies_ms)))
    save_plot(latency_time, latencies_ms, OUTPUT_DIR / "plot_latency.png", "Measurement number", "Latency, ms", "Estimated DeepStream latency")

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print("Total script time: {:.2f} s".format(total_script_time))
    print("DeepStream wall time: {:.2f} s".format(result.get("wall_time_s", 0)))
    print("Processed frames: {}".format(processed_frames))
    print("avg_fps: {:.2f}".format(result.get("avg_fps", 0)))
    print("Average time per frame: {:.2f} ms".format(result.get("avg_time_per_frame_s", 0) * 1000))
    print("p95_latency_ms: {:.2f}".format(p95_latency_ms))
    print("p99_latency_ms: {:.2f}".format(p99_latency_ms))
    print("Average CPU usage: {:.2f} %".format(avg_cpu_usage))
    print("Average GPU usage from tegrastats: {:.2f} %".format(avg_gpu_usage))
    print("Average GPU power: {:.2f} W".format(avg_gpu_power))
    print("Average CPU power: {:.2f} W".format(avg_cpu_power))
    print("energy_per_frame_j: {:.6f}".format(energy_per_frame_j))
    print("DeepStream log: {}".format(result.get("deepstream_log")))
    print("tegrastats log: {}".format(result.get("tegrastats_log")))
    print("Results folder: {}".format(OUTPUT_DIR))


if __name__ == "__main__":
    main()
