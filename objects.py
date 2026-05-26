#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
objects.py for Jetson Nano + DeepStream.

This version runs YOLO through DeepStream and saves:
- text summary with the same key benchmark fields as the Hailo version;
- CPU usage graph;
- GPU usage graph;
- RAM usage graph in MB;
- CPU power graph;
- GPU power graph;
- latency graph;
- FPS graph;
- CSV file with raw sampled metrics.

Important:
Jetson power sensor files usually need root permissions.
Run with:
    sudo -E python3 objects.py
"""

import csv
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

# Power paths for Jetson Nano.
# On your current Jetson readings, values look like milliwatts:
# 2329 = 2.329 W
JETSON_POWER_DIR = Path("/sys/devices/50000000.host1x/546c0000.i2c/i2c-6/6-0040/iio:device0")
GPU_POWER_PATH = JETSON_POWER_DIR / "in_power1_input"
CPU_POWER_PATH = JETSON_POWER_DIR / "in_power2_input"

MONITOR_INTERVAL_S = 0.2


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


def get_video_info(video_path):
    """
    Returns:
        frame_count, source_fps
    """
    if not video_path or not os.path.exists(video_path):
        return 0, 0.0

    try:
        import cv2
    except Exception:
        return 0, 0.0

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return 0, 0.0

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = float(cap.get(cv2.CAP_PROP_FPS))

    cap.release()

    if frames < 0:
        frames = 0
    if source_fps < 0:
        source_fps = 0.0

    return frames, source_fps


def extract_fps_from_deepstream_line(line):
    """
    Extracts FPS from DeepStream PERF lines.

    Example DeepStream output usually contains:
        **PERF:  23.4 (23.2)
    """
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
    valid_x = []
    valid_y = []

    n = min(len(x), len(y))
    for i in range(n):
        value = y[i]
        if value is None:
            continue
        try:
            if np.isnan(value):
                continue
        except Exception:
            pass

        valid_x.append(x[i])
        valid_y.append(value)

    if not valid_x or not valid_y:
        print("[INFO] Graph was not created, no data: {}".format(filename))
        return

    plt.figure(figsize=(10, 6))
    plt.plot(valid_x, valid_y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(str(filename), dpi=150)
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
            match = re.search(r"GR3D_FREQ\s+(\d+)%", line)
            if match:
                gpu_usage.append(float(match.group(1)))

    return gpu_usage


def read_power_w(path):
    """
    Reads Jetson Nano power sensor.

    If you see 0.00 W, run:
        sudo -E python3 objects.py
    """
    try:
        raw = path.read_text().strip()
        if raw == "":
            return 0.0

        return float(raw) / 1000.0

    except Exception:
        return 0.0


def monitor_system(stop_event, sample_time, cpu_usage, ram_used_mb, ram_percent, start_time):
    """
    Monitors CPU and RAM while DeepStream is running.
    RAM is saved in MB, as requested.
    """
    psutil.cpu_percent(interval=None)

    while not stop_event.is_set():
        now = time.time() - start_time
        vm = psutil.virtual_memory()

        sample_time.append(now)
        cpu_usage.append(psutil.cpu_percent(interval=None))
        ram_used_mb.append(vm.used / (1024.0 * 1024.0))
        ram_percent.append(vm.percent)

        time.sleep(MONITOR_INTERVAL_S)


def monitor_power(stop_event, power_time, gpu_power_values, cpu_power_values, start_time):
    while not stop_event.is_set():
        gpu_power_values.append(read_power_w(GPU_POWER_PATH))
        cpu_power_values.append(read_power_w(CPU_POWER_PATH))
        power_time.append(time.time() - start_time)

        time.sleep(MONITOR_INTERVAL_S)


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


def save_summary_txt(summary_path, result):
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("Jetson DeepStream YOLO benchmark summary\n")
        f.write("----------------------------------------\n")
        f.write("Model / pipeline: DeepStream-Yolo\n")
        f.write("DeepStream config: {}\n".format(result["deepstream_config"]))
        f.write("Input video: {}\n".format(result["input_video"]))
        f.write("Output directory: {}\n".format(result["output_dir"]))
        f.write("Return code: {}\n".format(result["return_code"]))
        f.write("Total time, s: {:.3f}\n".format(result["total_time_s"]))
        f.write("DeepStream wall time, s: {:.3f}\n".format(result["deepstream_wall_time_s"]))
        f.write("Frame count in video: {}\n".format(result["frame_count"]))
        f.write("Processed frames: {}\n".format(result["processed_frames"]))
        f.write("Source video FPS: {:.3f}\n".format(result["source_video_fps"]))
        f.write("Processing FPS: {:.3f}\n".format(result["processing_fps"]))
        f.write("Average latency, ms: {:.3f}\n".format(result["average_latency_ms"]))
        f.write("p95_latency_ms: {:.3f}\n".format(result["p95_latency_ms"]))
        f.write("p99_latency_ms: {:.3f}\n".format(result["p99_latency_ms"]))
        f.write("Average CPU usage, %: {:.3f}\n".format(result["avg_cpu_usage"]))
        f.write("Average GPU usage, %: {:.3f}\n".format(result["avg_gpu_usage"]))
        f.write("Average RAM used, MB: {:.3f}\n".format(result["avg_ram_used_mb"]))
        f.write("Max RAM used, MB: {:.3f}\n".format(result["max_ram_used_mb"]))
        f.write("Average CPU power, W: {:.3f}\n".format(result["avg_cpu_power_w"]))
        f.write("Average GPU power, W: {:.3f}\n".format(result["avg_gpu_power_w"]))
        f.write("energy_per_frame_j: {:.6f}\n".format(result["energy_per_frame_j"]))
        f.write("DeepStream log: {}\n".format(result["deepstream_log"]))
        f.write("tegrastats log: {}\n".format(result["tegrastats_log"]))
        f.write("Metrics CSV: {}\n".format(result["metrics_csv"]))


def save_metrics_csv(csv_path, sample_time, cpu_usage, ram_used_mb, ram_percent,
                     power_time, cpu_power_values, gpu_power_values,
                     fps_time, fps_values, latency_values):
    """
    Saves sampled metrics to CSV.
    Arrays are sampled independently, so rows are matched by index where possible.
    """
    max_len = max(
        len(sample_time),
        len(power_time),
        len(fps_time),
        1
    )

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "index",
            "sample_time_s",
            "cpu_usage_percent",
            "ram_used_mb",
            "ram_percent",
            "power_time_s",
            "cpu_power_w",
            "gpu_power_w",
            "fps_time_s",
            "fps",
            "latency_ms"
        ])

        for i in range(max_len):
            writer.writerow([
                i,
                sample_time[i] if i < len(sample_time) else "",
                cpu_usage[i] if i < len(cpu_usage) else "",
                ram_used_mb[i] if i < len(ram_used_mb) else "",
                ram_percent[i] if i < len(ram_percent) else "",
                power_time[i] if i < len(power_time) else "",
                cpu_power_values[i] if i < len(cpu_power_values) else "",
                gpu_power_values[i] if i < len(gpu_power_values) else "",
                fps_time[i] if i < len(fps_time) else "",
                fps_values[i] if i < len(fps_values) else "",
                latency_values[i] if i < len(latency_values) else "",
            ])


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
    total_frames, source_video_fps = get_video_info(video_path)

    if shutil.which("stdbuf"):
        cmd = ["stdbuf", "-oL", "-eL", "deepstream-app", "-c", str(DEEPSTREAM_CONFIG)]
    else:
        cmd = ["deepstream-app", "-c", str(DEEPSTREAM_CONFIG)]

    print("Starting DeepStream benchmark")
    print("DeepStream folder: {}".format(DEEPSTREAM_DIR))
    print("Config: {}".format(DEEPSTREAM_CONFIG))
    print("Video: {}".format(video_path))
    print("Frames in video: {}".format(total_frames))
    print("Source video FPS: {:.3f}".format(source_video_fps))
    print("Command: {}".format(" ".join(cmd)))
    print("-" * 60)

    fps_values = []
    fps_time = []

    start_time = time.time()
    tegrastats_process, tegrastats_file = start_tegrastats(tegrastats_log)

    return_code = None

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
                    fps_time.append(time.time() - start_time)

            return_code = process.wait()

    finally:
        stop_tegrastats(tegrastats_process, tegrastats_file)

    wall_time_s = time.time() - start_time

    if return_code != 0:
        raise RuntimeError(
            "DeepStream finished with error code {}. Check log: {}".format(
                return_code,
                deepstream_log
            )
        )

    if fps_values:
        avg_fps = float(sum(fps_values) / len(fps_values))
    elif total_frames > 0 and wall_time_s > 0:
        avg_fps = float(total_frames / wall_time_s)
    else:
        avg_fps = 0.0

    average_latency_ms = float(1000.0 / avg_fps) if avg_fps > 0 else 0.0
    latency_ms = [1000.0 / fps for fps in fps_values if fps > 0]

    if not latency_ms and avg_fps > 0:
        latency_ms = [average_latency_ms]

    if total_frames > 0 and avg_fps > 0:
        processing_time_s = float(total_frames / avg_fps)
    else:
        processing_time_s = float(wall_time_s)

    return {
        "return_code": return_code,
        "frame_count": total_frames,
        "source_video_fps": source_video_fps,
        "processed_frames": total_frames if return_code == 0 else 0,
        "total_processing_time_s": processing_time_s,
        "average_latency_ms": average_latency_ms,
        "latency_ms": latency_ms,
        "fps_time": fps_time,
        "fps_values": fps_values,
        "avg_fps": avg_fps,
        "wall_time_s": wall_time_s,
        "deepstream_log": str(deepstream_log),
        "tegrastats_log": str(tegrastats_log),
        "input_video": video_path,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    check_power_permissions()

    sample_time = []
    cpu_usage = []
    ram_used_mb = []
    ram_percent = []

    power_time = []
    gpu_power_values = []
    cpu_power_values = []

    stop_event = threading.Event()
    start_time = time.time()

    system_thread = threading.Thread(
        target=monitor_system,
        args=(stop_event, sample_time, cpu_usage, ram_used_mb, ram_percent, start_time)
    )

    power_thread = threading.Thread(
        target=monitor_power,
        args=(stop_event, power_time, gpu_power_values, cpu_power_values, start_time)
    )

    system_thread.start()
    power_thread.start()

    try:
        result = run_deepstream()

    finally:
        stop_event.set()
        system_thread.join()
        power_thread.join()

    total_script_time = time.time() - start_time

    latencies_ms = result.get("latency_ms", []) or []
    p95_latency_ms = float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0
    p99_latency_ms = float(np.percentile(latencies_ms, 99)) if latencies_ms else 0.0

    processed_frames = result.get("processed_frames", 0) or 0
    processing_time_s = result.get("total_processing_time_s", 0) or 0

    avg_cpu_usage = float(sum(cpu_usage) / len(cpu_usage)) if cpu_usage else 0.0

    avg_ram_used_mb = float(sum(ram_used_mb) / len(ram_used_mb)) if ram_used_mb else 0.0
    max_ram_used_mb = float(max(ram_used_mb)) if ram_used_mb else 0.0

    avg_gpu_power = float(sum(gpu_power_values) / len(gpu_power_values)) if gpu_power_values else 0.0
    avg_cpu_power = float(sum(cpu_power_values) / len(cpu_power_values)) if cpu_power_values else 0.0

    energy_per_frame_j = 0.0
    if processed_frames > 0 and processing_time_s > 0:
        energy_per_frame_j = ((avg_gpu_power + avg_cpu_power) * processing_time_s) / processed_frames

    tegrastats_log = Path(result.get("tegrastats_log"))
    gpu_usage = parse_gpu_usage_from_tegrastats(tegrastats_log)
    gpu_time = [i * (TEGRATS_INTERVAL_MS / 1000.0) for i in range(len(gpu_usage))]
    avg_gpu_usage = float(sum(gpu_usage) / len(gpu_usage)) if gpu_usage else 0.0

    fps_values = result.get("fps_values", []) or []
    fps_time = result.get("fps_time", []) or []

    if not fps_values and result.get("avg_fps", 0) > 0:
        fps_values = [float(result.get("avg_fps", 0))]
        fps_time = [0.0]

    # Required graphs
    save_plot(
        sample_time,
        cpu_usage,
        OUTPUT_DIR / "plot_cpu_usage.png",
        "Time, s",
        "CPU usage, %",
        "CPU usage during DeepStream run"
    )

    save_plot(
        gpu_time,
        gpu_usage,
        OUTPUT_DIR / "plot_gpu_usage.png",
        "Time, s",
        "GPU usage, %",
        "GPU usage during DeepStream run"
    )

    save_plot(
        sample_time,
        ram_used_mb,
        OUTPUT_DIR / "plot_ram_used_mb.png",
        "Time, s",
        "RAM used, MB",
        "RAM usage during DeepStream run"
    )

    save_plot(
        power_time,
        cpu_power_values,
        OUTPUT_DIR / "plot_cpu_power.png",
        "Time, s",
        "CPU power, W",
        "CPU power during DeepStream run"
    )

    save_plot(
        power_time,
        gpu_power_values,
        OUTPUT_DIR / "plot_gpu_power.png",
        "Time, s",
        "GPU power, W",
        "GPU power during DeepStream run"
    )

    latency_time = list(range(len(latencies_ms)))
    save_plot(
        latency_time,
        latencies_ms,
        OUTPUT_DIR / "plot_latency.png",
        "Measurement number",
        "Latency, ms",
        "Estimated DeepStream latency"
    )

    save_plot(
        fps_time,
        fps_values,
        OUTPUT_DIR / "plot_fps.png",
        "Time, s",
        "FPS",
        "FPS during DeepStream run"
    )

    metrics_csv = OUTPUT_DIR / "jetson_metrics.csv"
    save_metrics_csv(
        metrics_csv,
        sample_time,
        cpu_usage,
        ram_used_mb,
        ram_percent,
        power_time,
        cpu_power_values,
        gpu_power_values,
        fps_time,
        fps_values,
        latencies_ms
    )

    summary_result = {
        "deepstream_config": str(DEEPSTREAM_CONFIG),
        "input_video": result.get("input_video"),
        "output_dir": str(OUTPUT_DIR),
        "return_code": result.get("return_code", 0),
        "total_time_s": total_script_time,
        "deepstream_wall_time_s": result.get("wall_time_s", 0),
        "frame_count": result.get("frame_count", 0),
        "processed_frames": processed_frames,
        "source_video_fps": result.get("source_video_fps", 0),
        "processing_fps": result.get("avg_fps", 0),
        "average_latency_ms": result.get("average_latency_ms", 0),
        "p95_latency_ms": p95_latency_ms,
        "p99_latency_ms": p99_latency_ms,
        "avg_cpu_usage": avg_cpu_usage,
        "avg_gpu_usage": avg_gpu_usage,
        "avg_ram_used_mb": avg_ram_used_mb,
        "max_ram_used_mb": max_ram_used_mb,
        "avg_cpu_power_w": avg_cpu_power,
        "avg_gpu_power_w": avg_gpu_power,
        "energy_per_frame_j": energy_per_frame_j,
        "deepstream_log": result.get("deepstream_log"),
        "tegrastats_log": result.get("tegrastats_log"),
        "metrics_csv": str(metrics_csv),
    }

    summary_path = OUTPUT_DIR / "jetson_summary.txt"
    save_summary_txt(summary_path, summary_result)

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print("Total script time: {:.2f} s".format(total_script_time))
    print("DeepStream wall time: {:.2f} s".format(result.get("wall_time_s", 0)))
    print("Frame count in video: {}".format(result.get("frame_count", 0)))
    print("Processed frames: {}".format(processed_frames))
    print("Source video FPS: {:.3f}".format(result.get("source_video_fps", 0)))
    print("Processing FPS: {:.3f}".format(result.get("avg_fps", 0)))
    print("Average latency, ms: {:.3f}".format(result.get("average_latency_ms", 0)))
    print("p95_latency_ms: {:.3f}".format(p95_latency_ms))
    print("p99_latency_ms: {:.3f}".format(p99_latency_ms))
    print("Average CPU usage, %: {:.3f}".format(avg_cpu_usage))
    print("Average GPU usage, %: {:.3f}".format(avg_gpu_usage))
    print("Average RAM used, MB: {:.3f}".format(avg_ram_used_mb))
    print("Max RAM used, MB: {:.3f}".format(max_ram_used_mb))
    print("Average CPU power, W: {:.3f}".format(avg_cpu_power))
    print("Average GPU power, W: {:.3f}".format(avg_gpu_power))
    print("energy_per_frame_j: {:.6f}".format(energy_per_frame_j))
    print("Summary: {}".format(summary_path))
    print("Metrics CSV: {}".format(metrics_csv))
    print("DeepStream log: {}".format(result.get("deepstream_log")))
    print("tegrastats log: {}".format(result.get("tegrastats_log")))
    print("Results folder: {}".format(OUTPUT_DIR))


if __name__ == "__main__":
    main()
