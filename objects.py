import argparse
import csv
import math
import os
import re
import subprocess
import threading
import time
from pathlib import Path

import psutil
import pandas as pd
import matplotlib.pyplot as plt


FPS_REGEX = re.compile(r"(?i)\bfps\b[^0-9]*([0-9]+(?:\.[0-9]+)?)")


def monitor_system(metrics, stop_event, start_time, sample_interval):
    """
    Собирает системные параметры во время работы Hailo Apps.
    На Raspberry Pi нет таких датчиков CPU/GPU power, как на Jetson.
    Поэтому power-поля будут NaN, если не задать их вручную.
    """
    while not stop_event.is_set():
        now = time.time() - start_time

        cpu_usage = psutil.cpu_percent(interval=None)
        ram_usage = psutil.virtual_memory().percent

        metrics.append({
            "time_s": now,
            "cpu_usage_percent": cpu_usage,
            "ram_usage_percent": ram_usage,

            # Hailo 8 — это не GPU, поэтому GPU usage как на Jetson здесь недоступен.
            "gpu_usage_percent": math.nan,

            # На Raspberry Pi нет стандартного датчика мощности CPU/GPU.
            "cpu_power_w": math.nan,
            "gpu_power_w": math.nan,

            "fps": math.nan,
            "latency_ms": math.nan,
        })

        time.sleep(sample_interval)


def read_process_output(process, fps_values):
    """
    Читает вывод object_detection.py и пытается достать FPS.
    """
    for line in iter(process.stdout.readline, ""):
        print(line, end="")

        match = FPS_REGEX.search(line)
        if match:
            try:
                fps = float(match.group(1))
                if fps > 0:
                    fps_values.append((time.time(), fps))
            except ValueError:
                pass


def save_metrics_csv(metrics, csv_path):
    df = pd.DataFrame(metrics)
    df.to_csv(csv_path, index=False)
    return df


def plot_metric(df, x_col, y_col, title, ylabel, output_path):
    if y_col not in df.columns:
        return

    if df[y_col].dropna().empty:
        print(f"[INFO] График {y_col} не построен: нет данных.")
        return

    plt.figure(figsize=(10, 5))
    plt.plot(df[x_col], df[y_col])
    plt.title(title)
    plt.xlabel("Time, s")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def get_video_frame_count(video_path):
    """
    Получает количество кадров через OpenCV.
    Если OpenCV не сможет прочитать видео, вернёт None.
    """
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if frames > 0:
            return frames
    except Exception:
        pass

    return None


def main():
    parser = argparse.ArgumentParser(description="Run YOLO object detection on Hailo 8 and collect metrics.")

    parser.add_argument(
        "-i", "--input",
        default="/home/oleg/Bakalarka_Yolo/Video/Video_test1.mp4",
        help="Path to input video"
    )

    parser.add_argument(
        "-n", "--model",
        default="yolov5m",
        help="Hailo model name or path to .hef file"
    )

    parser.add_argument(
        "-o", "--output",
        default="/home/oleg/Bakalarka_Yolo/video_hailo",
        help="Output directory"
    )

    parser.add_argument(
        "--hailo-app",
        default="/home/oleg/hailo-apps/hailo_apps/python/standalone_apps/object_detection/object_detection.py",
        help="Path to Hailo Apps object_detection.py"
    )

    parser.add_argument(
        "--sample-interval",
        type=float,
        default=1.0,
        help="Monitoring interval in seconds"
    )

    parser.add_argument(
        "--cpu-power-watts",
        type=float,
        default=None,
        help="Manual average CPU power in watts, if measured externally"
    )

    parser.add_argument(
        "--gpu-power-watts",
        type=float,
        default=None,
        help="Manual average accelerator/GPU power in watts, if measured externally"
    )

    args = parser.parse_args()

    input_video = Path(args.input)
    output_dir = Path(args.output)
    hailo_app = Path(args.hailo_app)

    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    if not hailo_app.exists():
        raise FileNotFoundError(f"Hailo object_detection.py not found: {hailo_app}")

    metrics = []
    fps_values = []

    command = [
        str(hailo_app),
        "-n", args.model,
        "-i", str(input_video),
        "--show-fps",
        "--no-display",
        "--video-unpaced",
        "--track",
        "--save-output",
        "-o", str(output_dir),
    ]

    print("\nЗапускаю Hailo object detection:")
    print(" ".join(command))
    print()

    start_time = time.time()
    stop_event = threading.Event()

    monitor_thread = threading.Thread(
        target=monitor_system,
        args=(metrics, stop_event, start_time, args.sample_interval),
        daemon=True
    )
    monitor_thread.start()

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    output_thread = threading.Thread(
        target=read_process_output,
        args=(process, fps_values),
        daemon=True
    )
    output_thread.start()

    return_code = process.wait()

    stop_event.set()
    monitor_thread.join(timeout=2)

    end_time = time.time()
    total_time = end_time - start_time

    frame_count = get_video_frame_count(input_video)

    if frame_count and total_time > 0:
        avg_fps = frame_count / total_time
        avg_latency_ms = 1000.0 / avg_fps if avg_fps > 0 else math.nan
    else:
        avg_fps = math.nan
        avg_latency_ms = math.nan

    # Если Hailo Apps вывел FPS, используем его для графика latency.
    if fps_values and metrics:
        for row in metrics:
            row_time_abs = start_time + row["time_s"]

            nearest_fps = min(
                fps_values,
                key=lambda item: abs(item[0] - row_time_abs)
            )[1]

            row["fps"] = nearest_fps
            row["latency_ms"] = 1000.0 / nearest_fps if nearest_fps > 0 else math.nan

    # Если пользователь вручную задал мощность, заполняем power-поля.
    for row in metrics:
        if args.cpu_power_watts is not None:
            row["cpu_power_w"] = args.cpu_power_watts

        if args.gpu_power_watts is not None:
            row["gpu_power_w"] = args.gpu_power_watts

    avg_cpu_usage = pd.DataFrame(metrics)["cpu_usage_percent"].mean() if metrics else math.nan

    avg_cpu_power = args.cpu_power_watts if args.cpu_power_watts is not None else math.nan
    avg_gpu_power = args.gpu_power_watts if args.gpu_power_watts is not None else math.nan

    if frame_count and args.cpu_power_watts is not None and args.gpu_power_watts is not None:
        total_power = args.cpu_power_watts + args.gpu_power_watts
        energy_per_frame_j = total_power * total_time / frame_count
    else:
        energy_per_frame_j = math.nan

    csv_path = output_dir / "hailo_metrics.csv"
    summary_path = output_dir / "hailo_summary.txt"

    df = save_metrics_csv(metrics, csv_path)

    plot_metric(
        df,
        "time_s",
        "cpu_usage_percent",
        "CPU usage during Hailo YOLO inference",
        "CPU usage, %",
        output_dir / "cpu_usage.png"
    )

    plot_metric(
        df,
        "time_s",
        "gpu_usage_percent",
        "GPU usage during Hailo YOLO inference",
        "GPU usage, %",
        output_dir / "gpu_usage.png"
    )

    plot_metric(
        df,
        "time_s",
        "cpu_power_w",
        "CPU power during Hailo YOLO inference",
        "CPU power, W",
        output_dir / "cpu_power.png"
    )

    plot_metric(
        df,
        "time_s",
        "gpu_power_w",
        "GPU / accelerator power during Hailo YOLO inference",
        "GPU / accelerator power, W",
        output_dir / "gpu_power.png"
    )

    plot_metric(
        df,
        "time_s",
        "latency_ms",
        "Latency during Hailo YOLO inference",
        "Latency, ms",
        output_dir / "latency.png"
    )

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Hailo 8 YOLO benchmark summary\n")
        f.write("--------------------------------\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Input video: {input_video}\n")
        f.write(f"Output directory: {output_dir}\n")
        f.write(f"Return code: {return_code}\n")
        f.write(f"Total time, s: {total_time:.3f}\n")
        f.write(f"Frame count: {frame_count}\n")
        f.write(f"Average FPS: {avg_fps:.3f}\n")
        f.write(f"Average latency, ms: {avg_latency_ms:.3f}\n")
        f.write(f"Average CPU usage, %: {avg_cpu_usage:.3f}\n")
        f.write(f"Average CPU power, W: {avg_cpu_power}\n")
        f.write(f"Average GPU power, W: {avg_gpu_power}\n")
        f.write(f"energy_per_frame_j: {energy_per_frame_j}\n")

    print("\nГотово.")
    print(f"CSV с метриками: {csv_path}")
    print(f"Summary: {summary_path}")
    print(f"Графики сохранены в: {output_dir}")
    print()
    print(f"Average FPS: {avg_fps:.3f}")
    print(f"Average latency, ms: {avg_latency_ms:.3f}")
    print(f"Average CPU usage, %: {avg_cpu_usage:.3f}")
    print(f"Average CPU power: {avg_cpu_power}")
    print(f"Average GPU power: {avg_gpu_power}")
    print(f"energy_per_frame_j: {energy_per_frame_j}")


if __name__ == "__main__":
    main()