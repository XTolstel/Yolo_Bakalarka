import argparse
import math
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil
import pandas as pd
import matplotlib.pyplot as plt


FPS_REGEXES = [
    re.compile(r"(?i)processed\s+\d+\s+frames\s+at\s+([0-9]+(?:\.[0-9]+)?)\s*fps"),
    re.compile(r"(?i)([0-9]+(?:\.[0-9]+)?)\s*fps"),
]


def get_video_info(video_path: Path):
    """
    Возвращает количество кадров и Source video FPS исходного видео.
    """
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))

        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        source_fps = float(cap.get(cv2.CAP_PROP_FPS))

        cap.release()

        if frames <= 0:
            frames = None

        if source_fps <= 0:
            source_fps = math.nan

        return frames, source_fps

    except Exception:
        return None, math.nan
    

def get_process_memory_mb(pid):
    """
    Возвращает RAM, занятую процессом object_detection.py
    вместе с дочерними процессами, в мегабайтах.
    """
    try:
        process = psutil.Process(pid)

        total_memory = process.memory_info().rss

        for child in process.children(recursive=True):
            try:
                total_memory += child.memory_info().rss
            except psutil.NoSuchProcess:
                pass

        return total_memory / (1024 * 1024)

    except psutil.NoSuchProcess:
        return math.nan


def monitor_system(metrics, stop_event, start_time, sample_interval, process_pid):
    psutil.cpu_percent(interval=None)

    while not stop_event.is_set():
        now = time.time() - start_time

        metrics.append({
            "time_s": now,
            "cpu_usage_percent": psutil.cpu_percent(interval=None),

            # RAM процесса object_detection.py в мегабайтах
            "ram_used_mb": get_process_memory_mb(process_pid),

            "fps": math.nan,
            "latency_ms": math.nan,

            # Для совместимости с Jetson-кодом поля оставлены.
            "cpu_power_w": math.nan,
            "gpu_power_w": math.nan,
        })

        time.sleep(sample_interval)


def read_process_output(process, fps_values):
    for line in iter(process.stdout.readline, ""):
        print(line, end="")

        for regex in FPS_REGEXES:
            match = regex.search(line)
            if match:
                try:
                    fps = float(match.group(1))
                    if fps > 0:
                        fps_values.append((time.time(), fps))
                except ValueError:
                    pass
                break


def fill_fps_and_latency(metrics, fps_values, start_time, processing_fps):
    if not metrics:
        return

    # Для Hailo Apps надёжнее использовать итоговый processing_fps,
    # потому что строка FPS часто выводится только в конце.
    if processing_fps and not math.isnan(processing_fps):
        for row in metrics:
            row["fps"] = processing_fps
            row["latency_ms"] = 1000.0 / processing_fps if processing_fps > 0 else math.nan


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

def plot_fps_comparison(df, output_path):
        """
        Строит график двух FPS:
        Source video FPS — FPS исходного видео.
        Processing FPS — скорость обработки Hailo.
        """
        if df.empty:
            return

        plt.figure(figsize=(10, 5))

        if "processing_fps" in df.columns and not df["processing_fps"].dropna().empty:
            plt.plot(df["time_s"], df["processing_fps"], label="Processing FPS")

        if "source_video_fps" in df.columns and not df["source_video_fps"].dropna().empty:
            plt.plot(df["time_s"], df["source_video_fps"], label="Source video FPS")

        plt.title("Source video FPS vs Processing FPS")
        plt.xlabel("Time, s")
        plt.ylabel("FPS")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
def main():
    parser = argparse.ArgumentParser(
        description="Run YOLO object detection on Hailo 8 and collect benchmark metrics."
    )

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
        help="Manual average CPU power in watts"
    )

    parser.add_argument(
        "--gpu-power-watts",
        type=float,
        default=None,
        help="Manual average Hailo/GPU power in watts"
    )

    args = parser.parse_args()

    input_video = Path(args.input)
    output_dir = Path(args.output)
    hailo_app = Path(args.hailo_app)

    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    if not hailo_app.exists():
        raise FileNotFoundError(f"Hailo object_detection.py not found: {hailo_app}")

    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
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

    print("\nЗапускаю Hailo YOLO detection:")
    print(" ".join(command))
    print()

    metrics = []
    fps_values = []

    start_time = time.time()
    stop_event = threading.Event()

    process = subprocess.Popen(
    command,
    cwd=str(hailo_app.parent),
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
    )

    monitor_thread = threading.Thread(
        target=monitor_system,
        args=(metrics, stop_event, start_time, args.sample_interval, process.pid),
        daemon=True
    )
    monitor_thread.start()

    output_thread = threading.Thread(
        target=read_process_output,
        args=(process, fps_values),
        daemon=True
    )
    output_thread.start()

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

    frame_count, source_video_fps = get_video_info(input_video)

    if return_code == 0 and frame_count:
        processed_frames = frame_count
    else:
        processed_frames = 0

    if processed_frames > 0 and total_time > 0:
        processing_fps = processed_frames / total_time
        avg_latency_ms = 1000.0 / processing_fps if processing_fps > 0 else math.nan
    else:
        processing_fps = math.nan
        avg_latency_ms = math.nan

    fill_fps_and_latency(metrics, fps_values, start_time, processing_fps)

    for row in metrics:
        if args.cpu_power_watts is not None:
            row["cpu_power_w"] = args.cpu_power_watts

        if args.gpu_power_watts is not None:
            row["gpu_power_w"] = args.gpu_power_watts

    for row in metrics:
        row["source_video_fps"] = source_video_fps
        row["processing_fps"] = row["fps"]

    df = pd.DataFrame(metrics)

    avg_cpu_usage = df["cpu_usage_percent"].mean() if not df.empty else math.nan
    avg_ram_used_mb = df["ram_used_mb"].mean() if not df.empty else math.nan
    max_ram_used_mb = df["ram_used_mb"].max() if not df.empty else math.nan

    avg_cpu_power = args.cpu_power_watts if args.cpu_power_watts is not None else math.nan
    avg_gpu_power = args.gpu_power_watts if args.gpu_power_watts is not None else math.nan

    if processed_frames > 0 and args.cpu_power_watts is not None and args.gpu_power_watts is not None:
        total_power_w = args.cpu_power_watts + args.gpu_power_watts
        energy_per_frame_j = total_power_w * total_time / processed_frames
    else:
        energy_per_frame_j = math.nan

    csv_path = output_dir / "hailo_metrics.csv"
    summary_path = output_dir / "hailo_summary.txt"

    df.to_csv(csv_path, index=False)

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
    "ram_used_mb",
    "RAM used during Hailo YOLO inference",
    "RAM used, MB",
    output_dir / "ram_used_mb.png"
    )   

    plot_metric(
    df,
    "time_s",
    "processing_fps",
    "Processing FPS during Hailo YOLO inference",
    "Processing FPS",
    output_dir / "processing_fps.png"
    )

    plot_metric(
        df,
        "time_s",
        "source_video_fps",
        "Source video FPS",
        "Source video FPS",
        output_dir / "source_video_fps.png"
    )

    plot_fps_comparison(
        df,
        output_dir / "fps_comparison.png"
    )

    plot_metric(
        df,
        "time_s",
        "latency_ms",
        "Latency during Hailo YOLO inference",
        "Latency, ms",
        output_dir / "latency.png"
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
        "Hailo accelerator power during YOLO inference",
        "Hailo/GPU power, W",
        output_dir / "gpu_power.png"
    )



    


    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Hailo 8 YOLO benchmark summary\n")
        f.write("--------------------------------\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Input video: {input_video}\n")
        f.write(f"Output directory: {output_dir}\n")
        f.write(f"Return code: {return_code}\n")
        f.write(f"Total time, s: {total_time:.3f}\n")
        f.write(f"Frame count in video: {frame_count}\n")
        f.write(f"Processed frames: {processed_frames}\n")
        f.write(f"Source video FPS: {source_video_fps:.3f}\n")
        f.write(f"Processing FPS: {processing_fps:.3f}\n")
        f.write(f"Average latency, ms: {avg_latency_ms:.3f}\n")
        f.write(f"Average CPU usage, %: {avg_cpu_usage:.3f}\n")
        f.write(f"Average RAM used, MB: {avg_ram_used_mb:.3f}\n")
        f.write(f"Max RAM used, MB: {max_ram_used_mb:.3f}\n")
        f.write(f"Average CPU power, W: {avg_cpu_power}\n")
        f.write(f"Average GPU power, W: {avg_gpu_power}\n")
        f.write(f"energy_per_frame_j: {energy_per_frame_j}\n")

    print("\nГотово.")
    print(f"CSV с метриками: {csv_path}")
    print(f"Summary: {summary_path}")
    print(f"Графики сохранены в: {output_dir}")
    print()
    print(f"Processed frames: {processed_frames}")
    print(f"Source video FPS: {source_video_fps:.3f}")
    print(f"Processing FPS: {processing_fps:.3f}")
    print(f"Average latency, ms: {avg_latency_ms:.3f}")
    print(f"Average CPU usage, %: {avg_cpu_usage:.3f}")
    print(f"Average RAM used, MB: {avg_ram_used_mb:.3f}")
    print(f"Max RAM used, MB: {max_ram_used_mb:.3f}")
    print(f"Average CPU power: {avg_cpu_power}")
    print(f"Average GPU power: {avg_gpu_power}")
    print(f"energy_per_frame_j: {energy_per_frame_j}")


if __name__ == "__main__":
    main()