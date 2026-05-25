import os
import re
import time
import subprocess

try:
    import cv2
except Exception:
    cv2 = None


class object_detection:
    @staticmethod
    def _get_video_path_from_deepstream_config(config_path):
        """
        Берёт путь к видео из строки:
        uri=file:///home/jetson/...
        """
        try:
            with open(config_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("uri=file://"):
                        uri = line.split("=", 1)[1].strip()
                        return uri.replace("file://", "")
        except FileNotFoundError:
            return None

        return None

    @staticmethod
    def _count_video_frames(video_path):
        if cv2 is None or not video_path or not os.path.exists(video_path):
            return 0

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0

        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return frames

    @staticmethod
    def _extract_fps_from_deepstream_line(line):
        """
        Пробуем достать FPS из строк DeepStream вида:
        **PERF:  12.34 (12.01)
        """
        if "PERF" not in line:
            return None

        if "PERF:" in line:
            line = line.split("PERF:", 1)[1]

        numbers = re.findall(r"\d+(?:\.\d+)?", line)
        values = []

        for n in numbers:
            value = float(n)
            if 0.1 <= value <= 500:
                values.append(value)

        if not values:
            return None

        return values[0]

    @staticmethod
    def detection():
        home = os.path.expanduser("~")

        deepstream_dir = os.path.join(home, "DeepStream-Yolo")
        config_path = os.path.join(deepstream_dir, "deepstream_app_config.txt")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")

        video_path = object_detection._get_video_path_from_deepstream_config(config_path)
        total_frames = object_detection._count_video_frames(video_path)

        cmd = [
            "stdbuf", "-oL", "-eL",
            "deepstream-app",
            "-c", config_path
        ]

        fps_values = []
        log_path = os.path.join(deepstream_dir, "deepstream_python_wrapper.log")

        print("Starting DeepStream...")
        print("Config:", config_path)
        print("Video:", video_path)
        print("Command:", " ".join(cmd))

        start_time = time.perf_counter()

        with open(log_path, "w") as log_file:
            process = subprocess.Popen(
                cmd,
                cwd=deepstream_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            for line in process.stdout:
                print(line, end="")
                log_file.write(line)

                fps = object_detection._extract_fps_from_deepstream_line(line)
                if fps is not None:
                    fps_values.append(fps)

            return_code = process.wait()

        end_time = time.perf_counter()
        wall_time_s = end_time - start_time

        if return_code != 0:
            raise RuntimeError(
                f"DeepStream finished with error code {return_code}. "
                f"Check log: {log_path}"
            )

        if fps_values:
            avg_fps = sum(fps_values) / len(fps_values)
        elif total_frames > 0 and wall_time_s > 0:
            avg_fps = total_frames / wall_time_s
        else:
            avg_fps = 0

        if total_frames > 0 and avg_fps > 0:
            processing_time_s = total_frames / avg_fps
        else:
            processing_time_s = wall_time_s

        avg_time_per_frame_s = 1 / avg_fps if avg_fps > 0 else 0

        # Это не настоящие per-frame latency, а приближение из FPS DeepStream.
        latencies_ms = [1000 / fps for fps in fps_values if fps > 0]

        return {
            "avg_time_per_frame_s": avg_time_per_frame_s,
            "total_frames": total_frames,
            "total_processing_time_s": processing_time_s,
            "latencies_ms": latencies_ms,
            "stage_latencies_ms": {},
            "timeline_s": [],
            "deepstream_avg_fps": avg_fps,
            "deepstream_wall_time_s": wall_time_s,
            "deepstream_log_path": log_path,
        }