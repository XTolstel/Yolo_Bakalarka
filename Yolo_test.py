from ultralytics import YOLO
import cv2
import time
import os
import matplotlib.pyplot as plt


class object_detection:
    def detection():
        debug_mode = os.getenv("YOLO_DEBUG", "0") == "1"
        model_path = os.getenv("YOLO_MODEL_PATH", "best.pt")
        video_path = os.getenv("YOLO_VIDEO_PATH", "Video/video_yolo_cars.mp4")

        if debug_mode:
            print(f"[YOLO_DEBUG] cwd={os.getcwd()}")
            print(f"[YOLO_DEBUG] model_path={model_path} exists={os.path.exists(model_path)}")
            print(f"[YOLO_DEBUG] video_path={video_path} exists={os.path.exists(video_path)}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}. "
                "Set YOLO_MODEL_PATH to your .pt weights path."
            )
        if not os.path.exists(video_path):
            raise FileNotFoundError(
                f"Video file not found: {video_path}. "
                "Set YOLO_VIDEO_PATH to your input video path."
            )

        net = YOLO(model_path)
        cap = cv2.VideoCapture(video_path)
        if debug_mode:
            print(f"[YOLO_DEBUG] cap_opened={cap.isOpened()}")
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if debug_mode:
            print(f"[YOLO_DEBUG] width={frame_width} height={frame_height} fps={fps}")
        if frame_width <= 0 or frame_height <= 0 or fps <= 0:
            raise RuntimeError(
                f"Invalid video metadata (w={frame_width}, h={frame_height}, fps={fps})."
            )

        output_path = "video_yolo.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_yolo = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        frame_rate = 1
        frame_count = 0
        total_frames = 0
        total_time = 0
        frame_latencies_ms = []
        stage_latencies_ms = {
            "capture_to_decode_start": [],
            "inference": [],
            "postprocess": [],
            "render_output": [],
            "end_to_end": [],
        }
        timeline_s = []
        pipeline_start_time = None

        while True:
            t0 = time.perf_counter()
            ret, frame = cap.read()
            t1 = time.perf_counter()
            if pipeline_start_time is None:
                pipeline_start_time = t0
            if not ret:
                if debug_mode and frame_count == 0:
                    print("[YOLO_DEBUG] First cap.read() returned ret=False")
                break

            frame_count += 1
            if frame_count % frame_rate != 0:
                continue

            start_time = time.perf_counter()
            t5 = time.perf_counter()
            results = net.predict(source=frame, verbose=False)
            t6 = time.perf_counter()

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    if confidence > 0.5:
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        label = f"{net.names[class_id]}: {confidence:.2f}"
                        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

            t7 = time.perf_counter()
            out_yolo.write(frame)
            t8 = time.perf_counter()
            print(f"Frame of video: {frame_count:.1f}")

            end_time = time.perf_counter()
            frame_time_s = end_time - start_time
            total_time += frame_time_s
            frame_latencies_ms.append(frame_time_s * 1000.0)
            stage_latencies_ms["capture_to_decode_start"].append((t1 - t0) * 1000.0)
            stage_latencies_ms["inference"].append((t6 - t5) * 1000.0)
            stage_latencies_ms["postprocess"].append((t7 - t6) * 1000.0)
            stage_latencies_ms["render_output"].append((t8 - t7) * 1000.0)
            stage_latencies_ms["end_to_end"].append((t8 - t0) * 1000.0)
            timeline_s.append(t8 - pipeline_start_time)
            total_frames += 1

        cap.release()
        out_yolo.release()

        if total_frames == 0:
            raise RuntimeError(
                "0 frames were processed. Check model/video paths and video codec support."
            )

        avg_time_per_frame = total_time / total_frames

        plt.figure(figsize=(12, 6))
        plot_latency_limit_ms = 100.0
        for stage_name, values in stage_latencies_ms.items():
            if stage_name == "end_to_end":
                continue
            clipped_values = [value if value <= plot_latency_limit_ms else None for value in values]
            plt.plot(timeline_s, clipped_values, label=stage_name)
        plt.title("YOLO pipeline latency over time")
        plt.xlabel("Time from start (s)")
        plt.ylabel("Latency (ms)")
        plt.ylim(0, plot_latency_limit_ms)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("pipeline_latency_plot.png")
        plt.close()

        return {
            "avg_time_per_frame_s": avg_time_per_frame,
            "total_frames": total_frames,
            "total_processing_time_s": total_time,
            "latencies_ms": frame_latencies_ms,
            "stage_latencies_ms": stage_latencies_ms,
            "timeline_s": timeline_s,
            "latency_plot_path": "pipeline_latency_plot.png",
            "model_path": model_path,
            "video_path": video_path,
        }
