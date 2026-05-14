from ultralytics import YOLO
import cv2
import time
import matplotlib.pyplot as plt

class object_detection:
    def detection():
        # Загрузка модели YOLOv8
        #net = YOLO("/home/oleg/runs/detect/train6/weights/best.pt") #model pre auta
        net = YOLO("/home/oleg/runs/detect/train9/weights/best.pt")
        #net = YOLO("yolov8x.pt")  # Наприклад, використання YOLOv8m

        #net = YOLO("/home/oleh/runs/detect/train9/weights/best.pt") # location for notebook

        # Загрузка видео
        #video_path = "Video/video_yolo_cars.mp4"
        video_path = "Video/Video_test3.mp4"
        cap = cv2.VideoCapture(video_path)

        # Получение параметров исходного видео
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Создание объекта для записи видео
        output_path = "video_yolo.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Кодек для сжатия
        out_yolo = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        frame_rate = 5  # Какой по счету берем кадр
        frame_count = 0
        # Обработка кадров
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
            ret, frame = cap.read()  # Считываем кадр с видео
            t1 = time.perf_counter()
            if pipeline_start_time is None:
                pipeline_start_time = t0
            if not ret:
                break
            frame_count += 1
            if frame_count % frame_rate != 0:
                continue
            start_time = time.perf_counter()

            # Предсказание с использованием YOLOv8
            t5 = time.perf_counter()  # Inference Start
            results = net.predict(source=frame,verbose=False)
            t6 = time.perf_counter()  # Inference End

            # Обработка результатов предсказания
            for result in results:# перебирает слои 
                for box in result.boxes:#перебирает объекты на слоях 
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Координаты
                    confidence = box.conf[0].cpu().numpy()  # Уверенность
                    class_id = int(box.cls[0].cpu().numpy())  # Класс объекта
                    if confidence > 0.5:
                        # Рисуем прямоугольник и текст
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        label = f"{net.names[class_id]}: {confidence:.2f}"
                        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

            t7 = time.perf_counter()  # Postprocess End

            # Запись кадра в видео
            out_yolo.write(frame)
            t8 = time.perf_counter()  # Render / Output
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

        avg_time_per_frame = total_time / total_frames if total_frames else 0

        if total_frames:
            plt.figure(figsize=(12, 6))
            plot_latency_limit_ms = 100.0
            for stage_name, values in stage_latencies_ms.items():
                if stage_name == "end_to_end":
                    continue  # raw metric is kept, but not shown on chart by request
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
        }
