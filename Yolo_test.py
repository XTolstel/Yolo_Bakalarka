from ultralytics import YOLO
import cv2
import time

class object_detection:
    def detection():
        # Загрузка модели YOLOv8
        #net = YOLO("/home/oleg/runs/detect/train6/weights/best.pt") #model pre auta
        net = YOLO("/home/oleg/runs/detect/train9/weights/best.pt")
        #net = YOLO("yolov8x.pt")  # Наприклад, використання YOLOv8m

        #net = YOLO("/home/oleh/runs/detect/train9/weights/best.pt") # location for notebook

        # Загрузка видео
        video_path = "Video/video_yolo_cars.mp4"
        cap = cv2.VideoCapture(video_path)

        # Получение параметров исходного видео
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Создание объекта для записи видео
        output_path = "video_yolo.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Кодек для сжатия
        out_yolo = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        frame_rate = 1  # Какой по счету берем кадр
        frame_count = 0
        # Обработка кадров
        total_frames = 0
        total_time = 0
        frame_latencies_ms = []
        while True:
            ret, frame = cap.read()  # Считываем кадр с видео
            if not ret:
                break
            frame_count += 1
            if frame_count % frame_rate != 0:
                continue
            start_time = time.time()

            # Предсказание с использованием YOLOv8
            results = net.predict(source=frame,verbose=False)

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

            # Запись кадра в видео
            out_yolo.write(frame)
            print(f"Frame of video: {frame_count:.1f}")

            end_time = time.time()
            frame_time_s = end_time - start_time
            total_time += frame_time_s
            frame_latencies_ms.append(frame_time_s * 1000.0)
            total_frames += 1

        cap.release()
        out_yolo.release()

        avg_time_per_frame = total_time / total_frames if total_frames else 0
        return {
            "avg_time_per_frame_s": avg_time_per_frame,
            "total_frames": total_frames,
            "total_processing_time_s": total_time,
            "latencies_ms": frame_latencies_ms,
        }
