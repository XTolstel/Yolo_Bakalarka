import cv2
import torch


class object_detection:
    @staticmethod
    def detection(
        model_name: str = "yolov5n",
        video_path: str = "Video/video_yolo_cars.mp4",
        output_path: str = "video_yolo.mp4",
        confidence_threshold: float = 0.5,
        use_repo_weights: bool = True,
    ) -> dict:
        """
        Детекция объектов с помощью YOLOv5 (через torch.hub), совместимо с Jetson Nano.

        Args:
            model_name: Название модели YOLOv5 (например, yolov5n, yolov5s) или путь к .pt файлу.
            video_path: Путь к входному видео.
            output_path: Путь для сохранения выходного видео.
            confidence_threshold: Минимальный порог уверенности детекции.
            use_repo_weights: Если True — загружает предобученные веса из репозитория ultralytics/yolov5.
                              Если False — model_name трактуется как путь к локальному файлу весов.
        """
        if use_repo_weights:
            model = torch.hub.load("ultralytics/yolov5", model_name, pretrained=True)
        else:
            model = torch.hub.load("ultralytics/yolov5", "custom", path=model_name)

        model.conf = confidence_threshold

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Не удалось открыть видео: {video_path}")

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        class_names = model.names
        processed_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)

            for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
                x1, y1, x2, y2 = map(int, xyxy)
                confidence = float(conf)
                class_id = int(cls)
                label = class_names[class_id] if class_id < len(class_names) else str(class_id)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label_text = f"{label} {confidence:.2f}"
                text_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
                cv2.putText(
                    frame,
                    label_text,
                    (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            writer.write(frame)
            processed_frames += 1
            print(f"Готово. Обработано кадров: {processed_frames}")

        cap.release()
        writer.release()

        return {
            "processed_frames": processed_frames,
            "output_path": output_path,
        }


if __name__ == "__main__":
    result = object_detection.detection()
    print(f"Готово. Обработано кадров: {result['processed_frames']}")
    print(f"Видео сохранено: {result['output_path']}")
