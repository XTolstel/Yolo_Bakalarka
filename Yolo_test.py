from ultralytics import YOLO
import cv2


class object_detection:
    @staticmethod
    def detection(
        model_path: str = "/home/oleg/runs/detect/train9/weights/best.pt",
        video_path: str = "Video/video_yolo_cars.mp4",
        output_path: str = "video_yolo.mp4",
        confidence_threshold: float = 0.5,
    ) -> dict:
        model = YOLO(model_path)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise RuntimeError(f"Не удалось открыть видео: {video_path}")

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        processed_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(source=frame, verbose=False)

            for result in results:
                for box in result.boxes:
                    confidence = float(box.conf[0].cpu().numpy())
                    if confidence < confidence_threshold:
                        continue

                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cv2.rectangle(
                        frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (0, 255, 0),
                        2,
                    )

            writer.write(frame)
            print(f"Готово. Обработано кадров: {processed_frames}")
            processed_frames += 1
            

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
