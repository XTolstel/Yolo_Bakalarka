from ultralytics import YOLO
import cv2
import os


class object_detection:
    def detection():
        debug_mode = os.getenv("YOLO_DEBUG", "0") == "1"
        model_path = os.getenv("YOLO_MODEL_PATH", "best.pt")
        video_path = os.getenv("YOLO_VIDEO_PATH", "Video/video_yolo_cars.mp4")
        output_path = os.getenv("YOLO_OUTPUT_PATH", "video_yolo.mp4")

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
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if frame_width <= 0 or frame_height <= 0 or fps <= 0:
            raise RuntimeError(
                f"Invalid video metadata (w={frame_width}, h={frame_height}, fps={fps})."
            )

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_yolo = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        processed_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = net.predict(source=frame, verbose=False)

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())

                    if confidence > 0.5:
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        label = f"{net.names[class_id]}: {confidence:.2f}"
                        cv2.putText(
                            frame,
                            label,
                            (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                        )

            out_yolo.write(frame)
            processed_frames += 1
            if debug_mode:
                print(f"[YOLO_DEBUG] frame={processed_frames}")

        cap.release()
        out_yolo.release()

        if processed_frames == 0:
            raise RuntimeError(
                "0 frames were processed. Check model/video paths and video codec support."
            )

        return {
            "processed_frames": processed_frames,
            "output_path": output_path,
            "model_path": model_path,
            "video_path": video_path,
        }
