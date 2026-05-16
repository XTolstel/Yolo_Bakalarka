from Yolo_test import object_detection


if __name__ == "__main__":
    result = object_detection.detection()
    print(f"Готово. Обработано кадров: {result['processed_frames']}")
    print(f"Видео сохранено: {result['output_path']}")
