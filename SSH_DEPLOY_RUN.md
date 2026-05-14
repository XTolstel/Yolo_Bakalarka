# Запуск проекта на «пустом» ноутбуке по SSH

Ниже — пошаговая инструкция, если на ноутбуке **вообще ничего нет**.

## Краткий запуск (5 команд)

На ноутбуке после SSH выполни:

```bash
git clone <URL_ВАШЕГО_РЕПО> && cd Yolo_Bakalarka
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip ultralytics opencv-python matplotlib psutil numpy nvidia-ml-py
YOLO_DEBUG=1 YOLO_MODEL_PATH=/home/<USER>/models/best.pt YOLO_VIDEO_PATH=Video/video_yolo_cars.mp4 python3 -c "from Yolo_test import object_detection; print(object_detection.detection())"
ls -lh video_yolo.mp4 pipeline_latency_plot.png
```

В консоли получишь словарь со всеми метриками, а в папке появятся:
- `video_yolo.mp4`
- `pipeline_latency_plot.png`

## 1) Установи базовые пакеты на ноутбуке

Подключись к ноутбуку:

```bash
ssh <USER>@<LAPTOP_IP>
```

Установи Python, git и venv (Ubuntu/Debian):

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip ffmpeg
```

## 2) Скопируй проект на ноутбук

### Вариант A (рекомендуется): через GitHub

На ноутбуке:

```bash
git clone <URL_ВАШЕГО_РЕПО>
cd Yolo_Bakalarka
```

### Вариант B: напрямую с твоего ПК (если нет GitHub)

Запусти эту команду **на твоём основном ПК**, не на ноутбуке:

```bash
scp -r /путь/к/Yolo_Bakalarka <USER>@<LAPTOP_IP>:~/
```

Потом зайди по SSH и открой папку:

```bash
cd ~/Yolo_Bakalarka
```

## 3) Создай окружение и поставь зависимости

Внутри проекта на ноутбуке:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install ultralytics opencv-python matplotlib
```

## 4) Подготовь входные данные

Убедись, что существуют:

- видео: `Video/video_yolo_cars.mp4`
- веса модели `.pt` (путь в коде должен совпадать с фактическим на ноутбуке)

Если весов нет на ноутбуке, скопируй:

```bash
scp /локальный/путь/best.pt <USER>@<LAPTOP_IP>:/home/<USER>/models/best.pt
```

И обнови путь в `Yolo_test.py`, если нужно.

## 5) Запусти программу

```bash
YOLO_DEBUG=1 YOLO_MODEL_PATH=/home/<USER>/models/best.pt YOLO_VIDEO_PATH=Video/video_yolo_cars.mp4 python3 -c "from Yolo_test import object_detection; print(object_detection.detection())"
```

Результат:

- выходное видео: `video_yolo.mp4`
- график latency: `pipeline_latency_plot.png`

## 6) Запуск в фоне (чтобы не зависеть от SSH-сессии)

```bash
nohup .venv/bin/python -c "from Yolo_test import object_detection; print(object_detection.detection())" > run.log 2>&1 &
tail -f run.log
```

## 7) Скачать результаты обратно на основной ПК

На основном ПК:

```bash
scp <USER>@<LAPTOP_IP>:~/Yolo_Bakalarka/video_yolo.mp4 ./
scp <USER>@<LAPTOP_IP>:~/Yolo_Bakalarka/pipeline_latency_plot.png ./
```


## Важно для запуска objects.py

`objects.py` использует `pynvml` (пакет `nvidia-ml-py`).
Если его нет, будет ошибка `ModuleNotFoundError: No module named 'pynvml'`.

Установи:

```bash
pip install nvidia-ml-py
```
