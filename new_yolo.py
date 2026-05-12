import os
HOME = os.path.expanduser("~")

from roboflow import Roboflow
rf = Roboflow(api_key="5HJFkhzx777bLHpOHCdo")
project = rf.workspace("object-detection-yolo-bp").project("my-first-project-iwxqd")
version = project.version(4)
dataset = version.download("yolov5")

project.version(4).deploy(model_type="yolov8", model_path=f"{HOME}/runs/detect/train4/")
