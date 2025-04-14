import os
import urllib.request

urls = [
    "https://github.com/WongKinYiu/ScaledYOLOv4/releases/download/weights/yolov4-p7.pt",
    "https://github.com/WongKinYiu/ScaledYOLOv4/releases/download/weights/yolov4-p6_.pt",
    "https://github.com/WongKinYiu/ScaledYOLOv4/releases/download/weights/yolov4-p6.pt",
    "https://github.com/WongKinYiu/ScaledYOLOv4/releases/download/weights/yolov4-p5_.pt",
    "https://github.com/WongKinYiu/ScaledYOLOv4/releases/download/weights/yolov4-p5.pt",
]

for url in urls:
    filename = os.path.basename(url)
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
    else:
        print(f"{filename} already exists.")
