import glob
import json
import math
import os
import os.path as osp
import time
import psutil
import GPUtil
import torch
import numpy as np
import cv2
from PIL import ImageFont
from tqdm import tqdm
from collections import deque

from yolov6.data.data_augment import letterbox
from yolov6.layers.common import DetectBackend
from yolov6.utils.events import LOGGER
from yolov6.utils.nms import non_max_suppression

# Define YOLOv6 model versions
models = ["yolov6s.pt", "yolov6n.pt", "yolov6m.pt", "yolov6l.pt", "yolov6n6.pt", "yolov6s6.pt", "yolov6m6.pt",
          "yolov6l6.pt"]
EXCLUDED_DATASETLS = []

models = [model for model in models if model not in EXCLUDED_DATASETLS]
models = []

# Multiple of 64
WIDTH, HEIGHT = 1920, 1088


def get_gpu_metrics():
    gpus = GPUtil.getGPUs()
    gpu = gpus[0]  # Get the first GPU
    return {
        "gpu_name": gpu.name,
        "gpu_load": gpu.load * 100,  # Convert to percentage
        "gpu_memory_used": gpu.memoryUsed,
        "gpu_memory_total": gpu.memoryTotal
    }


# Define IoU calculation function
def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection

    return intersection / union if union > 0 else 0


class CalcFPS:
    def __init__(self, nsamples: int = 50):
        self.framerate = deque(maxlen=nsamples)

    def update(self, duration: float):
        self.framerate.append(duration)

    def accumulate(self):
        if len(self.framerate) > 1:
            return np.average(self.framerate)
        else:
            return 0.0


class YOLOv6Benchmark:
    def __init__(self, model_path, device, img_size, half):
        self.model_path = model_path
        self.device = device
        self.img_size = img_size
        self.half = half
        self.model_name = os.path.splitext(os.path.basename(model_path))[0]

        # Init model
        cuda = self.device != 'cpu' and torch.cuda.is_available()
        self.device = torch.device(f'cuda:{device}' if cuda else 'cpu')
        self.model = DetectBackend(model_path, device=self.device)
        self.stride = self.model.stride
        self.img_size = self.check_img_size(self.img_size, s=self.stride)  # check image size

        # Switch model to deploy status
        self.model_switch(self.model.model, self.img_size)

        # Half precision
        if self.half & (self.device.type != 'cpu'):
            self.model.model.half()
        else:
            self.model.model.float()
            self.half = False

        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, *self.img_size).to(self.device).type_as(
                next(self.model.model.parameters())))  # warmup

        # Font setup
        self.font_check()

    def model_switch(self, model, img_size):
        ''' Model switch to deploy status '''
        from yolov6.layers.common import RepVGGBlock
        for layer in model.modules():
            if isinstance(layer, RepVGGBlock):
                layer.switch_to_deploy()
            elif isinstance(layer, torch.nn.Upsample) and not hasattr(layer, 'recompute_scale_factor'):
                layer.recompute_scale_factor = None  # torch 1.11.0 compatibility

        LOGGER.info("Switch model to deploy modality.")

    def check_img_size(self, img_size, s=32, floor=0):
        """Make sure image size is a multiple of stride s in each dimension, and return a new shape list of image."""
        if isinstance(img_size, int):
            new_size = max(self.make_divisible(img_size, int(s)), floor)
        elif isinstance(img_size, list):
            new_size = [max(self.make_divisible(x, int(s)), floor) for x in img_size]
        else:
            raise Exception(f"Unsupported type of img_size: {type(img_size)}")

        if new_size != img_size:
            print(f'WARNING: --img-size {img_size} must be multiple of max stride {s}, updating to {new_size}')
        return new_size if isinstance(img_size, list) else [new_size] * 2

    def make_divisible(self, x, divisor):
        # Upward revision the value x to make it evenly divisible by the divisor.
        return math.ceil(x / divisor) * divisor

    @staticmethod
    def process_image(img_src, img_size, stride, half):
        '''Process image before image inference.'''
        image = letterbox(img_src, img_size, stride=stride)[0]
        # Convert
        image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        image = torch.from_numpy(np.ascontiguousarray(image))
        image = image.half() if half else image.float()  # uint8 to fp16/32
        image /= 255  # 0 - 255 to 0.0 - 1.0

        return image, img_src

    @staticmethod
    def rescale(ori_shape, boxes, target_shape):
        '''Rescale the output to the original image shape'''
        ratio = min(ori_shape[0] / target_shape[0], ori_shape[1] / target_shape[1])
        padding = (ori_shape[1] - target_shape[1] * ratio) / 2, (ori_shape[0] - target_shape[0] * ratio) / 2

        boxes[:, [0, 2]] -= padding[0]
        boxes[:, [1, 3]] -= padding[1]
        boxes[:, :4] /= ratio

        boxes[:, 0].clamp_(0, target_shape[1])  # x1
        boxes[:, 1].clamp_(0, target_shape[0])  # y1
        boxes[:, 2].clamp_(0, target_shape[1])  # x2
        boxes[:, 3].clamp_(0, target_shape[0])  # y2

        return boxes

    @staticmethod
    def font_check(font='./yolov6/utils/Arial.ttf', size=10):
        """Return a PIL TrueType Font, downloading to CONFIG_DIR if necessary"""
        assert osp.exists(font), f'font path not exists: {font}'
        try:
            return ImageFont.truetype(str(font) if font.exists() else font.name, size)
        except Exception as e:  # download if missing
            return ImageFont.truetype(str(font), size)

    def process_video(self, args):
        """Process a video with benchmarking metrics"""
        video_path, model_version, DATASET, benchmark_mode = args

        # Extract video name without extension
        video_name = os.path.basename(video_path)
        video_name_no_ext = os.path.splitext(video_name)[0]

        # Initialize benchmark metrics
        benchmark_metrics = {
            "model": self.model_name,
            "video": video_name,
            "total_time": 0,
            "frames_processed": 0,
            "fps": 0,
            "max_memory_usage_mb": 0,
            "avg_memory_usage_mb": 0,
            "max_gpu_memory_mb": 0,
            "avg_gpu_memory_mb": 0,
            "max_gpu_load": 0,
            "avg_gpu_load": 0,
            "accuracy": 0
        }

        # Data containers for metrics
        memory_samples = []
        gpu_memory_samples = []
        gpu_load_samples = []


        # Create output directory
        output_dir = f"output/{video_name_no_ext}"
        os.makedirs(output_dir, exist_ok=True)

        # Track metrics
        total_annotations = 0
        detected_annotations = 0
        frame_count = 0


        # Open video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        benchmark_metrics["video_fps"] = fps
        benchmark_metrics["video_total_frames"] = total_frames

        # Start tracking GPU and memory metrics
        gpu_metrics = get_gpu_metrics()
        start_gpu_memory_used = gpu_metrics["gpu_memory_used"]
        benchmark_metrics["start_gpu_memory_used"] = start_gpu_memory_used

        start_gpu_load = gpu_metrics["gpu_load"]
        benchmark_metrics["start_gpu_load"] = start_gpu_load

        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / (1024 * 1024)  # Convert to MB
        benchmark_metrics["start_memory_usage_mb"] = start_memory

        # Process frames
        i = 0
        while cap.isOpened():
            i += 1
            if i % 1000 == 0:
                print(f"Processing frame {i}/{total_frames}...")

            if i % 10 == 0:
                current_memory = process.memory_info().rss / (1024 * 1024)
                memory_samples.append(current_memory)
                gpu_metrics = get_gpu_metrics()
                gpu_memory_samples.append(gpu_metrics["gpu_memory_used"])
                gpu_load_samples.append(gpu_metrics["gpu_load"])

            ret, frame = cap.read()
            if not ret:
                break

            frame_resized = cv2.resize(frame, (self.img_size[0], self.img_size[1]))

            # Prepare image for YOLOv6
            img, _ = self.process_image(frame_resized, self.img_size, self.stride, self.half)
            img = img.to(self.device)
            if len(img.shape) == 3:
                img = img[None]  # expand for batch dim

            # Start timing for benchmarking
            if i % 10 == 0:
                frame_start_time = time.time()

            # Inference
            with torch.no_grad():
                pred_results = self.model(img)

            # Apply NMS
            det = non_max_suppression(pred_results, 0.25, 0.45, classes=[0, 1, 2, 3], agnostic=False, max_det=1000)[0]

            # Record time for benchmarking
            if i % 10 == 0:
                frame_process_time = time.time() - frame_start_time
                benchmark_metrics["total_time"] += frame_process_time
                benchmark_metrics["frames_processed"] += 1

            if not benchmark_mode:
                predictions = []
                if len(det):
                    det[:, :4] = self.rescale(img.shape[2:], det[:, :4], frame_resized.shape).round()

                    for *xyxy, conf, cls in det:
                        x1, y1, x2, y2 = [int(x) for x in xyxy]
                        predictions.append(np.array([x1, y1, x2, y2, conf.cpu().numpy(), cls.cpu().numpy()]))

            frame_count += 1

        # Calculate metrics
        memory_samples = np.array(memory_samples)
        benchmark_metrics["max_memory_usage_mb"] = np.max(memory_samples)
        benchmark_metrics["avg_memory_usage_mb"] = np.mean(memory_samples)

        # Calculate GPU metrics
        gpu_memory_samples = np.array(gpu_memory_samples)
        gpu_load_samples = np.array(gpu_load_samples)

        benchmark_metrics["max_gpu_memory_mb"] = np.max(gpu_memory_samples)
        benchmark_metrics["avg_gpu_memory_mb"] = np.mean(gpu_memory_samples)
        benchmark_metrics["max_gpu_load"] = np.max(gpu_load_samples)
        benchmark_metrics["avg_gpu_load"] = np.mean(gpu_load_samples)

        # Calculate frames per second
        if benchmark_metrics["frames_processed"] > 0:
            benchmark_metrics["fps"] = benchmark_metrics["frames_processed"] / benchmark_metrics["total_time"]

        # Calculate final accuracy
        final_accuracy = detected_annotations / total_annotations if total_annotations > 0 else -1
        benchmark_metrics["accuracy"] = final_accuracy

        print(benchmark_metrics)

        # Release resources
        cap.release()

        return video_name, self.model_name, benchmark_metrics


def process_all_videos(DATASET):
    benchmark_mode = False
    if DATASET == "benchmark":
        benchmark_mode = True
        video_paths = glob.glob("../dataset/camera/*.mp4")

    # Set up multiprocessing
    torch.multiprocessing.set_start_method("spawn", force=True)

    # Create results directory
    os.makedirs("results", exist_ok=True)

    # Store all benchmark results
    all_benchmark_results = []

    # Process one video at a time with progress bar
    with open(f"../results/all_results_{DATASET}.txt", "a+") as f_all:
        for video_path in tqdm(video_paths):
            video_name = os.path.basename(video_path)
            f_all.write(f"\n{video_name}:\n")
            print(f"\nProcessing video: {video_name}")

            for model in tqdm(models):
                # Create benchmark instance for this model
                benchmark = YOLOv6Benchmark(f"weights/{model}", device='0', img_size=[WIDTH, HEIGHT], half=False)
                video_name, model_name, metrics = benchmark.process_video((video_path, model, DATASET, benchmark_mode))

                if benchmark_mode:
                    # Store results for sorting
                    all_benchmark_results.append(metrics)

                    # Write CSV-formatted line
                    f_all.write(
                        f"  {model_name}: {metrics['accuracy']:.4f} {metrics['fps']:.2f},{metrics['total_time']:.2f},{metrics['frames_processed']},{metrics['max_memory_usage_mb']:.2f},{metrics['avg_memory_usage_mb']:.2f},{metrics['max_gpu_memory_mb']:.2f},{metrics['avg_gpu_memory_mb']:.2f},{metrics['max_gpu_load']:.2f},{metrics['avg_gpu_load']:.2f},{metrics['start_gpu_memory_used']:.2f},{metrics['start_gpu_load']:.2f},{metrics['start_memory_usage_mb']:.2f}\n")
                else:
                    f_all.write(f"  {model_name}: {metrics['accuracy']:.4f}\n")


if __name__ == "__main__":
    DATASET = "benchmark"  # Options: "DETRAC", "benchmark", "personpath22"
    process_all_videos(DATASET)
