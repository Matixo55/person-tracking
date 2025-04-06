import glob
import json
import multiprocessing
import os
import time
from collections import deque

import cv2
import numpy as np
import torch
from tqdm import tqdm

from models.models import Darknet
from utils.general import (
    check_img_size, non_max_suppression, scale_coords)

# Define YOLOr model versions
models = ["yolor_p6.pt", "yolor_w6.pt", "yolor_csp.pt", "yolor_s.pt"]
excluded_models = []

models = [model for model in models if model not in excluded_models]
models = ["yolor_p6.pt"]  # Only using p6 model for evaluation

# Hardcoded parameters
resolution_width = 1920
resolution_height = 1088
conf_thres = 0.25
iou_thres = 0.45
device = '0'  # CUDA device
img_size = [resolution_width, resolution_height]  # Input resolution
half = True  # Use FP16 half-precision inference
agnostic_nms = False
max_det = 1000
cfg = 'cfg/yolor_p6.cfg'
names_file = 'data/coco.names'
classes = [0, 2, 3, 5, 7]  # Same class filter as YOLOv6 implementation


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


def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)


class YOLOrEvaluator:
    def __init__(self, model_path, img_size, device='0' , half=True):
        self.model_path = model_path
        self.device = device
        self.img_size = img_size
        self.half = half
        self.model_name = os.path.splitext(os.path.basename(model_path))[0]
        self.cfg = cfg

        # Init model
        cuda = self.device != 'cpu' and torch.cuda.is_available()
        self.device = torch.device(f'cuda:{device}' if cuda else 'cpu')

        # Initialize model
        self.model = Darknet(self.cfg).to(self.device)

        # Load weights
        try:
            ckpt = torch.load(model_path, map_location=self.device)  # load checkpoint
            self.model.load_state_dict(ckpt['model'])
        except:
            # Alternative loading method
            from utils.general import load_darknet_weights
            load_darknet_weights(self.model, model_path)

        self.stride = max(int(self.model.stride.max()), 32) if hasattr(self.model, 'stride') else 32
        self.img_size = check_img_size(self.img_size[0], s=self.stride)  # check image size

        # Half precision
        if self.half & (self.device.type != 'cpu'):
            self.model.half()
        else:
            self.model.float()
            self.half = False

        self.model.eval()  # Set in evaluation mode

        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.img_size, self.img_size).to(self.device).type_as(
                next(self.model.parameters())))  # warmup

        # Load class names
        self.names = load_classes(names_file)
        self.colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]

    def process_image(self, img_src):
        '''Process image before image inference, similar to YOLOv6.'''
        from utils.datasets import letterbox

        image = letterbox(img_src, self.img_size, stride=self.stride)[0]

        # Convert
        image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        image = np.ascontiguousarray(image)
        image = torch.from_numpy(image).to(self.device)
        image = image.half() if self.half else image.float()  # uint8 to fp16/32
        image /= 255.0  # 0 - 255 to 0.0 - 1.0

        if image.ndimension() == 3:
            image = image.unsqueeze(0)

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
    def box_convert(x):
        # Convert boxes with shape [n, 4] from [x1, y1, x2, y2] to [x, y, w, h] where x1y1=top-left, x2y2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
        y[:, 2] = x[:, 2] - x[:, 0]  # width
        y[:, 3] = x[:, 3] - x[:, 1]  # height
        return y

    @staticmethod
    def generate_colors(i, bgr=False):
        """Generate color for class i"""
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        palette = []
        for iter in hex:
            h = '#' + iter
            palette.append(tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4)))
        num = len(palette)
        color = palette[int(i) % num]
        return (color[2], color[1], color[0]) if bgr else color

    def process_video(self, video_path, annotation_path, conf_thres=0.25, iou_thres=0.45,
                      agnostic_nms=False, max_det=1000):
        """Process video with annotations and generate comparison video, matching YOLOv6 processing"""

        # Load annotations
        with open(annotation_path, 'r') as f:
            annotations = json.load(f)

        # Open video
        cap = cv2.VideoCapture(video_path)
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Get video name
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        # Create output directory for this video (same structure as YOLOv6)
        output_dir = f"../output/{video_name}"
        os.makedirs(output_dir, exist_ok=True)

        # Set width and height
        width, height = resolution_width, resolution_height
        model_name = self.model_name

        print(f"Processing {model_name} on {video_name}...")

        # Track metrics (same as YOLOv6)
        total_annotations = 0
        detected_annotations = 0
        frame_count = 0
        fps_calculator = CalcFPS()

        # Store frames for writing to video after processing
        frames = []

        # Process frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame to target resolution
            frame = cv2.resize(frame, (width, height))

            # Get annotations for this frame (same format as YOLOv6)
            frame_annotations = [entity for entity in annotations.get("entities", [])
                                 if entity.get("blob", {}).get("frame_idx") == frame_count]

            # Skip frames without annotations if needed (same as YOLOv6)
            if not frame_annotations:
                frame_count += 1
                continue

            # Prepare image for YOLOr
            img, img_src = self.process_image(frame)

            # Inference
            t1 = time.time()
            with torch.no_grad():
                pred_results = self.model(img, augment=False)[0]
            t2 = time.time()

            # Apply NMS
            det = non_max_suppression(
                pred_results, conf_thres, iou_thres, classes=classes,
                agnostic=agnostic_nms, max_det=max_det
            )[0]

            # Process predictions (same format as YOLOv6)
            predictions = []
            if len(det):
                # Rescale boxes from img_size to frame size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()

                # Convert to numpy array format (same as YOLOv6)
                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = [int(x) for x in xyxy]
                    predictions.append(np.array([x1, y1, x2, y2, conf.cpu().numpy(), cls.cpu().numpy()]))

            has_annotations = len(frame_annotations) > 0
            frame_detected = 0

            # Draw annotations (red by default) - same as YOLOv6
            for annotation in frame_annotations:
                bb = annotation.get("bb", [])
                if bb:
                    # Scale bounding box coordinates to match the resized frame
                    scale_x = width / orig_width
                    scale_y = height / orig_height

                    x = int(bb[0] * scale_x)
                    y = int(bb[1] * scale_y)
                    w = int(bb[2] * scale_x)
                    h = int(bb[3] * scale_y)

                    # Red by default, green if detected
                    box_color = (0, 0, 255)  # Red by default (BGR format)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
                    label = f"GT:{annotation.get('type', '')}"
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

                    # Check if this annotation is detected by YOLO
                    detected = False
                    for pred in predictions:
                        pred_box = pred[:4]
                        anno_box_xyxy = [x, y, x + w, y + h]
                        iou = calculate_iou(pred_box, anno_box_xyxy)
                        if iou > 0.3:  # Same threshold as YOLOv6
                            detected = True
                            break

                    if detected:
                        frame_detected += 1

                    # Draw detection status and update box color if detected
                    status = "DETECTED" if detected else "MISSED"
                    status_color = (0, 255, 0) if detected else (0, 0, 255)

                    # Change box color to green if detected
                    if detected:
                        box_color = (0, 255, 0)  # Green (BGR format)
                        # Redraw the box with green color
                        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
                        # Redraw the label
                        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

                    cv2.putText(frame, status, (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)

            # Update metrics if frame has annotations
            if has_annotations:
                total_annotations += len(frame_annotations)
                detected_annotations += frame_detected

            # Draw YOLO detections (blue boxes)
            for pred in predictions:
                box = pred[:4].astype(int)
                conf = pred[4]
                x1, y1, x2, y2 = box

                # Make all YOLO boxes blue (same as YOLOv6)
                color = (255, 0, 0)  # Blue in BGR format
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"YOLO: {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Draw frame info and metrics (same as YOLOv6)
            accuracy = detected_annotations / total_annotations if total_annotations > 0 else 0
            inference_time = (t2 - t1) * 1000  # ms

            # Update FPS calculator
            fps_calculator.update(1.0 / (t2 - t1))
            avg_fps = fps_calculator.accumulate()

            # Draw info on frame (same positions as YOLOv6)
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Accuracy: {accuracy:.2f} ({detected_annotations}/{total_annotations})",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Model: {model_name}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Video: {video_name}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Inference: {inference_time:.1f}ms", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)

            # Store frame
            frames.append(frame)
            frame_count += 1

        # Calculate final accuracy
        final_accuracy = detected_annotations / total_annotations if total_annotations > 0 else 0

        # Create output video with accuracy in the filename (same format as YOLOv6)
        output_video_path = os.path.join(output_dir, f"{final_accuracy:.4f}_{model_name}_{width}x{height}.mp4")

        # Write all stored frames to the video
        if frames:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            for frame in frames:
                out.write(frame)
            out.release()

        # Release resources
        cap.release()
        return video_name, model_name, final_accuracy


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


def process_video_for_model(args):
    """Process video for a specific model, same as YOLOv6"""
    video_path, model_path, MODE = args

    if MODE == "personpath22":
        annotation_path = f"../dataset/personpath22/annotation/anno_visible_2022/{os.path.basename(video_path)}.json"
    else:
        annotation_path = f"../dataset/DETRAC_Upload/labels/annotations/{os.path.basename(video_path)}.json"

    # Skip if annotation doesn't exist
    evaluator = YOLOrEvaluator(model_path)
    return evaluator.process_video(video_path, annotation_path)


def process_all_videos(MODE):
    """Process all videos, using the same datasets as YOLOv6"""
    # Create results directory
    os.makedirs("../results", exist_ok=True)

    # Get all MP4 files in the dataset directory (same paths as YOLOv6)
    if MODE == "personpath22":
        video_paths = glob.glob("../dataset/personpath22/raw_data/*.mp4")
    else:
        video_paths = glob.glob("../dataset/DETRAC_Upload/videos/*.mp4")

    print(f"Found {len(video_paths)} videos to process")

    # Set up multiprocessing (same as YOLOv6)
    torch.multiprocessing.set_start_method("spawn", force=True)

    with open(f"../results/all_results_{MODE}.txt", "a+") as f_all:
        for video_path in tqdm(video_paths):
            video_name = os.path.basename(video_path)
            f_all.write(f"\n{video_name}:\n")
            print(f"\nProcessing video: {video_name}")

            # Create tasks for current video only
            video_tasks = [(video_path, model, MODE) for model in models]

            # Process each model for this video in parallel
            with multiprocessing.Pool(2) as pool:
                video_results = pool.map(process_video_for_model, video_tasks)

            for _, model_name, accuracy in video_results:
                f_all.write(f"  {model_name}: {accuracy:.4f}\n")


if __name__ == "__main__":
    MODE = "personpath22"  # DETRAC or personpath22, same as YOLOv6
    process_all_videos(MODE)