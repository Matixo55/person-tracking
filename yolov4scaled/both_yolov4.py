import glob
import json
import multiprocessing
import os
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import (
    check_img_size, non_max_suppression, scale_coords)
from utils.torch_utils import select_device

# Define YOLOv4 model versions
models = ["yolov4-p5.pt", "yolov4-p6.pt", "yolov4-p7.pt", "yolov4-p5_.pt", "yolov4-p6_.pt"]
# You can exclude models if needed
excluded_models = []

models = [model for model in models if model not in excluded_models]

# Multiple of 128
resolution_width = 1920
resolution_height = 1152


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
    def __init__(self, nsamples=50):
        self.framerate = deque(maxlen=nsamples)

    def update(self, duration):
        self.framerate.append(duration)

    def accumulate(self):
        if len(self.framerate) > 1:
            return np.average(self.framerate)
        else:
            return 0.0


def draw_text(
        img,
        text,
        font=cv2.FONT_HERSHEY_SIMPLEX,
        pos=(0, 0),
        font_scale=1,
        font_thickness=2,
        text_color=(0, 255, 0),
        text_color_bg=(0, 0, 0),
):
    """Draw text with background on image"""
    offset = (5, 5)
    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    rec_start = tuple(x - y for x, y in zip(pos, offset))
    rec_end = tuple(x + y for x, y in zip((x + text_w, y + text_h), offset))
    cv2.rectangle(img, rec_start, rec_end, text_color_bg, -1)
    cv2.putText(
        img,
        text,
        (x, int(y + text_h + font_scale - 1)),
        font,
        font_scale,
        text_color,
        font_thickness,
        cv2.LINE_AA,
    )
    return text_size


def process_image(img_src, img_size, stride, half=False):
    '''Process image before image inference.'''
    image = letterbox(img_src, img_size)[0]
    # Convert
    image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    image = torch.from_numpy(np.ascontiguousarray(image))
    image = image.half() if half else image.float()  # uint8 to fp16/32
    image /= 255  # 0 - 255 to 0.0 - 1.0

    return image, img_src


def evaluate_with_annotations(source, weights, annotation_path, img_size, conf_thres=0.25, iou_thres=0.45,
                              agnostic_nms=False, device='0'):
    """Process video with annotations and evaluate detection performance"""
    # Initialize
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride

    # Fix: Properly handle img_size
    # Make sure img_size is divisible by stride (typically 32 for YOLO models)
    img_width = check_img_size(img_size[0], s=stride)
    img_height = check_img_size(img_size[1], s=stride)
    input_shape = (img_width, img_height)
    model_name = Path(weights).stem

    if half:
        model.half()  # to FP16

    # Load annotations
    with open(annotation_path, 'r') as f:
        annotations = json.load(f)

    # Open video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video {source}")
        return os.path.basename(source), Path(weights).stem, 0

    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video properties: {orig_width}x{orig_height} @ {fps}fps, {total_frames} frames")

    # Get video name
    video_name = Path(source).stem

    # Create output directory
    output_dir = f"../output/{video_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize FPS calculator
    fps_calculator = CalcFPS(nsamples=50)

    # Get names
    names = model.module.names if hasattr(model, 'module') else model.names

    # Warmup - Use an input that's properly sized according to model requirements
    img = torch.zeros((1, 3, img_height, img_width), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    # Track metrics
    total_annotations = 0
    detected_annotations = 0
    frame_count = 0
    processed_frames = 0

    # Store frames for video
    frames = []

    # Determine if annotations are expected for all frames or if we need a mapping
    # This is a key addition to handle potential frame index mismatches
    frame_indices = set()
    for entity in annotations.get("entities", []):
        frame_idx = entity.get("blob", {}).get("frame_idx")
        if frame_idx is not None:
            frame_indices.add(frame_idx)

    # Process frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame to target resolution if specified
        if input_shape[0] != orig_width or input_shape[1] != orig_height:
            frame = cv2.resize(frame, (input_shape[0], input_shape[1]))

        # Get annotations for this frame
        frame_annotations = [entity for entity in annotations.get("entities", [])
                             if entity.get("blob", {}).get("frame_idx") == frame_count]
        if len(frame_annotations) > 0:
            if frame_annotations:
                processed_frames += 1
            # Prepare image for inference
            img, _ = process_image(frame, input_shape, stride, half)
            img = img.to(device)
            if len(img.shape) == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time.time()
            with torch.no_grad():
                pred = model(img)[0]
            t2 = time.time()

            # Apply NMS
            classes = [0, 2, 3, 5, 7]
            det = non_max_suppression(pred, conf_thres, iou_thres,
                                      classes=classes, agnostic=agnostic_nms)[0]

            # Update FPS calculator
            inference_time = t2 - t1
            fps_calculator.update(1.0 / max(inference_time, 1e-5))
            avg_fps = fps_calculator.accumulate()

            # Process predictions
            predictions = []
            if det is not None and len(det):
                # Rescale boxes from img_size to frame size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()

                # Convert to numpy array
                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = [int(x) for x in xyxy]
                    predictions.append(np.array([x1, y1, x2, y2, conf.cpu().numpy(), cls.cpu().numpy()]))

            has_annotations = len(frame_annotations) > 0
            frame_detected = 0

            # Draw annotations (red by default)
            for annotation in frame_annotations:
                bb = annotation.get("bb", [])
                if bb:
                    # Scale bounding box coordinates if needed
                    scale_x = input_shape[0] / orig_width if input_shape[0] != orig_width else 1
                    scale_y = input_shape[1] / orig_height if input_shape[1] != orig_height else 1

                    x = int(bb[0] * scale_x)
                    y = int(bb[1] * scale_y)
                    w = int(bb[2] * scale_x)
                    h = int(bb[3] * scale_y)

                    # Red by default, green if detected
                    box_color = (0, 0, 255)  # BGR format
                    cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
                    label = f"GT:{annotation.get('type', '')}"
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

                    # Check if annotation is detected by YOLO
                    detected = False
                    for pred in predictions:
                        pred_box = pred[:4]
                        anno_box_xyxy = [x, y, x + w, y + h]
                        iou = calculate_iou(pred_box, anno_box_xyxy)
                        if iou > 0.3:  # IOU threshold
                            detected = True
                            break

                    if detected:
                        frame_detected += 1

                    # Draw detection status and update box color
                    status = "DETECTED" if detected else "MISSED"
                    status_color = (0, 255, 0) if detected else (0, 0, 255)

                    # Change box color to green if detected
                    if detected:
                        box_color = (0, 255, 0)  # Green
                        # Redraw with green color
                        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
                        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

                    cv2.putText(frame, status, (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)

            # Update metrics
            if has_annotations:
                total_annotations += len(frame_annotations)
                detected_annotations += frame_detected

            # Draw YOLO detections (blue boxes)
            for pred in predictions:
                box = pred[:4].astype(int)
                conf = pred[4]
                cls = int(pred[5])
                x1, y1, x2, y2 = box

                # Blue for YOLO boxes
                color = (255, 0, 0)  # Blue in BGR
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"YOLO: {names[cls]} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Draw frame info and metrics
            accuracy = detected_annotations / total_annotations if total_annotations > 0 else 0

            # Draw info on frame
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Accuracy: {accuracy:.2f} ({detected_annotations}/{total_annotations})",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Model: {model_name}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Inference: {inference_time * 1000:.1f}ms", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)

            frames.append(frame)
        frame_count += 1

    # Calculate final accuracy
    final_accuracy = detected_annotations / total_annotations if total_annotations > 0 else 0

    # Create output video with accuracy in filename
    output_video_path = os.path.join(output_dir,
                                     f"{final_accuracy:.4f}_{model_name}_{input_shape[0]}x{input_shape[1]}.mp4")

    # Write frames to video
    if frames:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (input_shape[0], input_shape[1]))
        for frame in frames:
            out.write(frame)
        out.release()
        print(f"Output video saved to {output_video_path}")
    else:
        print("Warning: No frames to write to video")

    # Release resources
    cap.release()
    return video_name, model_name, final_accuracy


def process_video_for_model(args):
    """Process video for a specific model"""
    video_path, model_path, dataset_mode = args

    video_name = os.path.basename(video_path)
    if dataset_mode == "personpath22":
        # Just the filename without extension for personpath22
        annotation_path = f"../dataset/personpath22/annotation/anno_visible_2022/{video_name}.json"
    else:
        annotation_path = f"../dataset/DETRAC_Upload/labels/annotations/{video_name}.json"

    # Proceed with evaluation
    return evaluate_with_annotations(
        source=video_path,
        weights=model_path,
        annotation_path=annotation_path,
        img_size=(resolution_width, resolution_height),
        conf_thres=0.25,
        iou_thres=0.45
    )


def process_all_videos(dataset_mode):
    """Process all videos in the dataset"""
    # Create results directory
    os.makedirs("../results", exist_ok=True)

    # Get base directory for videos
    if dataset_mode == "personpath22":
        video_paths = glob.glob("../dataset/personpath22/raw_data/*.mp4")
    else:
        video_paths = glob.glob("../dataset/DETRAC_Upload/videos/*.mp4")


    # Define specific videos to process
    target_videos = ["uid_vid_00226.mp4", "uid_vid_00115.mp4", "uid_vid_00111.mp4", "uid_vid_00105.mp4"]

    # Create full paths for the target videos
    video_paths = []
    for video_name in target_videos:
        full_path = os.path.join("../dataset/personpath22/raw_data", video_name)
        video_paths.append(full_path)

    print(f"Found {len(video_paths)} videos to process")

    # Set up multiprocessing
    torch.multiprocessing.set_start_method("spawn", force=True)

    with open(f"../results/all_results_{dataset_mode}.txt", "a+") as f_all:
        for video_path in tqdm(video_paths):
            video_name = os.path.basename(video_path)
            f_all.write(f"\n{video_name}:\n")
            print(f"\nProcessing video: {video_name}")

            # Create tasks for current video and all models
            video_tasks = []
            for model in models:
                video_tasks.append((video_path, model, dataset_mode))
            # Process sequentially
            with multiprocessing.Pool(2) as pool:
                video_results = pool.map(process_video_for_model, video_tasks)

            # Write results to file
            for result_video_name, model_name, accuracy in video_results:
                f_all.write(f"  {model_name}: {accuracy:.4f}\n")


if __name__ == '__main__':
    # Set the dataset mode: either "DETRAC" or "personpath22"
    DATASET_MODE = "personpath22"

    process_all_videos(DATASET_MODE)