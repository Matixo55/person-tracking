import cv2
import json
import os
import numpy as np
import torch
import time
import psutil
import GPUtil
import glob
from tqdm import tqdm
import sys

# Import YOLOv7 specific modules
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords, set_logging
from utils.torch_utils import select_device, time_synchronized

# Define YOLOv7 model versions
models = ["yolov7.pt", "yolov7x.pt", "yolov7-w6.pt", "yolov7-e6.pt", "yolov7-d6.pt", "yolov7-e6e.pt"]
EXCLUDED_MODELS = []

models = [model for model in models if model not in EXCLUDED_MODELS]

# multiples of 32
WIDTH, HEIGHT = 1920, 1088


# Define IoU calculation function
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection

    return intersection / union if union > 0 else 0


# Function to get GPU metrics (same as in the first script)
def get_gpu_metrics():
    gpus = GPUtil.getGPUs()
    gpu = gpus[0]  # Get the first GPU
    return {
        "gpu_name": gpu.name,
        "gpu_load": gpu.load * 100,  # Convert to percentage
        "gpu_memory_used": gpu.memoryUsed,
        "gpu_memory_total": gpu.memoryTotal
    }



def process_video(args):
    video_path, model_version, MODE, benchmark_mode = args

    # Extract video name without extension
    video_name = os.path.basename(video_path)
    video_name_no_ext = os.path.splitext(video_name)[0]

    # Initialize metrics, same as first script
    benchmark_metrics = {
        "model": model_version,
        "video": video_name,
        "total_time": 0,
        "frames_processed": 0,
        "fps": 0,
        "max_memory_usage_mb": 0,
        "avg_memory_usage_mb": 0,
        "max_gpu_memory_mb": 0,
        "avg_gpu_memory_mb": 0,
        "max_gpu_load": 0,
        "avg_gpu_load": 0
    }

    memory_samples = []
    gpu_memory_samples = []
    gpu_load_samples = []

    # Load annotations based on MODE
    if MODE == "personpath22":
        annotation_path = f"../dataset/personpath22/annotation/anno_visible_2022/{video_name}.json"
        with open(annotation_path, 'r') as f:
            annotations = json.load(f)
    elif MODE == "DETRAC":
        annotation_path = f"../dataset/DETRAC_Upload/labels/annotations/{video_name}.json"
        with open(annotation_path, 'r') as f:
            annotations = json.load(f)
    elif MODE == "benchmark" and not benchmark_mode:
        annotation_path = f"../dataset/camera/annotations/{video_name}.json"
        with open(annotation_path, 'r') as f:
            annotations = json.load(f)

    output_dir = f"output/{video_name_no_ext}"
    os.makedirs(output_dir, exist_ok=True)

    # Track metrics
    total_annotations = 0
    detected_annotations = 0
    frame_count = 0

    # We'll create the output writer after determining final accuracy
    frames = []

    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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

    # Load YOLOv7 model (different from first script)
    model_name = model_version.replace(".pt", "")
    device = select_device('0')  # Use GPU
    model = attempt_load(model_version, map_location=device)

    # Half precision
    model.half()
    model(torch.zeros(1, 3, HEIGHT, WIDTH).to(device).type_as(next(model.parameters())))

    i = 0
    # Process frames
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

        # Start timing at regular intervals

        # Resize frame to target resolution
        frame_resized = cv2.resize(frame, (WIDTH, HEIGHT))

        if not benchmark_mode:
            frame_annotations = [entity for entity in annotations.get("entities", [])
                                 if entity.get("blob", {}).get("frame_idx") == frame_count]

            # Skip frames without annotations
            if not frame_annotations:
                frame_count += 1
                continue

        # Prepare image for YOLOv7 - different preprocessing from YOLOv8
        img = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.half()  # uint8 to fp16
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        if i % 10 == 0:
            frame_start_time = time.time()

        # Inference
        with torch.no_grad():
            pred = model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=[0, 1, 2, 3])

        # Record time for benchmarking at regular intervals
        if i % 10 == 0:
            frame_process_time = time.time() - frame_start_time
            benchmark_metrics["total_time"] += frame_process_time
            benchmark_metrics["frames_processed"] += 1

        # Process predictions - convert to a format similar to the first script
        if not benchmark_mode:
            predictions = []
            for det in pred:
                if len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame_resized.shape).round()
                    for *xyxy, conf, cls in det:
                        x1, y1, x2, y2 = [int(x) for x in xyxy]
                        predictions.append(np.array([x1, y1, x2, y2, conf.cpu().numpy(), cls.cpu().numpy()]))

        if not benchmark_mode:
            has_annotations = len(frame_annotations) > 0
            frame_detected = 0

            # Scale factor for annotations
            scale_x = WIDTH / orig_width
            scale_y = HEIGHT / orig_height

            # Draw annotations (red by default)
            for annotation in frame_annotations:
                bb = annotation.get("bb", [])
                if bb:
                    x = int(bb[0] * scale_x)
                    y = int(bb[1] * scale_y)
                    w = int(bb[2] * scale_x)
                    h = int(bb[3] * scale_y)

                    box_color = (0, 0, 255)  # Red by default (BGR format)
                    cv2.rectangle(frame_resized, (x, y), (x + w, y + h), box_color, 2)
                    label = f"GT:{annotation.get('type', '')}"
                    cv2.putText(frame_resized, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

                    # Check if this annotation is detected by YOLO
                    detected = False
                    for pred in predictions:
                        pred_box = pred[:4]
                        anno_box_xyxy = [x, y, x + w, y + h]
                        iou = calculate_iou(pred_box, anno_box_xyxy)
                        if iou > 0.3:
                            detected = True
                            break

                    if detected:
                        frame_detected += 1

                    status = "DETECTED" if detected else "MISSED"
                    status_color = (0, 255, 0) if detected else (0, 0, 255)

                    # Change box color to green if detected
                    if detected:
                        box_color = (0, 255, 0)  # Green (BGR format)
                        # Redraw the box with green color
                        cv2.rectangle(frame_resized, (x, y), (x + w, y + h), box_color, 2)
                        # Redraw the label
                        cv2.putText(frame_resized, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

                    cv2.putText(frame_resized, status, (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)

            if has_annotations:
                total_annotations += len(frame_annotations)
                detected_annotations += frame_detected

        if not benchmark_mode:
            # Draw YOLO detections (blue boxes)
            for pred in predictions:
                box = pred[:4].astype(int)
                conf = pred[4]
                cls = int(pred[5])
                x1, y1, x2, y2 = box

                # Make all YOLO boxes blue
                color = (255, 0, 0)  # Blue in BGR format
                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color, 2)
                label = f"YOLO:{cls} {conf:.2f}"
                cv2.putText(frame_resized, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Draw frame info and metrics
            accuracy = detected_annotations / total_annotations if total_annotations > 0 else 0
            cv2.putText(frame_resized, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)
            cv2.putText(frame_resized, f"Accuracy: {accuracy:.2f} ({detected_annotations}/{total_annotations})",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame_resized, f"Model: {model_name}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),
                        2)
            cv2.putText(frame_resized, f"Time: {frame_process_time:.4f}s", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)

            frames.append(frame_resized)

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

    output_video_path = os.path.join(output_dir, f"{final_accuracy:.4f}_{model_name}_{WIDTH}x{HEIGHT}.mp4")
    print(benchmark_metrics)

    # Write all stored frames to the video
    if frames:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (WIDTH, HEIGHT))
        for frame in frames:
            out.write(frame)
        out.release()

    # Release resources
    cap.release()

    return video_name, model_name, benchmark_metrics


def process_all_videos(MODE):
    benchmark_mode = False
    if MODE == "benchmark":
        benchmark_mode = True
        video_paths = glob.glob("../dataset/camera/*.mp4")
    elif MODE == "DETRAC":
        video_paths = glob.glob("../dataset/DETRAC_Upload/videos/*.mp4")
    elif MODE == "personpath22":
        video_paths = glob.glob("../dataset/personpath22/raw_data/*.mp4")

    # Set up multiprocessing
    torch.multiprocessing.set_start_method("spawn", force=True)

    # Create results directory
    os.makedirs("results", exist_ok=True)

    # Store all benchmark results
    all_benchmark_results = []

    # Process one video at a time with progress bar
    with open(f"../results/all_results_{MODE}.txt", "a+") as f_all:
        for video_path in tqdm(video_paths):
            video_name = os.path.basename(video_path)
            f_all.write(f"\n{video_name}:\n")
            print(f"\nProcessing video: {video_name}")

            for model in tqdm(models):
                video_name, model_name, metrics = process_video((video_path, model, MODE, benchmark_mode))

                if benchmark_mode:
                    # Store results for sorting
                    all_benchmark_results.append(metrics)

                    # Write CSV-formatted line
                    f_all.write(
                        f"  {model_name}: {metrics['accuracy']:.4f} {metrics['fps']:.2f},{metrics['total_time']:.2f},{metrics['frames_processed']},{metrics['max_memory_usage_mb']:.2f},{metrics['avg_memory_usage_mb']:.2f},{metrics['max_gpu_memory_mb']:.2f},{metrics['avg_gpu_memory_mb']:.2f},{metrics['max_gpu_load']:.2f},{metrics['avg_gpu_load']:.2f},{metrics['start_gpu_memory_used']:.2f},{metrics['start_gpu_load']:.2f},{metrics['start_memory_usage_mb']:.2f}\n")
                else:
                    f_all.write(
                        f"  {model_name}: {metrics['accuracy']:.4f}\n")


if __name__ == "__main__":
    # Set up logging for YOLOv7
    set_logging()

    # Allow the Model class in torch.serialization for YOLOv7
    # This is important for YOLOv7 model loading
    from models.yolo import Model

    torch.serialization.add_safe_globals([Model])

    MODE = "benchmark"  # Options: "DETRAC", "benchmark", "personpath22"
    process_all_videos(MODE)