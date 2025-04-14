import glob
import json
import multiprocessing
import os

import cv2
import numpy as np
import torch
import torch.serialization
from tqdm import tqdm

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords, set_logging
from utils.torch_utils import select_device, time_synchronized

# Define YOLOv7 model versions
models = ["yolov7.pt", "yolov7x.pt", "yolov7-w6.pt", "yolov7-e6.pt", "yolov7-d6.pt", "yolov7-e6e.pt"]

# Multiple of 64
WIDTH = 1920
HEIGHT = 1088


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


def process_video(args):
    video_path, model_path, DATASET = args

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if DATASET == "personpath22":
        annotation_path = f"../dataset/personpath22/annotation/anno_visible_2022/{video_name}.mp4.json"

        with open(annotation_path, 'r') as f:
            annotations = json.load(f)
    else:
        annotation_path = f"../dataset/DETRAC_Upload/labels/annotations/{video_name}.mp4.json"
        with open(annotation_path, 'r') as f:
            annotations = json.load(f)

    # Create output directory
    output_dir = f"../output/{video_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(video_path)
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Set output resolution
    width, height = WIDTH, HEIGHT

    # Extract model name
    model_name = os.path.splitext(os.path.basename(model_path))[0]

    print(f"Processing {model_name} on {video_name}...")

    # Initialize YOLOv7
    device = select_device('0')  # Use '' for CPU or '0' for GPU
    # Use weights_only=False to handle YOLOv7 model loading
    model = attempt_load(model_path, map_location=device)  # load FP32 model

    # Half precision
    model.half()  # to FP16
    model(torch.zeros(1, 3, width, height).to(device).type_as(next(model.parameters())))

    # Track metrics
    total_annotations = 0
    detected_annotations = 0
    frame_count = 0

    # We'll create the output writer after determining final accuracy
    frames = []

    # Process frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame to target resolution
        frame = cv2.resize(frame, (width, height))

        # Get annotations for this frame
        frame_annotations = [entity for entity in annotations.get("entities", [])
                             if entity.get("blob", {}).get("frame_idx") == frame_count]

        # Skip frames without annotations
        if not frame_annotations:
            frame_count += 1
            continue

        # Prepare image for YOLOv7
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.half()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():
            pred = model(img, augment=False)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=[0, 1, 2, 3])

        # Process predictions
        predictions = []
        for i, det in enumerate(pred):
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()

                # Convert to numpy array format similar to our other script
                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = [int(x) for x in xyxy]
                    predictions.append(np.array([x1, y1, x2, y2, conf.cpu().numpy(), cls.cpu().numpy()]))

        has_annotations = len(frame_annotations) > 0
        frame_detected = 0

        # Draw annotations (red by default)
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
                    if iou > 0.3:
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
            cls = int(pred[5])
            x1, y1, x2, y2 = box

            # Make all YOLO boxes blue
            color = (255, 0, 0)  # Blue in BGR format
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"YOLO:{cls} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw frame info and metrics
        accuracy = detected_annotations / total_annotations if total_annotations > 0 else 0
        inference_time = (t2 - t1) * 1000  # ms

        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Accuracy: {accuracy:.2f} ({detected_annotations}/{total_annotations})",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Model: {model_name}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Video: {video_name}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Inference: {inference_time:.1f}ms", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)

        # Store frame
        frames.append(frame)
        frame_count += 1

    # Calculate final accuracy
    final_accuracy = detected_annotations / total_annotations if total_annotations > 0 else 0

    # Create output video with accuracy in the filename
    output_video_path = os.path.join(output_dir, f"{final_accuracy:.4f}_{model_name}_{width}x{height}.mp4")

    # Write all stored frames to the video
    if frames:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (WIDTH, HEIGHT))
        for frame in frames:
            out.write(frame)
        out.release()

    # Release resources
    cap.release()
    return video_name, model_name, final_accuracy


def process_all_videos(DATASET, threads_num):
    # Create results directory
    os.makedirs("../results", exist_ok=True)

    # Get all MP4 files in the dataset directory
    if DATASET == "personpath22":
        video_paths = glob.glob("../dataset/personpath22/raw_data/*.mp4")
    else:
        video_paths = glob.glob("../dataset/DETRAC_Upload/videos/*.mp4")


    print(f"Found {len(video_paths)} videos to process")

    # Set up multiprocessing
    torch.multiprocessing.set_start_method("spawn", force=True)

    # Process one video at a time with progress bar
    with open(f"../results/all_results_{DATASET}.txt", "a+") as f_all:
        for video_path in tqdm(video_paths):
            video_name = os.path.basename(video_path)
            f_all.write(f"\n{video_name}.mp4:\n")
            print(f"\nProcessing video: {video_name}")

            # Create tasks for current video only
            video_tasks = [(video_path, f"weights/{model}", DATASET) for model in models]

            # Process each model for this video in parallel
            with multiprocessing.Pool(threads_num) as pool:
                video_results = pool.map(process_video, video_tasks)

            for _, model_name, accuracy in video_results:
                f_all.write(f"  {model_name}: {accuracy:.4f}\n")

if __name__ == "__main__":
    # Set up logging
    set_logging()

    from models.yolo import Model

    torch.serialization.add_safe_globals([Model])
    DATASET = "personpath22" # DETRAC personpath22

    # Process all videos
    process_all_videos(DATASET, threads_num=2)