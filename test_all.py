import cv2
import json
import os
import numpy as np
import multiprocessing
import torch
from ultralytics import YOLO
import sys
import glob

"""
Tylko modele ultralytics
"""

# Define YOLO model versions
models = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt", "yolov5n6u", "yolov5s6u", "yolov5m6u",
          "yolov5l6u", "yolov5x6u", "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt", "yolov10n.pt",
          "yolov10s.pt", "yolov10m.pt", "yolov10b.pt", "yolov10x.pt", "yolov9t.pt", "yolov9s.pt", "yolov9m.pt",
          "yolov9c.pt", "yolov9e.pt", "yolo12n.pt", "yolo12s.pt", "yolo12m.pt", "yolo12l.pt", "yolo12x.pt"]
EXCLUDED_DATASETLS = []

models = [model for model in models if model not in EXCLUDED_DATASETLS]

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


def process_video(args):
    video_path, model_version, DATASET = args
    sys.stderr = open(os.devnull, 'w')

    # Extract video name without extension
    video_name = os.path.basename(video_path)
    video_name_no_ext = os.path.splitext(video_name)[0]

    # Load YOLO model
    model_name = model_version.replace(".pt", "")
    model = YOLO(f"weights/{model_version}")

    if DATASET == "personpath22":
        annotation_path = f"dataset/personpath22/annotation/anno_visible_2022/{video_name}.json"

        with open(annotation_path, 'r') as f:
            annotations = json.load(f)
    else:
        annotation_path = f"dataset/DETRAC_Upload/labels/annotations/{video_name}.json"
        with open(annotation_path, 'r') as f:
            annotations = json.load(f)
    output_dir = f"output/{video_name_no_ext}"
    os.makedirs(output_dir, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

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

        # Get annotations for this frame
        frame_annotations = [entity for entity in annotations.get("entities", [])
                             if entity.get("blob", {}).get("frame_idx") == frame_count]

        # Skip frames without annotations
        if not frame_annotations:
            frame_count += 1
            continue

        # Run YOLO on frame
        results = model(frame, classes=[0, 1, 2, 3], verbose=False, imgsz=(WIDTH, HEIGHT), augment=(not model_version.startswith("yolov10")), half=True)

        # Extract predictions from results
        predictions = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = box.cls[0].cpu().numpy()
                predictions.append(np.array([x1, y1, x2, y2, conf, cls]))

        has_annotations = len(frame_annotations) > 0
        frame_detected = 0

        # Draw annotations (red by default)
        for annotation in frame_annotations:
            bb = annotation.get("bb", [])
            if bb:
                x, y, w, h = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])

                # Red by default, green if detected
                box_color = (0, 0, 255)  # Red by default (BGR format)
                cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
                label = f"GT:{annotation.get('type', '')}"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

                # Check if this annotation is detected by YOLO
                detected = False
                for pred in predictions:
                    pred_box = pred[:4]
                    anno_box_xyxy = [bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]]
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
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Accuracy: {accuracy:.2f} ({detected_annotations}/{total_annotations})",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Model: {model_name}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Video: {video_name}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Store frame
        frames.append(frame)
        frame_count += 1

    # Calculate final accuracy
    final_accuracy = detected_annotations / total_annotations if total_annotations > 0 else 0

    # Create output video with accuracy in the filename
    output_video_path = os.path.join(output_dir, f"{final_accuracy:.4f}_{model_name}_{WIDTH}x{HEIGHT}.mp4")

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
    # Get all MP4 files in the dataset directory
    video_paths = glob.glob("dataset/personpath22/raw_data/*.mp4" if DATASET == "personpath22" else "dataset/DETRAC_Upload/videos/*.mp4")

    print(f"Found {len(video_paths)} videos to process")

    # Set up multiprocessing
    torch.multiprocessing.set_start_method("spawn", force=True)

    # Create results directory
    os.makedirs("results", exist_ok=True)

    # Import tqdm for progress tracking
    from tqdm import tqdm

    # Process one video at a time with progress bar
    with open(f"results/all_results_{DATASET}.txt", "a+") as f_all:
        for video_path in tqdm(video_paths):
            video_name = os.path.basename(video_path)
            f_all.write(f"\n{video_name}:\n")
            print(f"\nProcessing video: {video_name}")

            # Create tasks for current video only
            video_tasks = [(video_path, model, DATASET) for model in models]

            # Process each model for this video in parallel
            with multiprocessing.Pool(threads_num) as pool:
                video_results = pool.map(process_video, video_tasks)

            for _, model_name, accuracy in video_results:
                f_all.write(f"  {model_name}: {accuracy:.4f}\n")



if __name__ == "__main__":
    DATASET = "personpath22"  # personpath22 DETRAC
    process_all_videos(DATASET, threads_num=6)
