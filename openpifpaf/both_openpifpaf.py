import cv2
import json
import os
import numpy as np
import multiprocessing
import sys
import glob
import torch
import openpifpaf

# multiples of 32, same as original script
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
    video_path, model_name, MODE = args
    sys.stderr = open(os.devnull, 'w')

    # Extract video name without extension
    video_name = os.path.basename(video_path)
    video_name_no_ext = os.path.splitext(video_name)[0]

    # Initialize OpenPifPaf model
    predictor = openpifpaf.Predictor(checkpoint=model_name, device="cuda:0")

    # Load annotations
    if MODE == "personpath22":
        annotation_path = f"../dataset/personpath22/annotation/anno_visible_2022/{video_name}.json"
        with open(annotation_path, 'r') as f:
            annotations = json.load(f)
    else:
        annotation_path = f"../dataset/DETRAC_Upload/labels/annotations/{video_name}.json"
        with open(annotation_path, 'r') as f:
            annotations = json.load(f)

    output_dir = f"output/{video_name_no_ext}"
    os.makedirs(output_dir, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Track metrics
    total_annotations = 0
    detected_annotations = 0
    frame_count = 0

    # Store frames to write to video later
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

        # Run OpenPifPaf on frame
        # OpenPifPaf expects PIL images
        pil_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        predictions, gt_anns, image_meta = predictor.numpy_image(pil_img)

        # Extract bounding boxes from predictions
        boxes = []
        for pred in predictions:
            # In detection mode, we can directly use the bounding box
            if hasattr(pred, 'bbox') and pred.bbox is not None:
                x1, y1, w, h = pred.bbox
                x2, y2 = x1 + w, y1 + h
                conf = pred.score if hasattr(pred, 'score') else 0.9  # Use score or default
                boxes.append(np.array([x1, y1, x2, y2, conf, 0]))  # Class 0 for person
            # Fallback to keypoint method if needed
            elif hasattr(pred, 'data') and pred.data is not None:
                keypoints = pred.data
                keypoints = keypoints[keypoints[:, 2] > 0]  # Filter out low confidence keypoints
                if len(keypoints) > 0:
                    x1, y1 = keypoints[:, 0].min(), keypoints[:, 1].min()
                    x2, y2 = keypoints[:, 0].max(), keypoints[:, 1].max()
                    conf = np.mean(keypoints[:, 2])  # Average confidence
                    boxes.append(np.array([x1, y1, x2, y2, conf, 0]))  # Class 0 for person

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

                # Check if this annotation is detected by OpenPifPaf
                detected = False
                for pred_box in boxes:
                    pred_box_xyxy = pred_box[:4]
                    anno_box_xyxy = [bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]]
                    iou = calculate_iou(pred_box_xyxy, anno_box_xyxy)
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

        # Draw OpenPifPaf detections (blue boxes)
        for box in boxes:
            x1, y1, x2, y2, conf, cls = box.astype(float)
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # All OpenPifPaf boxes in blue
            color = (255, 0, 0)  # Blue in BGR format
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"PifPaf: {conf:.2f}"
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
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        for frame in frames:
            out.write(frame)
        out.release()

    # Release resources
    cap.release()

    return video_name, model_name, final_accuracy


def process_all_videos():
    # Define available OpenPifPaf models
    pifpaf_models = ["shufflenetv2k16", "shufflenetv2k30", "resnet50", "resnet101", "resnet152"]

    # Get all MP4 files in the dataset directory
    MODE = "personpath22"  # Change to "DETRAC" for DETRAC dataset
    video_paths = glob.glob(
        "../dataset/personpath22/raw_data/*.mp4" if MODE == "personpath22" else "../dataset/DETRAC_Upload/videos/*.mp4")

    print(f"Found {len(video_paths)} videos to process")

    # Set up multiprocessing
    torch.multiprocessing.set_start_method("spawn", force=True)

    # Create results directory
    os.makedirs("results", exist_ok=True)

    # Import tqdm for progress tracking
    from tqdm import tqdm

    # Process one video at a time with progress bar
    with open(f"../results/all_results_{MODE}.txt", "a+") as f_all:
        for video_path in tqdm(video_paths):
            video_name = os.path.basename(video_path)
            f_all.write(f"\n{video_name}:\n")
            print(f"\nProcessing video: {video_name}")

            # Create tasks for current video only
            video_tasks = [(video_path, model, MODE) for model in pifpaf_models]

            # Process each model for this video in parallel
            with multiprocessing.Pool(min(len(pifpaf_models), 6)) as pool:
                video_results = pool.map(process_video, video_tasks)

            for _, model_name, accuracy in video_results:
                f_all.write(f"  {model_name}: {accuracy:.4f}\n")


if __name__ == "__main__":
    process_all_videos()