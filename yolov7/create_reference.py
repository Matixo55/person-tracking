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
models = ["yolov7x.pt" ]#"yolov7.pt", "yolov7-e6e.pt", "yolov7-w6.pt", "yolov7-e6.pt", "yolov7-d6.pt", ]

# Multiple of 64
WIDTH = 1920
HEIGHT = 1088


def process_video(model_path):
    video_path = "video.mp4"
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}!")
        return model_name, 0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)

    # Extract model name
    model_name = os.path.splitext(os.path.basename(model_path))[0]

    print(f"Processing {model_name} on {video_name}...")

    # Initialize YOLOv7
    device = select_device('0' if torch.cuda.is_available() else 'cpu')
    try:
        model = attempt_load(model_path, map_location=device)
        model.half()  # to FP16
        model(torch.zeros(1, 3, WIDTH, HEIGHT).to(device).type_as(next(model.parameters())))
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return model_name, 0

    # Track metrics for reporting
    frame_count = 0
    total_people = 0
    avg_inference_time = 0

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_path = f"output/{video_name}_{model_name}.mp4"
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (WIDTH, HEIGHT))

    # Process frames
    with tqdm(desc=f"Processing {model_name}", total=total_frames, unit="frames") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame to target resolution
            frame = cv2.resize(frame, (WIDTH, HEIGHT))

            # Make a copy of the frame for drawing detections
            output_frame = frame.copy()

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
            inference_time = (t2 - t1) * 1000  # ms
            avg_inference_time += inference_time

            # Apply NMS - only for people (class 0)
            pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=[0])

            # Process predictions
            frame_people = 0
            for i, det in enumerate(pred):
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()

                    # Draw detections on output frame - only people (class 0)
                    for *xyxy, conf, cls in det:
                        x1, y1, x2, y2 = [int(x) for x in xyxy]
                        confidence = float(conf.cpu().numpy())

                        # Calculate area of bounding box
                        area = (x2 - x1) * (y2 - y1)

                        if area < 7000:
                            continue

                        # Red color for people
                        color = (0, 0, 255)  # Red in BGR format

                        # Draw bounding box
                        cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)

                        # Draw label with confidence and area
                        label = f"{confidence:.2f}"
                        cv2.putText(output_frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                        # Count only detections with area >= 500
                        frame_people += 1

            # Update metrics
            total_people += frame_people

            # Add info text to the frame
            cv2.putText(output_frame, f"Frame: {frame_count}/{total_frames}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(output_frame, f"Model: {model_name}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(output_frame, f"People detected: {frame_people}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(output_frame, f"Total people: {total_people}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(output_frame, f"Inference: {inference_time:.1f}ms", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Write the frame to output video
            out.write(output_frame)

            frame_count += 1
            pbar.update(1)

    # Calculate average inference time
    avg_inference_time = avg_inference_time / frame_count if frame_count > 0 else 0

    # Release resources
    cap.release()
    out.release()

    print(f"Saved output video to {output_video_path}")
    print(f"  - Total frames: {frame_count}")
    print(f"  - Total people detected: {total_people}")
    print(f"  - Avg people per frame: {total_people / frame_count:.2f}")
    print(f"  - Avg inference time: {avg_inference_time:.2f}ms")

    return model_name, total_people


def main():
    # Set up logging
    set_logging()

    # Allow the Model class in torch.serialization for YOLOv7
    from models.yolo import Model
    torch.serialization.add_safe_globals([Model])

    print(f"Processing video.mp4 with {len(models)} YOLOv7 models")
    print(f"Detecting only people (class 0)")
    print(f"Output videos will be saved to the 'output' directory")

    # Process each model for the video
    results = []
    for model_path in models:
        model_name, total_people = process_video(model_path)
        results.append((model_name, total_people))

    # Print results
    print("\nPerson detection results for video.mp4:")
    for model_name, total_people in results:
        print(f"  {model_name}: {total_people} people detected")

    print("\nAll output videos have been saved to the 'output' directory")


if __name__ == "__main__":
    main()