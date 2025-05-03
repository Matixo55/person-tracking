import os
import glob
import json
import torch
import numpy as np
import cv2
from tqdm import tqdm

from yolov6.data.data_augment import letterbox
from yolov6.layers.common import DetectBackend
from yolov6.utils.events import LOGGER
from yolov6.utils.nms import non_max_suppression

# Configuration
MODEL_PATH = "weights/yolov6l.pt"  # Using YOLOv6l as requested
INPUT_DIR = "../dataset/camera/videos/resized"
OUTPUT_DIR = "../dataset/camera/annotations"
ANNOTATED_VIDEO_DIR = "../dataset/camera/ground_truth"
DEVICE = '0'  # GPU device id (use 'cpu' for CPU)
CONF_THRES = 0.25  # Confidence threshold
IOU_THRES = 0.45  # NMS IoU threshold
CLASSES = [0, 2, 3, 5, 7]  # person, car, motorcycle, bus, truck
CLASS_NAMES = {
    0: "person",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}
MAX_DET = 300  # Maximum number of detections per image
IMG_SIZE = (1920, 1088)

# Create output directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ANNOTATED_VIDEO_DIR, exist_ok=True)


# Define color mapping for different classes
def get_color_for_class(class_id):
    """Return BGR color for a specific class ID"""
    colors = {
        0: (0, 255, 0),  # person: green
        2: (0, 0, 255),  # car: red
        3: (255, 0, 0),  # motorcycle: blue
        5: (0, 255, 255),  # bus: yellow
        7: (255, 255, 0),  # truck: cyan
    }
    return colors.get(class_id, (255, 255, 255))  # default: white


class YOLOv6Detector:
    def __init__(self, model_path, device, img_size, half):
        self.model_path = model_path
        self.device = device
        self.half = half
        self.model_name = os.path.splitext(os.path.basename(model_path))[0]

        # Initialize model
        cuda = self.device != 'cpu' and torch.cuda.is_available()
        self.device = torch.device(f'cuda:{device}' if cuda else 'cpu')
        print(f"Using device: {self.device}")

        self.model = DetectBackend(model_path, device=self.device)
        self.stride = self.model.stride
        self.check_img_size(img_size, s=self.stride)
        self.img_size = img_size

        # Switch model to deploy status
        self.model_switch(self.model.model, self.img_size)

        # Half precision
        if self.half & (self.device.type != 'cpu'):
            self.model.model.half()
        else:
            self.model.model.float()
            self.half = False

        # Warmup
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, *self.img_size).to(self.device).type_as(
                next(self.model.model.parameters())))

    def model_switch(self, model, img_size):
        '''Model switch to deploy status'''
        from yolov6.layers.common import RepVGGBlock
        for layer in model.modules():
            if isinstance(layer, RepVGGBlock):
                layer.switch_to_deploy()
            elif isinstance(layer, torch.nn.Upsample) and not hasattr(layer, 'recompute_scale_factor'):
                layer.recompute_scale_factor = None  # torch 1.11.0 compatibility

        LOGGER.info("Switch model to deploy modality.")

    def check_img_size(self, img_size, s=32, floor=0):
        assert img_size[0] % s == 0 and img_size[1] % s == 0, "invalid img_size"

    def process_image(self, img_src):
        '''Process image before inference'''
        image = letterbox(img_src, self.img_size, stride=self.stride)[0]
        # Convert
        image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        image = torch.from_numpy(np.ascontiguousarray(image))
        image = image.half() if self.half else image.float()  # uint8 to fp16/32
        image /= 255  # 0 - 255 to 0.0 - 1.0

        return image, img_src

    def rescale(self, ori_shape, boxes, target_shape):
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

    def detect(self, frame, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic_nms=False, max_det=None):
        """Detect objects in the frame"""
        # Prepare image
        img, img_src = self.process_image(frame)
        img = img.to(self.device)
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        # Inference
        with torch.no_grad():
            pred_results = self.model(img)

        # Apply NMS
        det = non_max_suppression(pred_results, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]

        # Process detections
        detections = []
        if len(det):
            # Rescale boxes from img_size to frame size
            det[:, :4] = self.rescale(img.shape[2:], det[:, :4], frame.shape).round()

            # Convert to xywh format (top-left x, top-left y, width, height)
            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = [int(x.cpu().item()) for x in xyxy]
                w = x2 - x1
                h = y2 - y1

                detections.append({
                    "bb": [x1, y1, w, h],
                    "confidence": float(conf),
                    "class": int(cls)
                })

        return detections

    def process_video(self, video_path, output_path, annotated_video_path,
                      conf_thres=0.25, iou_thres=0.45, classes=None, generate_video=False):
        """Process video, save detections to JSON file and create annotated video simultaneously"""
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps else 0
        minutes = int(duration) // 60
        seconds = int(duration) % 60

        # Calculate scaling factors if video isn't already at target resolution
        scale_x = IMG_SIZE[0] / width
        scale_y = IMG_SIZE[1] / height

        # Create output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        if generate_video:
            out = cv2.VideoWriter(annotated_video_path, fourcc, fps, IMG_SIZE)

        frame_count = 0

        # Initialize annotations structure
        annotations = {
            "entities": []
        }

        # Process frames
        print(f"Processing {video_path}...")
        print(f"Dimensions: {width}x{height}, FPS: {fps}, Duration: {minutes:02d}:{seconds:02d} (mm:ss)")

        i = 0

        for _ in tqdm(range(total_frames)):
            ret, frame = cap.read()
            if not ret:
                break

            i+= 1

            if i%2 == 1:
                continue

            # Resize frame to target resolution if needed
            if width != IMG_SIZE[0] or height != IMG_SIZE[1]:
                frame = cv2.resize(frame, IMG_SIZE)

            # Create a copy for annotation
            if generate_video:
                annotated_frame = frame.copy()

            # Detect objects in frame
            detections = self.detect(
                frame,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                classes=classes,
                max_det=MAX_DET
            )

            # Add detections to annotations and draw on frame
            for det in detections:
                # Add to JSON annotations
                entity = {
                    "bb": det["bb"],
                    "blob": {
                        "frame_idx": frame_count
                    },
                    "confidence": det["confidence"],
                    "class": det["class"],
                    "class_name": CLASS_NAMES.get(det["class"], "unknown")
                }
                annotations["entities"].append(entity)

                # Draw on frame
                if generate_video:
                    x, y, w, h = det["bb"]
                    class_id = det["class"]
                    confidence = det["confidence"]
                    class_name = CLASS_NAMES.get(class_id, "unknown")

                    color = get_color_for_class(class_id)
                    cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)

                    # Add label
                    label = f"{class_name} ({confidence:.2f})"
                    cv2.putText(annotated_frame, label, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if generate_video:
                out.write(annotated_frame)

            frame_count += 1

        # Release video resources
        cap.release()
        if generate_video:
            out.release()

        # Save annotations to JSON file
        with open(output_path, 'w') as f:
            json.dump(annotations, f, indent=2)

        return annotations


def main():
    # Initialize detector
    detector = YOLOv6Detector(
        model_path=MODEL_PATH,
        device=DEVICE,
        img_size=IMG_SIZE,
        half=True
    )

    # Get all videos
    video_files = glob.glob(os.path.join(INPUT_DIR, "*.mp4"))
    video_files= [
        "../dataset/camera/videos/resized/Golski_Rejestrator 2_20250415121828_20250415123535_519830320.mp4",
        "../dataset/camera/videos/resized/Aerodynamika_192.168.5.149_20250415073958_20250415083559_501510727.mp4",
        "../dataset/camera/videos/resized/Mel_192.168.5.149_20250415114008_20250415123534_518974247.mp4",
        "../dataset/camera/videos/resized/PLAC_192.168.5.1_20250415120616_20250415124035_518024578.mp4",
    ]

    print(f"Found {len(video_files)} videos")

    # Process each video
    for i, video_path in enumerate(video_files):
        print(f"Processing video {i + 1}/{len(video_files)}")
        video_name = os.path.basename(video_path)

        # Define output paths
        json_output_path = os.path.join(OUTPUT_DIR, f"{video_name}.json")
        annotated_video_path = os.path.join(ANNOTATED_VIDEO_DIR, video_name)

        # Process video, save annotations, and create annotated video
        detector.process_video(
            video_path=video_path,
            output_path=json_output_path,
            annotated_video_path=annotated_video_path,
            conf_thres=CONF_THRES,
            iou_thres=IOU_THRES,
            classes=CLASSES,
            generate_video=True
        )


if __name__ == "__main__":
    main()