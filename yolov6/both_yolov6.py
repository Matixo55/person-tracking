import glob
import json
import math
import multiprocessing
import os
import os.path as osp
import time
from collections import deque

import cv2
import numpy as np
import torch
from PIL import ImageFont
from tqdm import tqdm

from openpifpaf.both_openpifpaf import WIDTH, HEIGHT
from yolov6.data.data_augment import letterbox
from yolov6.layers.common import DetectBackend
from yolov6.utils.events import LOGGER
from yolov6.utils.nms import non_max_suppression

# Define YOLOv6 model versions
models = ["yolov6s.pt", "yolov6n.pt", "yolov6m.pt", "yolov6l.pt", "yolov6n6.pt", "yolov6s6.pt", "yolov6m6.pt",
          "yolov6l6.pt"]
excluded_models = ["yolov6l6.pt"]

models = [model for model in models if model not in excluded_models]
models = ["yolov6l6.pt"]

# Multiple of 64
resolution_width = 960
resolution_height = 576


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


class YOLOv6Evaluator:
    def __init__(self, model_path, device='0', img_size=[1920, 1088], half=True):
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
        if isinstance(img_size, int):  # integer i.e. img_size=640
            new_size = max(self.make_divisible(img_size, int(s)), floor)
        elif isinstance(img_size, list):  # list i.e. img_size=[640, 480]
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
    def box_convert(x):
        # Convert boxes with shape [n, 4] from [x1, y1, x2, y2] to [x, y, w, h] where x1y1=top-left, x2y2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
        y[:, 2] = x[:, 2] - x[:, 0]  # width
        y[:, 3] = x[:, 3] - x[:, 1]  # height
        return y

    @staticmethod
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

    @staticmethod
    def plot_box_and_label(image, lw, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255),
                           font=cv2.FONT_HERSHEY_COMPLEX):
        """Add one xyxy box to image with label"""
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
        if label:
            tf = max(lw - 1, 1)  # font thickness
            w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
            outside = p1[1] - h - 3 >= 0  # label fits outside box
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(image, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), font, lw / 3, txt_color,
                        thickness=tf, lineType=cv2.LINE_AA)

    @staticmethod
    def font_check(font='./yolov6/utils/Arial.ttf', size=10):
        """Return a PIL TrueType Font, downloading to CONFIG_DIR if necessary"""
        assert osp.exists(font), f'font path not exists: {font}'
        try:
            return ImageFont.truetype(str(font) if font.exists() else font.name, size)
        except Exception as e:  # download if missing
            return ImageFont.truetype(str(font), size)

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
        classes = [0, 2, 3, 5, 7]
        """Process video with annotations and generate comparison video"""
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

        # Create output directory for this video
        output_dir = f"../output/{video_name}"
        os.makedirs(output_dir, exist_ok=True)

        # Set width and height
        width, height = self.img_size
        model_name = self.model_name

        print(f"Processing {model_name} on {video_name}...")

        # Track metrics
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

            # Get annotations for this frame
            frame_annotations = [entity for entity in annotations.get("entities", [])
                                 if entity.get("blob", {}).get("frame_idx") == frame_count]

            # Skip frames without annotations if needed
            # If you want to process all frames, remove this condition
            if not frame_annotations:
                frame_count += 1
                continue

            # Prepare image for YOLOv6
            img, img_src = self.process_image(frame, self.img_size, self.stride, self.half)
            img = img.to(self.device)
            if len(img.shape) == 3:
                img = img[None]  # expand for batch dim

            # Inference
            t1 = time.time()
            with torch.no_grad():
                pred_results = self.model(img)
            t2 = time.time()

            # Apply NMS
            det = non_max_suppression(pred_results, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]

            # Process predictions
            predictions = []
            if len(det):
                det[:, :4] = self.rescale(img.shape[2:], det[:, :4], frame.shape).round()

                # Convert to numpy array format
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
                x1, y1, x2, y2 = box

                # Make all YOLO boxes blue
                color = (255, 0, 0)  # Blue in BGR format
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"YOLO: {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Draw frame info and metrics
            accuracy = detected_annotations / total_annotations if total_annotations > 0 else 0
            inference_time = (t2 - t1) * 1000  # ms

            # Update FPS calculator
            fps_calculator.update(1.0 / (t2 - t1))
            avg_fps = fps_calculator.accumulate()

            # Draw info on frame
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Accuracy: {accuracy:.2f} ({detected_annotations}/{total_annotations})",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Model: {model_name}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Inference: {inference_time:.1f}ms", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
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
    """Process video for a specific model"""
    video_path, model_path, MODE = args

    if MODE == "personpath22":
        annotation_path = f"../dataset/personpath22/annotation/anno_visible_2022/{os.path.basename(video_path)}.json"
    else:
        annotation_path = f"../dataset/DETRAC_Upload/labels/annotations/{os.path.basename(video_path)}.json"

    # Skip if annotation doesn't exist
    evaluator = YOLOv6Evaluator(model_path)
    return evaluator.process_video(video_path, annotation_path)


def process_all_videos(MODE):
    # Create results directory
    os.makedirs("../results", exist_ok=True)

    # Get all MP4 files in the dataset directory
    if MODE == "personpath22":
        video_paths = glob.glob("../dataset/personpath22/raw_data/*.mp4")
    else:
        video_paths = glob.glob("../dataset/DETRAC_Upload/videos/*.mp4")

    print(f"Found {len(video_paths)} videos to process")

    # Set up multiprocessing
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
    MODE = "DETRAC" # DETRAC personpath22
    process_all_videos(MODE)