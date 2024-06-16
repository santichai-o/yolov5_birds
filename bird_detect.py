import torch
import argparse
import time
import cv2
from pathlib import Path
from models.common import DetectMultiBackend
from utils.dataloaders import LoadStreams
from utils.general import check_img_size, non_max_suppression, scale_boxes, print_args
from utils.torch_utils import select_device
from ultralytics.utils.plotting import Annotator

def draw_cross(im, center, color=(0, 0, 255), thickness=2, size=10):
    center_x, center_y = center
    cv2.line(im, (center_x - size, center_y), (center_x + size, center_y), color, thickness)
    cv2.line(im, (center_x, center_y - size), (center_x, center_y + size), color, thickness)

def draw_label(im, label, xyxy, color=(0, 0, 255), thickness=2):
    x1, y1 = int(xyxy[0]), int(xyxy[1])
    cv2.putText(im, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

def run(
    weights='yolov5s.pt',  # model.pt path(s)
    source=0,  # file/dir/URL/glob, 0 for webcam
    imgsz=640,  # inference size (pixels)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    line_thickness=3,  # bounding box thickness (pixels)
    # half=False,  # use FP16 half-precision inference
    disp=True # show window
):
    # Convert source to string if it's not
    source = str(source)

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Convert imgsz to tuple if it's not already
    if isinstance(imgsz, int):
        imgsz = (imgsz, imgsz)

    # Dataloader
    dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
    bs = len(dataset)  # batch_size

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time.time()  # Start time
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        pred = model(im, augment=False, visualize=False)
        t2 = time.time()  # End time

        # Calculate and print the time taken for inference
        inference_time = t2 - t1
        print(f"Inference time: {inference_time:.3f} seconds")

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=[14], agnostic=False)  # class 14 is for birds

        # Process predictions
        for i, det in enumerate(pred):  # per image
            p, im0 = path[i], im0s[i].copy()
            p = Path(p)  # to Path

            if disp:
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Find the largest bounding box
                largest_box = None
                max_area = 0
                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = xyxy
                    area = (x2 - x1) * (y2 - y1)
                    if area > max_area:
                        max_area = area
                        largest_box = (xyxy, conf, cls)

                if largest_box:
                    # Draw the largest bounding box and add cross at its center
                    xyxy, conf, cls = largest_box
                    c = int(cls)  # integer class
                    label = f"{names[c]} {conf:.2f}"

                    # Calculate center of the largest bounding box
                    center_x = int((xyxy[0] + xyxy[2]) / 2)
                    center_y = int((xyxy[1] + xyxy[3]) / 2)

                    # Print the label and center coordinates in the console
                    if disp:
                        # annotator.box_label(xyxy, label, color=(0, 255, 0))
                        draw_label(im0, label, xyxy)
                        draw_cross(im0, (center_x, center_y))
                    else:
                        print(f"Detected: {label} at center ({center_x}, {center_y})")

            # Stream results
            if disp:
                # Draw cross at center
                h, w, _ = im0.shape
                draw_cross(im0, (w // 2, h // 2), color=(0, 255, 0))

                im0 = annotator.result()
                cv2.imshow(str(p), im0)

                if cv2.waitKey(1) == ord('q'):  # Press 'q' to quit
                    break

    if disp:
        cv2.destroyAllWindows()

def parse_opt():
    """Parses command-line arguments for YOLOv5 detection, setting inference options and model configurations."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default="yolov5s.pt", help="model path or triton URL")
    parser.add_argument("--source", type=str, default="0", help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--disp", default=False, action="store_true", help="show window")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt

def main(opt):
    """Executes YOLOv5 model inference with given options, checking requirements before running the model."""
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
