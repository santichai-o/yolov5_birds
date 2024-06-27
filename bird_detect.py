import torch
import argparse
import time
import cv2
import serial
from pathlib import Path
from models.common import DetectMultiBackend
from utils.dataloaders import LoadStreams
from utils.general import (check_img_size, non_max_suppression, scale_boxes, print_args)
from utils.torch_utils import select_device
from ultralytics.utils.plotting import Annotator
from utils.draw import draw_cross, draw_label, draw_circle

# Define Arduino port
arduino_port = "/dev/tty.usbserial-0001"
arduino_baudrate = 9600

def initialize_serial(port=arduino_port, baudrate=arduino_baudrate):
    try:
        ser = serial.Serial(port, baudrate)
        time.sleep(2)  # Wait for the serial connection to initialize
        return ser
    except serial.SerialException as e:
        print(f"Failed to connect to serial port: {e}")
        return None

def send_angles_to_arduino(ser, angleX, angleY):
    try:
        if ser is not None and ser.is_open:
            ser.write(f'{angleX},{angleY}\n'.encode())
    except serial.SerialException as e:
        print(f"Failed to write to serial port: {e}")
        ser.close()
        ser = initialize_serial()  # Reinitialize serial connection
    return ser

def reset_angles(ser):
    """Reset the angles to 90,90."""
    angleX, angleY = 90, 90
    send_angles_to_arduino(ser, angleX, angleY)

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
    # Initialize serial connection to Arduino
    ser = initialize_serial()
    if ser is None:
        print("Serial connection could not be established. Exiting.")
        return

    angleX, angleY = 90, 90

    send_angles_to_arduino(ser, angleX, angleY)

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
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=[39], agnostic=False, max_det=3)  # class 14 is for birds
        # pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            p, im0 = path[i], im0s[i].copy()
            p = Path(p)  # to Path
            h, w, _ = im0.shape

            # Calculate angles to move the camera
            frame_center_x = w // 2
            frame_center_y = h // 2

            if disp:
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Find the largest detection
                max_area = 0
                max_det = None
                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = xyxy
                    area = (x2 - x1) * (y2 - y1)
                    if conf > 0.3 and area > max_area:
                        max_area = area
                        max_det = (x1, y1, x2, y2, conf, cls)

                if max_det:
                    x1, y1, x2, y2, conf, cls = max_det
                    c = int(cls)  # integer class
                    label = f"{names[c]} {conf:.2f}"

                    # Print the label and center coordinates in the console
                    if disp:
                        # annotator.box_label(xyxy, label, color=(0, 255, 0))
                        draw_label(im0, label, (int(x1), int(y1)))

                        # Calculate the center of the largest detection
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2

                        dx, dy = abs(frame_center_x - center_x), abs(frame_center_y - center_y)
                        if dx < 10 and dy < 10:
                            draw_circle(im0, (frame_center_x, frame_center_y), radius=15)
                        else:
                            draw_cross(im0, (int(center_x), int(center_y)))

                            # Calculate angles (this part may need adjustment based on your setup) (320, center_x 356.0), angleX 90, angleY 76 (320, center_x 155.0), angleX 90, angleY 75
                            if dx > 10:
                                if frame_center_x - center_x > 0 and angleX < 180:
                                    angleX += 1
                                elif frame_center_x - center_x < 0 and angleX > 0:
                                    angleX -= 1

                            # if dy < 20:
                            #     if frame_center_y - center_y > 0 and angleY < 180:
                            #         angleY -= 1
                            #     elif frame_center_y - center_y < 0 and angleY > 0:
                            #         angleY += 1

                            # print(f"frame_center_x ({frame_center_x}, center_x {center_x}), angleX {angleX}, angleY {angleY}")

                            send_angles_to_arduino(ser, angleX, angleY)

                            # time.sleep(0.5)
                    # else:
                    #     print(f"Detected: {label} at center ({center_x}, {center_y})")

            # Stream results
            if disp:
                # Draw cross at center
                draw_cross(im0, (frame_center_x, frame_center_y), color=(0, 255, 0))

                im0 = annotator.result()
                cv2.imshow(str(p), im0)

                if cv2.waitKey(1) == ord('q'):  # Press 'q' to quit
                    reset_angles(ser)

                    break

    if disp:
        cv2.destroyAllWindows()

    reset_angles(ser)  # Reset angles when exiting

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
