import cv2

def draw_cross(im, center, color=(0, 0, 255), thickness=1, size=10):
    center_x, center_y = center
    cv2.line(im, (center_x - size, center_y), (center_x + size, center_y), color, thickness)
    cv2.line(im, (center_x, center_y - size), (center_x, center_y + size), color, thickness)

def draw_label(im, label, xyxy, color=(0, 0, 255), thickness=1):
    x1, y1 = int(xyxy[0]), int(xyxy[1])
    cv2.putText(im, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

def draw_circle(im, center, color=(0, 0, 255), thickness=1, radius=10):
    cv2.circle(im, center, radius=radius, color=color, thickness=thickness )