import cv2
import numpy as np
import time
from collections import deque


CAM_INDEX = 0             
MIN_CONTOUR_AREA = 1500   
WARNING_PX = 100          
DANGER_PX = 10            
SMOOTH_QUEUE = 5          


# Virtual boundary (x1,y1) top-left and (x2,y2) bottom-right
# We'll draw a rectangle in the center of the frame — you can change this
def make_boundary(frame_w, frame_h):
    bw = int(frame_w * 0.4)
    bh = int(frame_h * 0.3)
    x1 = (frame_w - bw) 
    y1 = int(frame_h * 0.4) - bh    # * 0.55
    x2 = x1 + bw
    y2 = y1 + bh
    return (x1, y1, x2, y2)

def distance_to_rect(pt, rect):
    x, y = pt
    x1, y1, x2, y2 = rect
    
    if x1 <= x <= x2 and y1 <= y <= y2:
        return 0
    
    dx = max(x1 - x, 0, x - x2)
    dy = max(y1 - y, 0, y - y2)
    return int((dx*dx + dy*dy) ** 0.5)

def create_hsv_trackbars(window_name='hsv'):
    cv2.namedWindow(window_name)
    cv2.createTrackbar('Hmin', window_name, 0, 179, lambda x: None)
    cv2.createTrackbar('Hmax', window_name, 20, 179, lambda x: None)
    cv2.createTrackbar('Smin', window_name, 48, 255, lambda x: None)
    cv2.createTrackbar('Smax', window_name, 215, 255, lambda x: None)
    cv2.createTrackbar('Vmin', window_name, 80, 255, lambda x: None)
    cv2.createTrackbar('Vmax', window_name, 255, 255, lambda x: None)

def get_hsv_from_trackbars(window_name='hsv'):
    hmin = cv2.getTrackbarPos('Hmin', window_name)
    hmax = cv2.getTrackbarPos('Hmax', window_name)
    smin = cv2.getTrackbarPos('Smin', window_name)
    smax = cv2.getTrackbarPos('Smax', window_name)
    vmin = cv2.getTrackbarPos('Vmin', window_name)
    vmax = cv2.getTrackbarPos('Vmax', window_name)
    lower = np.array([hmin, smin, vmin], dtype=np.uint8)
    upper = np.array([hmax, smax, vmax], dtype=np.uint8)
    return lower, upper

# Segment hand by HSV skin color + morphology
def segment_hand_hsv(frame, lower, upper):
    blurred = cv2.GaussianBlur(frame, (7, 7), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    # morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask_open = cv2.morphologyEx(mask_close, cv2.MORPH_OPEN, kernel, iterations=2)  
    mask_dilate = cv2.dilate(mask_open, kernel, iterations=1)   #increase white region

    blur_hsv = np.append(blurred, hsv, axis=1)
    combined = np.append(mask_close, mask_open, axis=1)

    # print(blur_hsv.shape, combined.shape)
    cv2.imshow('blr_hsv', blur_hsv)
    cv2.imshow('mask_close_open', combined)
    # cv2.imshow('blr_hsv_close_open', np.append(blur_hsv.re, combined, axis=0))

    return mask_dilate

# Find largest contour and return contour, centroid
def largest_contour_and_centroid(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    # take largest by area
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < MIN_CONTOUR_AREA:
        return None, None
    M = cv2.moments(c)
    if M['m00'] == 0:
        return c, None
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return c, (cx, cy)

# Optional: fingertip candidate by finding farthest point from centroid along contour
def estimate_fingertip(contour, centroid):
    if centroid is None or contour is None:
        return None
    cx, cy = centroid
    # compute distances of contour points to centroid, find the farthest
    pts = contour.reshape(-1, 2)
    dists = np.linalg.norm(pts - np.array([cx, cy]), axis=1)
    idx = np.argmax(dists)
    return tuple(pts[idx])

def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    
    create_hsv_trackbars('hsv')

    fingertip_queue = deque(maxlen=SMOOTH_QUEUE)
    centroid_queue = deque(maxlen=SMOOTH_QUEUE)

    prev_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # print('Frame captured Status:', ret)
        frame = cv2.flip(frame, 1)  # mirror
        h, w = frame.shape[:2]
        # get user-tuned HSV
        lower, upper = get_hsv_from_trackbars('hsv')

        mask = segment_hand_hsv(frame, lower, upper)

        contour, centroid = largest_contour_and_centroid(mask)
        fingertip = estimate_fingertip(contour, centroid)

        # smoothing
        if centroid:
            centroid_queue.append(centroid)
            sx = int(sum([p[0] for p in centroid_queue]) / len(centroid_queue))
            sy = int(sum([p[1] for p in centroid_queue]) / len(centroid_queue))
            s_centroid = (sx, sy)
        else:
            s_centroid = None

        if fingertip:
            fingertip_queue.append(fingertip)
            fx = int(sum([p[0] for p in fingertip_queue]) / len(fingertip_queue))
            fy = int(sum([p[1] for p in fingertip_queue]) / len(fingertip_queue))
            s_fingertip = (fx, fy)
        else:
            s_fingertip = None

        # define boundary and compute distance (use fingertip if available, else centroid)
        boundary = make_boundary(w, h)
        keypoint = s_fingertip if s_fingertip is not None else s_centroid
        if keypoint is not None:
            dist = distance_to_rect(keypoint, boundary)
        else:
            dist = None

        # classify
        state = "NO HAND"
        color = (200, 200, 200)
        if dist is None:
            state = "NO HAND"
            color = (180, 180, 180)
        else:
            if dist == 0:
                state = "DANGER DANGER"
                color = (0, 0, 255)
            elif dist <= DANGER_PX:
                state = "DANGER DANGER"
                color = (0, 0, 255)
            elif dist <= WARNING_PX:
                state = "WARNING"
                color = (0, 165, 255)
            else:
                state = "SAFE"
                color = (0, 255, 0)

        # overlay
        # draw mask small window
        mask_col = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_col = cv2.resize(mask_col, (w//4, h//4))
        frame[0:h//4, 0:w//4] = mask_col



        # draw boundary
        x1, y1, x2, y2 = boundary
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        # draw keypoint
        if keypoint is not None:
            cv2.circle(frame, keypoint, 8, color, -1)
        # draw approx fingertip / contour
        if s_fingertip is not None:
            cv2.circle(frame, s_fingertip, 6, (255, 0, 0), -1)
        if contour is not None:
            cv2.drawContours(frame, [contour], -1, (100, 255, 100), 2)

        # draw state text
        cv2.putText(frame, f"State: {state}", (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)
        if dist is not None:
            cv2.putText(frame, f"Dist(px): {dist}", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1)

        # fps
        now = time.time()
        fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
        prev_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (w-140, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

        cv2.imshow("Hand Boundary Monitor", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q') :
            break
        elif key == ord('s'):
            # save sample for debugging
            cv2.imwrite("sample_frame.png", frame)
            cv2.imwrite("sample_mask.png", mask)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
