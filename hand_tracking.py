import cv2
import numpy as np
from handtracking_model import load_inference_graph, detect_objects, draw_box_on_image

MIN_CONTOUR_AREA = 1000
MAX_CONTOUR_AREA = 40000
WARNING_PX = 100          
DANGER_PX = 10 

cap = cv2.VideoCapture(0)
print('camera accessed SUCCESSFULLY!!!!')

detection_graph, sess = load_inference_graph()

def expand_bbox(xmin, ymin, xmax, ymax, pad, img_width, img_height):
    xmin_new = max(0, xmin - pad)
    ymin_new = max(0, ymin - (pad * 3))
    xmax_new = min(img_width - 1, xmax + pad)
    ymax_new = min(img_height - 1, ymax + pad)
    return xmin_new, ymin_new, xmax_new, ymax_new

def max_contour_area(x):
    area = cv2.contourArea(x)
    if MIN_CONTOUR_AREA < area < MAX_CONTOUR_AREA:
        return area
    else:
        return 0
f = True


while True:
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

    ret, frame = cap.read()
    if not ret:
        print('Frame not read()')
        break
    
    frame = cv2.flip(frame, 1)  # mirror
    im_height, im_width, _ = frame.shape
    # image = cv2.imread('data/satyatma teertharu.jpg')
    boxes, scores = detect_objects(frame, detection_graph, sess)
    best_idx = np.argmax(scores)    
    ymin = int(boxes[best_idx][0]  * im_height) 
    xmin = int(boxes[best_idx][1]  * im_width)
    ymax = int(boxes[best_idx][2]  * im_height)
    xmax = int(boxes[best_idx][3]  * im_width)

    xmin, ymin, xmax, ymax = expand_bbox(xmin, ymin, xmax, ymax, pad=30, img_width=im_width, img_height=im_height)

    roi = frame[ymin:ymax, xmin:xmax]
    cv2.imshow('ROI', roi)
    lower = np.array([0, 20, 70], dtype=np.uint8)
    upper = np.array([20, 255, 255], dtype=np.uint8)
    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)

    dilate_4 = cv2.dilate(mask, np.ones((3,3), dtype=np.uint8), iterations=4)

    cont, _ = cv2.findContours(dilate_4, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print(f'Number of contours found: {len(cont)}')
    if len(cont) > 0:
        max_contour = max(cont, key=max_contour_area)
    
        cont_sorted = sorted(cont, key=max_contour_area, reverse=True)[:3]
        # max_contour = cont_sorted[0]
        for c in cont_sorted:
            for j in c:
                j[0][0] += xmin
                j[0][1] += ymin


        cv2.drawContours(frame, cont_sorted, 0, (0,255,0), 2)
        

        epsilon = 0.0005 * cv2.arcLength(max_contour, True)  
        # print(f'arc length: {cv2.arcLength(max_contour, True)}')
        approx = cv2.approxPolyDP(max_contour, epsilon, True)
        

        rect_x1 = int(im_width * 0.8)
        rect_y1 = 5
        rect_x2 = int(im_width - 5)
        rect_y2 = int(im_height * 0.2)

        cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (120, 20, 200),3)
        # cv2.addText(frame, f'ppt: {ppt}', (5,5),cv2.FONT_HERSHEY_PLAIN,)
        rect = np.array([
            [rect_x1, rect_y1],
            [rect_x2, rect_y1],
            [rect_x2, rect_y2],
            [rect_x1, rect_y2]
        ], dtype=np.float32).reshape((-1,1,2))

        min_dist = float('inf')
        min_pt = None
        for pt in max_contour:
            x, y = pt[0]

            dist = cv2.pointPolygonTest(rect, (float(x), float(y)), True)

            if abs(dist) < min_dist:
                min_dist = abs(dist)
                min_pt = (x, y)
        
        if min_pt is not None:
            cv2.circle(frame, min_pt, 7, (255,0,255), -1)
            cv2.putText(frame, f'dist: {int(min_dist)}', (min_pt[0]+10, min_pt[1]-10), cv2.FONT_HERSHEY_PLAIN, 1.5, (255,0,255), 2)


        defects = cv2.convexityDefects(approx, cv2.convexHull(approx, returnPoints=False))
        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(approx[s][0])
                end = tuple(approx[e][0])
                far = tuple(approx[f][0])
                cv2.circle(frame, far, 5, (60,250,180), -1)
                cv2.line(frame, start, end, (185,0,20), 2)
        
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
        
        cv2.putText(frame, f"State: {state}", (10, im_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)

    else:
        print('No contours found')

    draw_box_on_image(1, 0.2, scores, boxes, im_width, im_height, frame)
    cv2.imshow('Raama', frame)

cv2.destroyAllWindows()
