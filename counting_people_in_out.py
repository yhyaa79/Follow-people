import cv2
import pandas as pd
import torch
import math
import numpy as np

name_video = "inpot_video1.mp4"


class Tracker:
    def __init__(self):
        self.center_points = {}  # Track object center points
        self.id_count = 0  # Unique ID count
        self.tracking_info = {}  # Dictionary to track movement across areas

    def update(self, objects_rect):
        objects_bbs_ids = []

        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 35:
                    self.center_points[id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            if not same_object_detected:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        self.center_points = new_center_points.copy()
        return objects_bbs_ids

def resize_coordinates(points, original_width, original_height, new_width, new_height):
    return [(int(x * new_width / original_width), int(y * new_height / original_height)) for x, y in points]

def point_in_polygon(x, y, polygon):
    return cv2.pointPolygonTest(np.array(polygon, np.int32), (x, y), False) >= 0

model = torch.hub.load('yolov5', 'custom', path='yolov5n6.pt', source='local')

original_area1 = [(721, 553), (781, 571), (671, 709), (582, 703)]
original_area2 = [(700, 551), (582, 697), (531, 653), (664, 514)]

original_frame_width = 1920
original_frame_height = 1080
new_frame_width = 1020
new_frame_height = 500

area1 = resize_coordinates(original_area1, original_frame_width, original_frame_height, new_frame_width, new_frame_height)
area2 = resize_coordinates(original_area2, original_frame_width, original_frame_height, new_frame_width, new_frame_height)

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture(name_video)

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

# Video writer to save the output
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_video = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, (new_frame_width, new_frame_height))

tracker = Tracker()
in_count = 0
out_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (new_frame_width, new_frame_height))
    results = model(frame)
    a = results.xyxy[0].cpu().numpy()
    px = pd.DataFrame(a).astype("float")

    rects = []
    for index, row in px.iterrows():
        x1, y1, x2, y2, d = int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[5])
        c = class_list[d]
        if 'person' in c:
            rects.append((x1, y1, x2 - x1, y2 - y1))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, str(c), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

    objects = tracker.update(rects)

    for (x, y, w, h, object_id) in objects:
        cx, cy = (x + x + w) // 2, (y + y + h) // 2

        # Check if the person is in any area
        if object_id not in tracker.tracking_info:
            if point_in_polygon(cx, cy, area1):
                tracker.tracking_info[object_id] = 'area1'
            elif point_in_polygon(cx, cy, area2):
                tracker.tracking_info[object_id] = 'area2'
        else:
            prev_area = tracker.tracking_info[object_id]
            if prev_area == 'area1' and point_in_polygon(cx, cy, area2):
                in_count += 1
                tracker.tracking_info[object_id] = 'area2'
            elif prev_area == 'area2' and point_in_polygon(cx, cy, area1):
                out_count += 1
                tracker.tracking_info[object_id] = 'area1'

    # Calculate people currently in the area
    people_in = in_count - out_count

    # Draw the areas and counts on the frame
    cv2.polylines(frame, [np.array(area1, np.int32)], True, (255, 0, 0), 2)
    cv2.putText(frame, '1', (504, 471), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)

    cv2.polylines(frame, [np.array(area2, np.int32)], True, (255, 0, 0), 2)
    cv2.putText(frame, '2', (466, 485), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)

    # Display the counts on the frame
    cv2.putText(frame, f'In: {in_count}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f'Out: {out_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f'People In: {people_in}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Save the frame to the output video
    out_video.write(frame)

    # Commented out live display of video
    # cv2.imshow("RGB", frame)
    # if cv2.waitKey(1) & 0xFF == 27:
    #     break

cap.release()
out_video.release()
cv2.destroyAllWindows()
