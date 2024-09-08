import cv2
import torch
import numpy as np
import threading


name_video = "inpot_video1.mp4"

# شمارنده برای تعداد دفعات فراخوانی تابع
count = 0
# قفل برای همگام‌سازی دسترسی به شمارنده
lock = threading.Lock()

def load_points(file_path):
    points_dict = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        current_label = None
        current_points = []
        
        for line in lines:
            line = line.strip()
            if line.endswith(':'):  # Label line like "A:"
                if current_label:  # Save previous points
                    points_dict[current_label] = current_points
                current_label = line[:-1]  # Remove the colon
                current_points = []
            elif line:  # Points line
                point = tuple(map(int, line.split(',')))
                current_points.append(point)
        
        if current_label:  # Save the last points set
            points_dict[current_label] = current_points

    return points_dict

def draw_polygon(frame, points, color):
    cv2.polylines(frame, [np.array(points)], isClosed=True, color=color, thickness=5)

def call():
    print("Calling the function...")

def reset_count():
    global count
    with lock:
        count = 0
    print("Count reset")

def alarm():
    global count
    with lock:
        count += 1
        print("Alarm: Person entered the area! Count =", count)

        if count >= 5:
            call()
            count = 0

    # تنظیم تایمر برای بازنشانی شمارنده پس از ۳ ثانیه
    threading.Timer(3.0, reset_count).start()

def detect_and_alarm(frame, model, points_dict):
    results = model(frame)
    alarm_triggered = False
    for *box, conf, cls in results.xyxy[0]:
        if int(cls) == 0:  # Class 0 is person
            x1, y1, x2, y2 = map(int, box)
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

            # رسم کادر دور فرد
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # کادر سبز دور فرد

            # نمایش مختصات مرکزی فرد
            cv2.circle(frame, (center_x, center_y), radius=5, color=(255, 0, 0), thickness=-1)  # دایره آبی
            #cv2.putText(frame, f"({center_x}, {center_y})", (center_x + 10, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)  # مختصات

            # بررسی ورود فرد به منطقه‌های تعیین شده
            for points in points_dict.values():
                if cv2.pointPolygonTest(np.array(points), (center_x, center_y), False) >= 0:
                    alarm_triggered = True
                    alarm()
                    break
    return alarm_triggered

def process_video(input_video, output_video, points_dict):
    cap = cv2.VideoCapture(input_video)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    model = torch.hub.load('yolov5', 'custom', path='yolov5n6.pt', source='local')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        alarm_triggered = detect_and_alarm(frame, model, points_dict)
        color = (0, 0, 255) if alarm_triggered else (0, 255, 0)

        # رسم همه پلی‌گون‌ها
        for points in points_dict.values():
            draw_polygon(frame, points, color)
        
        out.write(frame)

    cap.release()
    out.release()

if __name__ == "__main__":
    input_video = name_video # نام فایل ویدیو
    output_video = 'output.mp4'
    points_dict = load_points('points.txt')
    process_video(input_video, output_video, points_dict)
