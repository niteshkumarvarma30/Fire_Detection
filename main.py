import torch
import cv2
import numpy as np
import os
import time
from datetime import datetime
import threading
from playsound import playsound

# Load YOLOv5 fire model (ensure fire.pt is in the current folder)
print("📦 Loading model...")
model = torch.hub.load('ultralytics/yolov5', 'custom', path='fire.pt', force_reload=True)
model.conf = 0.5  # Lowered confidence threshold to allow more detections

# Setup webcam
print("🎥 Initializing webcam...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():  
    print("❌ Webcam could not be opened. Try a different index.")
    exit()

os.makedirs("fire_frames", exist_ok=True)
last_alert_time = 0
alert_interval = 5  # seconds

# Sound alert function
def play_alert():
    try:
        playsound("alert.mp3")
    except Exception as e:
        print("🔇 Sound error:", e)

print("✅ Fire detection system started. Press 'q' to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read from webcam")
        continue

    # Optional: Resize for better speed
    # frame = cv2.resize(frame, (640, 480))

    # 🔍 Detection
    results = model([frame])
    detections = results.xyxy[0]

    fire_detected = False

    if detections.shape[0] == 0:
        print("🟡 No detections in this frame")
    else:
        for *xyxy, conf, cls in detections:
            if conf < 0.3:
                continue

            # Optional: Check specific class if multiple exist
            # if int(cls) != 0: continue

            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, " FIRE", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            fire_detected = True

    # 🔔 Alert and save
    current_time = time.time()
    if fire_detected and (current_time - last_alert_time) > alert_interval:
        print("🚨 FIRE DETECTED!")
        threading.Thread(target=play_alert).start()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fire_frames/fire_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        last_alert_time = current_time

    # 🖼️ Show frame
    cv2.imshow("🔥 Real-Time Fire Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
