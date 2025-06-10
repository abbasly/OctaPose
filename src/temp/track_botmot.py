import sys
import cv2
import torch
import numpy as np
import os
print(f"--- Running script from: {os.path.abspath(__file__)} ---")
from ultralytics import YOLO


# Setup
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
VIDEO_PATH = "../data/videos/conor.mp4"
video_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
OUTPUT_PATH = f"../data/output/{video_name}_bot_sort_full.mp4"
print('asdfsaf')
# Load YOLOv8 pose model
model = YOLO("yolov8n-pose.pt").to(DEVICE)

min_box_area = 30000

sys.path.append("/Users/anar/Desktop/tum/OctaPose/BoT-SORT")
from tracker.bot_sort import BoTSORT

# Init BoT-SORT tracker (uses botsort.yaml if in default path)
tracker = BoTSORT()

# Video IO
cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(
    OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO pose detection
    results = model(frame, verbose=False)[0]

    detections = []
    if results.boxes is not None:
        boxes = results.boxes.xyxy.cpu().numpy()  # [N, 4]
        confs = results.boxes.conf.cpu().numpy()  # [N]
        clss = results.boxes.cls.cpu().numpy()    # [N]

        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        valid_indices = np.where(areas > min_box_area)[0]
        print(f"Passing {len(valid_indices)} detections to tracker", flush=True)
        # Sort by confidence and keep top 3
        # top_indices = np.argsort(confs)[-3:][::-1]  # top-3 indices
        if len(valid_indices) > 0:
            top_indices = valid_indices[np.argsort(confs[valid_indices])[-3:][::-1]]
            for i in top_indices:
                x1, y1, x2, y2 = boxes[i]
                conf = confs[i]
                cls = clss[i]
                detections.append([x1, y1, x2, y2, conf, cls])

    # Format for BoT-SORT: [x1, y1, x2, y2, conf, cls]
    dets_np = np.array(detections) if detections else np.empty((0, 6), dtype=float)

    # Run BoT-SORT tracker
    tracks = tracker.update(dets_np, frame)

    # Draw results
    for track in tracks:
        x1, y1, x2, y2 = map(int, track.tlbr)
        track_id = int(track.track_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Optionally draw pose keypoints
    if results.keypoints is not None:
        keypoints = results.keypoints.xy.cpu().numpy()
        for person in keypoints:
            for x, y in person:
                cv2.circle(frame, (int(x), int(y)), 2, (255, 0, 0), -1)

    out.write(frame)
    cv2.imshow("BoT-SORT Pose Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()