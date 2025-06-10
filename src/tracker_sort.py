import cv2
from ultralytics import YOLO
import sys

sys.path.append("/Users/anar/Desktop/tum/OctaPose/sort")
from sort import Sort
import numpy as np
import random
import os

# Assign consistent colors for each track_id
track_colors = {}


def get_color(track_id):
    if track_id not in track_colors:
        # Random pastel-like color
        track_colors[track_id] = tuple(random.randint(100, 255) for _ in range(3))
    return track_colors[track_id]


# Input and output paths
input_video_path = "/Users/anar/Desktop/tum/OctaPose/data/videos/conor_vs_aldo.mp4"
video_name = os.path.splitext(os.path.basename(input_video_path))[0]
output_video_path = f"../data/output/{video_name}_tracked.mp4"


model = YOLO("yolov8n-pose.pt")  # or yolov8m-pose.pt for better accuracy
tracker = Sort()

cap = cv2.VideoCapture(input_video_path)
out = cv2.VideoWriter(
    output_video_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    30,
    (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)[0]

    detections = []

    if results.boxes is not None and results.keypoints is not None:
        boxes = results.boxes.xyxy.cpu().numpy()  # shape (N, 4)
        scores = results.boxes.conf.cpu().numpy()
        top_indices = np.argsort(scores)[-3:][::-1]  # top 3
        for i in top_indices:
            x1, y1, x2, y2 = boxes[i]
            conf = scores[i]
            detections.append([x1, y1, x2, y2, conf])

    if len(detections) > 0:
        dets_np = np.array(detections)
        tracks = tracker.update(dets_np)

        for track in tracks:
            x1, y1, x2, y2, track_id = map(int, track)
            color_id = get_color(track_id)
            cv2.putText(
                frame,
                f"Fighter {track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color_id,
                2,
            )
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_id, 2)

        # also draw pose keypoints
        kps = results.keypoints.xy.cpu().numpy()  # (N, 17, 2)
        for i, pose in enumerate(kps):
            for x, y in pose:
                cv2.circle(frame, (int(x), int(y)), 3, (255, 0, 0), -1)

    out.write(frame)
    cv2.imshow("Tracked Pose", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
