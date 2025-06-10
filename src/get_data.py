import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from ultralytics.engine.results import Keypoints
import cv2
import os

# --- CONFIG ---
VIDEO_PATH = "../data/videos/conor.mp4"
OUT_PATH   = "../data/pose_tables/conor.parquet"
DEVICE     = "cpu"  # or 'cuda:0' if you want and have GPU
TOP_K      = 3      # keep only top-K detections per frame

# --- Load model ---
model = YOLO("yolov8n-pose.pt")

# --- Run pose + tracking ---
results = model.track(
    source=VIDEO_PATH,
    tracker="botsort.yaml",
    stream=True,
    persist=True,
    device=DEVICE,
    save=True
)

# --- Collect per-frame pose data ---
records = []

for frame_idx, res in enumerate(results):
    if res.boxes is None or res.keypoints is None:
        continue

    boxes = res.boxes
    

    # Get top K by confidence
    confs = boxes.conf.cpu().numpy()
    keep_inds = confs.argsort()[::-1][:TOP_K]

    # top3_indices = boxes.conf.argsort(descending=True)[:3]
    # new_boxes = boxes.data[top3_indices]
    # new_keypoints = res.keypoints.data[top3_indices]

    # res.boxes = Boxes(new_boxes, boxes.orig_shape)
    # res.keypoint = Keypoints(new_keypoints, res.keypoints.orig_shape)

    kpts = res.keypoints

    for i in range(len(boxes)):
        bbox_xyxy = boxes.xyxy[i].cpu().numpy()            # (4,)
        pose_xy   = kpts.xy[i].cpu().numpy()               # (17, 2)
        pose_conf = kpts.data[i, :, 2].cpu().numpy()       # (17,)
        track_id  = int(boxes.id[i].cpu().item()) if boxes.id is not None else -1

        records.append({
            "video": os.path.basename(VIDEO_PATH),
            "frame_idx": frame_idx,
            "track_id": track_id,                      # ✅ ADDED: track ID
            "bbox_xyxy": bbox_xyxy.tolist(),
            "pose_xy": pose_xy.tolist(),
            "kp_conf": pose_conf.tolist(),
        })
    frame = res.plot()
    cv2.imshow("YOLOv8 Pose Tracking with Re-ID", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# --- Save to Parquet ---
table = pa.Table.from_pylist(records)
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
pq.write_table(table, OUT_PATH, compression="zstd")
print(f"✅ Saved {table.num_rows} rows to {OUT_PATH}")
