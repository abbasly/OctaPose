import cv2
from ultralytics import YOLO
from ultralytics.engine.results import Boxes



VIDEO_PATH = "/Users/anar/Desktop/tum/OctaPose/data/videos/conor.mp4"
DEVICE = "cpu"  # or 'cuda:0' if GPU is available

# Load YOLOv8 pose model
model = YOLO("yolov8n-pose.pt")

# Correct usage: pass custom tracker via tracker=
results = model.track(
    source=VIDEO_PATH,
    tracker="botsort.yaml",  # config file with re-ID enabled
    stream=True,
    persist=True,
    device=DEVICE,
    save=True
)


for res in results:
    boxes = res.boxes
    if boxes is not None and len(boxes.conf) > 3:
        # Get top 3 by confidence
        top3_indices = boxes.conf.argsort(descending=True)[:3]
        new_data = boxes.data[top3_indices]
        res.boxes = Boxes(new_data, boxes.orig_shape)
    frame = res.plot()
    cv2.imshow("YOLOv8 Pose Tracking with Re-ID", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
