import cv2
from ultralytics import YOLO


VIDEO_PATH = "../data/videos/conor_vs_aldo.mp4"
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
)


for res in results:
    frame = res.plot()
    cv2.imshow("YOLOv8 Pose Tracking with Re-ID", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
