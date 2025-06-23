from ultralytics import YOLO
import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python yolo_pose_runner.py <path_to_video>")
        exit(1)

    video_path = sys.argv[1]

    model = YOLO("yolov8n-pose.pt")  # Make sure it's in the root dir or give full path
    results = model(
        video_path, save=True, show=True
    )  # Inference and save annotated video

    print("Inference done. Output saved in the 'runs/pose/' directory.")
