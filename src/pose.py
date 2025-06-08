import cv2
import sys
import mediapipe as mp
from pathlib import Path

def view_and_save_pose(video_path: str, output_path: str, sample_rate: int = 1):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    paused = False

    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return

    # Get video info
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    out_fps = fps // sample_rate if sample_rate > 1 else fps

    # Create writer
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        out_fps,
        (width, height)
    )

    print("[INFO] Running pose estimation. Press 'space' to pause/resume, 'q' to quit.")

    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % sample_rate == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)

                if results.pose_landmarks:
                    drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                writer.write(frame)
                cv2.imshow("OctaPose – Annotated Video", frame)

            frame_count += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
            print("[Paused]" if paused else "[Resumed]")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Saved annotated video to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\n[!] Missing argument: video file path.\n")
        print("Usage:")
        print("    poetry run python src/pose.py <video_path> [output_path] [sample_rate]")
        print("\nExample:")
        print("    poetry run python src/pose.py data/videos/conor_vs_aldo.mp4 data/output/annotated_output.mp4 1\n")
        sys.exit(1)

    video_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "data/output/annotated_output.mp4"
    sample_rate = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    view_and_save_pose(video_path, output_path, sample_rate)