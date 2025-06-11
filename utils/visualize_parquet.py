import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import ast

FIGHTER_ID = 1
FILE_PATH = "../data/pose_tables/conor_fixed.parquet"

# Load Parquet data
df = pd.read_parquet(FILE_PATH)


# Safe parse if needed
def safe_parse(val):
    if isinstance(val, str):
        return ast.literal_eval(val)
    return val


df["pose_xy"] = df["pose_xy"].apply(safe_parse)
df["bbox_xyxy"] = df["bbox_xyxy"].apply(safe_parse)

# Filter track_id 1
df_track1 = (
    df[df["track_id"] == FIGHTER_ID].sort_values("frame_idx").reset_index(drop=True)
)

# Skeleton structure
skeleton = [
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 12),
    (5, 11),
    (6, 12),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (0, 5),
    (0, 6),
]

# Set up figure (larger size)
fig, ax = plt.subplots(figsize=(10, 6))  # Bigger figure
ax.set_xlim(0, 1920)
ax.set_ylim(1080, 0)
ax.set_aspect("equal")
ax.axis("off")

# Visual elements
bbox_patch = patches.Rectangle(
    (0, 0), 0, 0, linewidth=2, edgecolor="blue", facecolor="none"
)
ax.add_patch(bbox_patch)
(points,) = ax.plot([], [], "ro", markersize=6)  # Bigger keypoints
skeleton_lines = [ax.plot([], [], "g-", linewidth=2)[0] for _ in skeleton]


# Update function
def update(frame_idx):
    row = df_track1.iloc[frame_idx]
    pose_xy = row["pose_xy"]
    bbox = row["bbox_xyxy"]

    # Update bbox
    if bbox is not None and len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        bbox_patch.set_bounds(x1, y1, x2 - x1, y2 - y1)
        bbox_patch.set_visible(True)
        # Optional: zoom in on person
        ax.set_xlim(x1 - 50, x2 + 50)
        ax.set_ylim(y2 + 50, y1 - 50)
    else:
        bbox_patch.set_visible(False)
        ax.set_xlim(0, 1920)
        ax.set_ylim(1080, 0)

    # Update keypoints
    xs, ys = zip(*[(x, y) for x, y in pose_xy if x > 0 and y > 0])
    points.set_data(xs, ys)

    # Skeleton lines
    for line, (i, j) in zip(skeleton_lines, skeleton):
        if all(pose_xy[k][0] > 0 and pose_xy[k][1] > 0 for k in [i, j]):
            x = [pose_xy[i][0], pose_xy[j][0]]
            y = [pose_xy[i][1], pose_xy[j][1]]
            line.set_data(x, y)
        else:
            line.set_data([], [])

    # ax.set_title(f"Frame {row['frame_idx']}", fontsize=14)
    return [points, bbox_patch] + skeleton_lines


# Run animation
ani = FuncAnimation(
    fig, update, frames=len(df_track1), interval=50, blit=True, repeat=False
)
# ani.save("track1_animation.mp4", fps=20, dpi=200, codec="libx264") save optionally

plt.show()
