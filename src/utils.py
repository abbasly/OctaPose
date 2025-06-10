# utils.py


def filter_detections(boxes, scores, conf_thresh=0.5, min_area=5000, max_dets=3):
    """
    Filter detections by confidence and area, and return top N.
    """
    filtered = []

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        conf = scores[i]
        area = (x2 - x1) * (y2 - y1)

        if conf > conf_thresh and area > min_area:
            filtered.append([x1, y1, x2, y2, conf])

    # Sort and keep top N
    filtered.sort(key=lambda det: det[4], reverse=True)
    return filtered[:max_dets]
