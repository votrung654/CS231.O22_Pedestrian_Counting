import numpy as np

def overlapping_area(detection_1, detection_2):
    # Tính diện tích giao nhau giữa hai hình chữ nhật và trả về IoU
    x1_tl, y1_tl, _, width1, height1 = detection_1
    x2_tl, y2_tl, _, width2, height2 = detection_2
    x1_br, y1_br = x1_tl + width1, y1_tl + height1
    x2_br, y2_br = x2_tl + width2, y2_tl + height2

    x_overlap = max(0, min(x1_br, x2_br) - max(x1_tl, x2_tl))
    y_overlap = max(0, min(y1_br, y2_br) - max(y1_tl, y2_tl))
    overlap_area = x_overlap * y_overlap
    total_area = width1 * height1 + width2 * height2 - overlap_area

    return overlap_area / float(total_area)

def nms(detections, Nt=0.0):
    # Non-maximum suppression để loại bỏ các phát hiện chồng chéo
    detections = sorted(detections, key=lambda detections: detections[2], reverse=True)
    N = len(detections)
    i = 0
    while i < N:
        j = i + 1
        while j < N:
            if overlapping_area(detections[i], detections[j]) > Nt:
                del detections[j]
                N -= 1
            else:
                j += 1
        i += 1

    return detections