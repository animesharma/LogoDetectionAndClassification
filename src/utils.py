import numpy as np

def compute_iou(ground_truth: list, predicted: list) -> float:
    """
    Function to compute the IoU of two bounding boxes (format: x1, y1, x2, y2)
    Input params:
        - ground_truth : list of coordinates for ground truth bounding box
        - predicted : list of coordinates for predicted bounding box
    Returns : IoU of two bounding boxes
    """
    intersection_x_min = max(predicted[0], ground_truth[0])
    intersection_x_max = min(predicted[2], ground_truth[2])
    intersection_y_min = max(predicted[1], ground_truth[1])
    intersection_y_max = min(predicted[3], ground_truth[3])

    intersection_width = np.maximum((intersection_x_max - intersection_x_min + 1.), 0.)
    intersection_height = np.maximum((intersection_y_max - intersection_y_min + 1.), 0.)

    intersection_area = intersection_width * intersection_height

    union_area = (
        (predicted[2] - predicted[0] + 1.) * (predicted[3] - predicted[1] + 1.) +
        (ground_truth[2] - ground_truth[0] + 1.) * (ground_truth[3] - ground_truth[1] + 1.) -
        intersection_area
        )

    return intersection_area / union_area
