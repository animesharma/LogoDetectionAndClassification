import os
import torch

from PIL import Image

from yolov5.utils.metrics import bbox_iou


"""
author: Timothy C. Arlen
date: 28 Feb 2018

Calculate Mean Average Precision (mAP) for a set of bounding boxes corresponding to specific
image Ids. Usage:

> python calculate_mean_ap.py

Will display a plot of precision vs recall curves at 10 distinct IoU thresholds as well as output
summary information regarding the average precision and mAP scores.

NOTE: Requires the files `ground_truth_boxes.json` and `predicted_boxes.json` which can be
downloaded fromt this gist.
"""

#from __future__ import absolute_import, division, print_function

from copy import deepcopy
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style('white')
sns.set_context('poster')

COLORS = [
    '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
    '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
    '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
    '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']


def calc_iou_individual(pred_box, gt_box):
    """Calculate IoU of single predicted and ground truth box

    Args:
        pred_box (list of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (list of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]

    Returns:
        float: value of the IoU for the two boxes.

    Raises:
        AssertionError: if the box is obviously malformed
    """
    x1_t, y1_t, x2_t, y2_t = gt_box
    x1_p, y1_p, x2_p, y2_p = pred_box

    if (x1_p > x2_p) or (y1_p > y2_p):
        raise AssertionError(
            "Prediction box is malformed? pred box: {}".format(pred_box))
    if (x1_t > x2_t) or (y1_t > y2_t):
        raise AssertionError(
            "Ground Truth box is malformed? true box: {}".format(gt_box))

    if (x2_t < x1_p or x2_p < x1_t or y2_t < y1_p or y2_p < y1_t):
        return 0.0

    far_x = np.min([x2_t, x2_p])
    near_x = np.max([x1_t, x1_p])
    far_y = np.min([y2_t, y2_p])
    near_y = np.max([y1_t, y1_p])

    inter_area = (far_x - near_x + 1) * (far_y - near_y + 1)
    true_box_area = (x2_t - x1_t + 1) * (y2_t - y1_t + 1)
    pred_box_area = (x2_p - x1_p + 1) * (y2_p - y1_p + 1)
    iou = inter_area / (true_box_area + pred_box_area - inter_area)
    return iou


def get_single_image_results(gt_boxes, pred_boxes, iou_thr):
    """Calculates number of true_pos, false_pos, false_neg from single batch of boxes.

    Args:
        gt_boxes (list of list of floats): list of locations of ground truth
            objects as [xmin, ymin, xmax, ymax]
        pred_boxes (dict): dict of dicts of 'boxes' (formatted like `gt_boxes`)
            and 'scores'
        iou_thr (float): value of IoU to consider as threshold for a
            true prediction.

    Returns:
        dict: true positives (int), false positives (int), false negatives (int)
    """

    all_pred_indices = range(len(pred_boxes))
    all_gt_indices = range(len(gt_boxes))
    if len(all_pred_indices) == 0:
        tp = 0
        fp = 0
        fn = len(gt_boxes)
        return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}
    if len(all_gt_indices) == 0:
        tp = 0
        fp = len(pred_boxes)
        fn = 0
        return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}

    gt_idx_thr = []
    pred_idx_thr = []
    ious = []
    for ipb, pred_box in enumerate(pred_boxes):
        for igb, gt_box in enumerate(gt_boxes):
            iou = calc_iou_individual(pred_box, gt_box)
            if iou > iou_thr:
                gt_idx_thr.append(igb)
                pred_idx_thr.append(ipb)
                ious.append(iou)

    args_desc = np.argsort(ious)[::-1]
    if len(args_desc) == 0:
        # No matches
        tp = 0
        fp = len(pred_boxes)
        fn = len(gt_boxes)
    else:
        gt_match_idx = []
        pred_match_idx = []
        for idx in args_desc:
            gt_idx = gt_idx_thr[idx]
            pr_idx = pred_idx_thr[idx]
            # If the boxes are unmatched, add them to matches
            if (gt_idx not in gt_match_idx) and (pr_idx not in pred_match_idx):
                gt_match_idx.append(gt_idx)
                pred_match_idx.append(pr_idx)
        tp = len(gt_match_idx)
        fp = len(pred_boxes) - len(pred_match_idx)
        fn = len(gt_boxes) - len(gt_match_idx)

    return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}


def calc_precision_recall(img_results):
    """Calculates precision and recall from the set of images

    Args:
        img_results (dict): dictionary formatted like:
            {
                'img_id1': {'true_pos': int, 'false_pos': int, 'false_neg': int},
                'img_id2': ...
                ...
            }

    Returns:
        tuple: of floats of (precision, recall)
    """
    true_pos = 0; false_pos = 0; false_neg = 0
    for _, res in img_results.items():
        true_pos += res['true_pos']
        false_pos += res['false_pos']
        false_neg += res['false_neg']

    try:
        precision = true_pos/(true_pos + false_pos)
    except ZeroDivisionError:
        precision = 0.0
    try:
        recall = true_pos/(true_pos + false_neg)
    except ZeroDivisionError:
        recall = 0.0

    return (precision, recall)

def get_model_scores_map(pred_boxes):
    """Creates a dictionary of from model_scores to image ids.

    Args:
        pred_boxes (dict): dict of dicts of 'boxes' and 'scores'

    Returns:
        dict: keys are model_scores and values are image ids (usually filenames)

    """
    model_scores_map = {}
    for img_id, val in pred_boxes.items():
        #print(img_id, val)
        for score in val['scores']:
            if score not in model_scores_map.keys():
                model_scores_map[score] = [img_id]
            else:
                model_scores_map[score].append(img_id)
    return model_scores_map

def get_avg_precision_at_iou(gt_boxes, pred_boxes, iou_thr=0.5):
    """Calculates average precision at given IoU threshold.

    Args:
        gt_boxes (list of list of floats): list of locations of ground truth
            objects as [xmin, ymin, xmax, ymax]
        pred_boxes (list of list of floats): list of locations of predicted
            objects as [xmin, ymin, xmax, ymax]
        iou_thr (float): value of IoU to consider as threshold for a
            true prediction.

    Returns:
        dict: avg precision as well as summary info about the PR curve

        Keys:
            'avg_prec' (float): average precision for this IoU threshold
            'precisions' (list of floats): precision value for the given
                model_threshold
            'recall' (list of floats): recall value for given
                model_threshold
            'models_thrs' (list of floats): model threshold value that
                precision and recall were computed for.
    """
    model_scores_map = get_model_scores_map(pred_boxes)
    sorted_model_scores = sorted(model_scores_map.keys())

    # Sort the predicted boxes in descending order (lowest scoring boxes first):
    for img_id in pred_boxes.keys():
        arg_sort = np.argsort(pred_boxes[img_id]['scores'])
        pred_boxes[img_id]['scores'] = np.array(pred_boxes[img_id]['scores'])[arg_sort].tolist()
        pred_boxes[img_id]['boxes'] = np.array(pred_boxes[img_id]['boxes'])[arg_sort].tolist()

    pred_boxes_pruned = deepcopy(pred_boxes)

    precisions = []
    recalls = []
    model_thrs = []
    img_results = {}
    # Loop over model score thresholds and calculate precision, recall
    for ithr, model_score_thr in enumerate(sorted_model_scores[:-1]):
        # On first iteration, define img_results for the first time:
        img_ids = gt_boxes.keys() if ithr == 0 else model_scores_map[model_score_thr]
        for img_id in img_ids:
            gt_boxes_img = gt_boxes[img_id]
            box_scores = pred_boxes_pruned[img_id]['scores']
            start_idx = 0
            for score in box_scores:
                if score <= model_score_thr:
                    pred_boxes_pruned[img_id]
                    start_idx += 1
                else:
                    break

            # Remove boxes, scores of lower than threshold scores:
            pred_boxes_pruned[img_id]['scores'] = pred_boxes_pruned[img_id]['scores'][start_idx:]
            pred_boxes_pruned[img_id]['boxes'] = pred_boxes_pruned[img_id]['boxes'][start_idx:]

            # Recalculate image results for this image
            img_results[img_id] = get_single_image_results(
                gt_boxes_img, pred_boxes_pruned[img_id]['boxes'], iou_thr)

        prec, rec = calc_precision_recall(img_results)
        precisions.append(prec)
        recalls.append(rec)
        model_thrs.append(model_score_thr)

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    prec_at_rec = []
    for recall_level in np.linspace(0.0, 1.0, 11):
        try:
            args = np.argwhere(recalls >= recall_level).flatten()
            prec = max(precisions[args])
        except ValueError:
            prec = 0.0
        prec_at_rec.append(prec)
    avg_prec = np.mean(prec_at_rec)

    print(sum(recalls) / len(recalls))

    return {
        'avg_prec': avg_prec,
        'precisions': precisions,
        'recalls': recalls,
        'model_thrs': model_thrs}


def plot_pr_curve(
    precisions, recalls, category='Person', label=None, color=None, ax=None):
    """Simple plotting helper function"""

    if ax is None:
        plt.figure(figsize=(10,8))
        ax = plt.gca()

    if color is None:
        color = COLORS[0]
    ax.scatter(recalls, precisions, label=label, s=20, color=color)
    ax.set_xlabel('recall')
    ax.set_ylabel('precision')
    ax.set_title('Precision-Recall curve for {}'.format(category))
    ax.set_xlim([0.0,1.3])
    ax.set_ylim([0.0,1.2])
    return ax

class InferenceNMS:
    def __init__(self) -> None:
        self.img_dim_dict = {}
        self.gt_file_ids = set()

    def _get_file_ids(self, infer_img_dir):
        """
        
        """
        return [file_name.split(".")[0] for file_name in next(os.walk(infer_img_dir), (None, None, []))[2]]

    def _get_all_detections(self, file_ids, yolo_det_dirs):
        """
        
        """
        id_bbox_dict = {}
        for file_id in file_ids:
            detections = []
            for yolo_dir in yolo_det_dirs:
                file_path = os.path.join(yolo_dir, file_id + ".txt")
                try:
                    with open(file_path, "r") as f:
                        lines = f.readlines()        
                    for line in lines:
                        class_id, x, y, w, h, confidence = line.split(" ")
                        detections.append(
                            {
                                "class_id": int(class_id),
                                "bbox": [float(x), float(y), float(w), float(h)],
                                "confidence": float(confidence)
                            }
                        )
                except:
                    continue
            id_bbox_dict[file_id] = detections
        return id_bbox_dict

    def _non_max_supression(self, boxes, conf_threshold=0.7, iou_threshold=0.4):
        """
        
        """
        bbox_list_thresholded = []
        bbox_list_new = []
        boxes_sorted = sorted(boxes, reverse=True, key = lambda x : x["confidence"])
        for box in boxes_sorted:
            if box["confidence"] > conf_threshold:
                bbox_list_thresholded.append(box)
        while len(bbox_list_thresholded) > 0:
            current_box = bbox_list_thresholded.pop(0)
            bbox_list_new.append(current_box)
            for box in bbox_list_thresholded:
                if current_box["class_id"] == box["class_id"]:
                    iou = bbox_iou(torch.unsqueeze(torch.tensor(current_box["bbox"]), 0), torch.unsqueeze(torch.tensor(box["bbox"]), 0))
                    if iou[0][0].item() > iou_threshold:
                        bbox_list_thresholded.remove(box)
        return bbox_list_new

    def _get_img_dim(self, file_ids, path):
        """
        
        """
        for file_id in file_ids:
            file_path = os.path.join(path, file_id + ".jpg")
            if os.path.exists(file_path):
                with Image.open(file_path) as img:
                    self.img_dim_dict[file_id] = img.size

    def _yolobbox2bbox(self, id, x, y, w, h):
        width, height = self.img_dim_dict[id]
        x1, y1 = int((x - w/2) * width), int((y - h/2) * height)
        x2, y2 = int((x + w/2) * width), int((y + h/2) * height)
        return x1, y1, x2, y2

    def _get_predictions(self, id_bbox_dict: dict):
        """
        
        """
        predictions = {}
        for id, boxes in id_bbox_dict.items():
            bbox_nms = self._non_max_supression(boxes, 0.25, 0.5)
            if id in self.gt_file_ids:
                boxes = []
                scores = []
                for item in bbox_nms:
                    x, y, w, h = item["bbox"]
                    boxes.append(list(self._yolobbox2bbox(id, x, y, w, h)))
                    scores.append(item["confidence"])
                predictions[id] = {"boxes": boxes, "scores": scores}
        return predictions

    def _get_gt(self, file_ids):
        """
        
        """
        gt = {}
        for file_id in file_ids:
            path = os.path.join("yolo_dataset", "test", file_id + ".txt")
            if os.path.exists(path):
                self.gt_file_ids.add(file_id)
                bboxes = []
                with open(path) as f:
                    lines = f.readlines()
                for line in lines:
                    x, y, w, h = map(float, line.strip().split(" ")[1:])
                    x1, y1, x2, y2 = self._yolobbox2bbox(file_id, x, y, w, h)
                    bboxes.append([x1, y1, x2, y2])     
                gt[file_id] = bboxes
        return gt

    def _run(self):
        """
        
        """
        gt_path = "./yolo_dataset/test/"
        file_ids = self._get_file_ids(gt_path)
        file_ids = list(set(file_ids))
        self._get_img_dim(file_ids, gt_path)
        id_bbox_dict = self._get_all_detections(file_ids, ["./src/yolov5/runs/detect/exp3/labels", "./src/yolov5/runs/detect/exp4/labels"])
        ground_truth = self._get_gt(file_ids)
        predictions = self._get_predictions(id_bbox_dict)
        

        # Runs it for one IoU threshold
        iou_thr = 0.7
        start_time = time.time()
        data = get_avg_precision_at_iou(ground_truth, predictions, iou_thr=iou_thr)
        end_time = time.time()
        print('Single IoU calculation took {:.4f} secs'.format(end_time - start_time))
        print('avg precision: {:.4f}'.format(data['avg_prec']))

        start_time = time.time()
        ax = None
        avg_precs = []
        iou_thrs = []
        for idx, iou_thr in enumerate(np.linspace(0.5, 0.95, 10)):
            data = get_avg_precision_at_iou(ground_truth, predictions, iou_thr=iou_thr)
            avg_precs.append(data['avg_prec'])
            iou_thrs.append(iou_thr)

            precisions = data['precisions']
            recalls = data['recalls']
            ax = plot_pr_curve(
                precisions, recalls, label='{:.2f}'.format(iou_thr), color=COLORS[idx*2], ax=ax)

        # prettify for printing:
        avg_precs = [float('{:.4f}'.format(ap)) for ap in avg_precs]
        iou_thrs = [float('{:.4f}'.format(thr)) for thr in iou_thrs]
        print('map: {:.2f}'.format(100*np.mean(avg_precs)))
        print('avg precs: ', avg_precs)
        print('iou_thrs:  ', iou_thrs)
        plt.legend(loc='upper right', title='IOU Thr', frameon=True)
        for xval in np.linspace(0.0, 1.0, 11):
            plt.vlines(xval, 0.0, 1.1, color='gray', alpha=0.3, linestyles='dashed')
        end_time = time.time()
        print('\nPlotting and calculating mAP takes {:.4f} secs'.format(end_time - start_time))
        plt.show()
    


InferenceNMS()._run()