import cv2
import numpy as np
import imageio
from PIL import Image
import pandas as pd

import torch
from torchvision import transforms

import train, detect
from utils.metrics import bbox_iou

import itertools
import argparse
import os


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, help='initial weights path', required=False)
    # parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, help='dataset.yaml path', required=False)
    # parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=50, help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    # parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    # parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    # parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
    # parser.add_argument('--noplots', action='store_true', help='save no plot files')
    # parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    # parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    # parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    # parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')

    # Logger arguments
    parser.add_argument('--entity', default=None, help='Entity')
    parser.add_argument('--upload_dataset', nargs='?', const=True, default=False, help='Upload data, "val" option')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval')
    parser.add_argument('--artifact_alias', type=str, default='latest', help='Version of dataset artifact to use')

    # Saved model arguments
    parser.add_argument('--saved_path', type=str, help='Path to saved model weights', required=True)
    parser.add_argument('--data_dir', type=str, help='Path to data directory (not YAML file)', required=True)
    parser.add_argument('--pred_path', type=str, help='Path to detector predicted labels directory', required=True)


    # Inference settings
    parser.add_argument('--inf_confidence', type=float, default=0.5, help='Confidence threshold for cascade inference')


    return parser


class YOLOInferenceLoader(torch.utils.data.Dataset):
    def __init__(self, directory):
        super().__init__()
        self.directory = directory
        self.file_list = []
        for f in os.listdir(directory):
            if f.endswith('.jpg'):
                self.file_list.append(f)

    def __getitem__(self, idx):
        # image = imageio.imread(os.path.join(self.directory, self.file_list[idx]))
        # image = Image.fromarray(image, mode='RGB')
        # image = transforms.ToTensor()(image)
        # print(image.size())

        image = Image.open(os.path.join(self.directory, self.file_list[idx]))
        return image

    def __len__(self):
        return  len(self.file_list)   


def pipeline():
    args = parse_arguments().parse_args()
    # train_yolo(data=args.data, imgsz=args.imgsz, weights=args.weights, device=args.device)
    # stage_one_inference(model_path=args.saved_path, data_path=args.data_dir, yaml_path=args.data)
    # find_incorrect_images(ground_truth_path=args.data_dir, preds_path=args.pred_path)
    background_images, foreground_images, all_images = split_images(training_data_path=args.data_dir)
    metadata, confidences, files_to_resample, fp_files, fn_files = get_metadata(ground_truth_path=args.data_dir, preds_path=args.pred_path)
    generate_image_weights(metadata, confidences, files_to_resample, fp_files, fn_files, background_images, foreground_images, all_images)


def train_yolo(data, imgsz, weights, device):
    train.run(data=data, imgsz=imgsz, weights=weights, device=device)


def stage_one_inference(model_path, data_path, yaml_path):
    detect.run(weights=model_path, source=data_path, data=yaml_path, save_txt=True, save_conf=True, nosave=True)
    # model = torch.hub.load('./', 'custom', path=model_path, source='local')

    # file_list = []
    # for f in os.listdir(data_path):
    #     if f.endswith('.jpg'):
    #         file_list.append(f)

    # images = []
    # for f in file_list:
    #     image = Image.open(os.path.join(data_path, f))
    #     images.append(image)

    # for idx in range(0, len(file_list), 10):
    #     df_final = pd.DataFrame()
    #     results = model(images[idx: idx+10], size=640)
    #     # results.save()
    #     for sub_idx in range(0, 10):
    #         df_results = results.pandas().xyxy[sub_idx]
    #         df_final = pd.concat([df_final, df_results])

    #     if os.path.isfile('C:/Users/kunjc/Desktop/LogoDetectionAndClassification/src/yolov5/results.csv'):
    #         df_final.to_csv('C:/Users/kunjc/Desktop/LogoDetectionAndClassification/src/yolov5/results.csv', mode='a', header=False)
    #     else:
    #         df_final.to_csv('C:/Users/kunjc/Desktop/LogoDetectionAndClassification/src/yolov5/results.csv', mode='w', header=True)


def split_images(training_data_path):
    img_files = [x.split('.')[0] for x in os.listdir(training_data_path) if x.endswith('.jpg')]
    txt_files = [x.split('.')[0] for x in os.listdir(training_data_path) if x.endswith('.txt')]

    background_images = [x + '.jpg' for x in list(set(img_files) - set(txt_files))]
    foreground_images = [x + '.jpg' for x in txt_files]
    all_images = background_images.extend(foreground_images)

    return background_images, foreground_images, all_images


def get_metadata(ground_truth_path, preds_path):
    train_files = [x.split('.')[0] for x in os.listdir(ground_truth_path) if x.endswith('.txt')]
    pred_files = [x.split('.')[0] for x in os.listdir(preds_path)]
    metadata = {}
    confidences = {}
    files_to_resample = []

    # detecting false positives - txt files in detections but not in ground truth = ignore
    fp_files = set(pred_files) - set(train_files)
    # detecting false negatives - txt files in ground truth but not in detections = higher resampling
    fn_files = set(train_files) - set(pred_files)

    for train_file in train_files:
        if train_file not in fn_files:
            train_metadata, detect_metadata = compute_intersection(os.path.join(ground_truth_path, train_file + '.txt'),
                                            os.path.join(preds_path, train_file + '.txt'))

            metadata[train_file] = {'train': train_metadata,
                                    'detection': detect_metadata}

            avg_confidence = 0
            for detection in metadata[train_file]['detection']:
                avg_confidence += detection['confidence']
            confidences[train_file] = avg_confidence / len(metadata[train_file]['detection'])
        
            if len(metadata[train_file]['train']) == len(metadata[train_file]['detection']):
                for detection in metadata[train_file]['detection']:
                    if detection['confidence'] < 0.5:
                        files_to_resample.append(train_file)
            else:
                files_to_resample.append(train_file)

    return metadata, confidences, files_to_resample, fp_files, fn_files


def generate_image_weights(metadata, confidences, files_to_resample, fp_files, fn_files, background_images, foreground_images, all_images):
    

def find_incorrect_images(ground_truth_path, preds_path):
    train_files = [x.split('.')[0] for x in os.listdir(ground_truth_path) if x.endswith('.txt')]
    pred_files = [x.split('.')[0] for x in os.listdir(preds_path)]

    # detecting false positives - txt files in detections but not in ground truth = ignore
    fp_files = set(pred_files) - set(train_files)
    # detecting false negatives - txt files in ground truth but not in detections = higher resampling
    fn_files = set(train_files) - set(pred_files)
    
    files_to_resample = []
    for train_file in train_files:
        if train_file not in fn_files:
            train_metadata, detect_metadata = compute_intersection(os.path.join(ground_truth_path, train_file + '.txt'),
                                            os.path.join(preds_path, train_file + '.txt'))

            if len(train_metadata) == len(detect_metadata):
                for detection in detect_metadata:
                    if detection['confidence'] < 0.5:
                        files_to_resample.append(train_file)
            else:
                files_to_resample.append(train_file)

    # ~14% to resample
    print(len(files_to_resample)) 

    return files_to_resample


def compute_intersection(detection_file_1, detection_file_2, iou_threshold=0.5):
    # train files
    with open(detection_file_1) as f:
        lines = f.readlines()
        yolo1_detections = process_detections(lines, is_conf_present=False)
    
    # detection files
    with open(detection_file_2) as f:
        lines = f.readlines()
        yolo2_detections = process_detections(lines, is_conf_present=True)

    return yolo1_detections, yolo2_detections


def process_detections(lines, is_conf_present=True):
    detections = []
    for line in lines:
        if is_conf_present:
            class_id, x, y, w, h, confidence = line.split(" ")
            detections.append(
                {
                    "class_id": int(class_id),
                    "bbox": torch.tensor([float(x), float(y), float(w), float(h)]),
                    "confidence": float(confidence)
                }
            )

        else:
            class_id, x, y, w, h = line.split(" ")
            detections.append(
                {
                    "class_id": int(class_id),
                    "bbox": torch.tensor([float(x), float(y), float(w), float(h)]),
                }
            )

    return detections


if __name__ == "__main__":
    pipeline()