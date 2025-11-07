import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import cv2
import torch
from super_gradients.training import models
from super_gradients.training.dataloaders.dataloaders import (
    coco_detection_yolo_format_train, 
    coco_detection_yolo_format_val
)
from super_gradients.training import Trainer
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback

# ==============================================================================
# ST-IoU Metric Implementation
# ==============================================================================

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    Boxes format: [x1, y1, x2, y2]
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0
    
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def calculate_st_iou(ground_truth: Dict, predictions: Dict, iou_threshold: float = 0.5) -> float:
    """
    Calculate Spatio-Temporal Intersection-over-Union (ST-IoU).
    
    Args:
        ground_truth: Dict with 'video_id' and 'annotations' containing frame-wise boxes
        predictions: Dict with 'video_id' and 'detections' containing frame-wise boxes
        iou_threshold: IoU threshold for considering a match
    
    Returns:
        ST-IoU score for the video
    """
    # Get all frames from ground truth and predictions
    gt_frames = set()
    pred_frames = set()
    
    for annotation in ground_truth.get('annotations', []):
        for box_data in annotation.get('bboxes', []):
            gt_frames.add(box_data['frame'])
    
    for detection in predictions.get('detections', []):
        for box_data in detection.get('bboxes', []):
            pred_frames.add(box_data['frame'])
    
    # Union: all frames that belong to either ground-truth or predicted
    union_frames = gt_frames.union(pred_frames)
    
    if len(union_frames) == 0:
        return 0.0
    
    # Intersection: frames where predictions match ground-truth
    intersection_score = 0.0
    
    for frame_id in union_frames:
        # Get boxes for this frame
        gt_boxes = []
        for annotation in ground_truth.get('annotations', []):
            for box_data in annotation.get('bboxes', []):
                if box_data['frame'] == frame_id:
                    gt_boxes.append([
                        box_data['x1'], box_data['y1'], 
                        box_data['x2'], box_data['y2']
                    ])
        
        pred_boxes = []
        for detection in predictions.get('detections', []):
            for box_data in detection.get('bboxes', []):
                if box_data['frame'] == frame_id:
                    pred_boxes.append([
                        box_data['x1'], box_data['y1'], 
                        box_data['x2'], box_data['y2']
                    ])
        
        # Calculate frame-level IoU contribution
        if len(gt_boxes) > 0 and len(pred_boxes) > 0:
            # For each ground truth box, find best matching predicted box
            max_iou = 0.0
            for gt_box in gt_boxes:
                for pred_box in pred_boxes:
                    iou = calculate_iou(gt_box, pred_box)
                    if iou > max_iou:
                        max_iou = iou
            
            # If best match exceeds threshold, add to intersection
            if max_iou >= iou_threshold:
                intersection_score += max_iou
    
    # ST-IoU is intersection sum divided by union count
    st_iou = intersection_score / len(union_frames)
    
    return st_iou


def calculate_final_score(predictions_file: str, ground_truth_file: str) -> float:
    """
    Calculate the final score as mean ST-IoU across all videos.
    """
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)
    
    with open(ground_truth_file, 'r') as f:
        ground_truth = json.load(f)
    
    # Group by video_id
    gt_by_video = {}
    for item in ground_truth:
        video_id = item['video_id']
        if video_id not in gt_by_video:
            gt_by_video[video_id] = {'video_id': video_id, 'annotations': []}
        gt_by_video[video_id]['annotations'].append(item)
    
    pred_by_video = {}
    for item in predictions:
        video_id = item['video_id']
        if video_id not in pred_by_video:
            pred_by_video[video_id] = {'video_id': video_id, 'detections': []}
        pred_by_video[video_id]['detections'].append(item)
    
    # Calculate ST-IoU for each video
    st_iou_scores = []
    for video_id in gt_by_video.keys():
        gt = gt_by_video[video_id]
        pred = pred_by_video.get(video_id, {'video_id': video_id, 'detections': []})
        
        st_iou = calculate_st_iou(gt, pred)
        st_iou_scores.append(st_iou)
    
    # Final score is mean ST-IoU
    final_score = np.mean(st_iou_scores) if st_iou_scores else 0.0
    
    return final_score


# ==============================================================================
# Data Preparation
# ==============================================================================

def convert_to_yolo_format(dataset_path: str, output_path: str):
    """
    Convert the drone dataset to YOLO format for YOLO-NAS training.
    
    Expected structure:
    dataset/
        samples/
            drone_video_001/
                object_images/
                    img_1.jpg, img_2.jpg, ...
                drone_video.mp4
        annotations/
            annotations.json
    """
    annotations_file = os.path.join(dataset_path, 'annotations', 'annotations.json')
    
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    
    # Create output directories
    images_dir = os.path.join(output_path, 'images')
    labels_dir = os.path.join(output_path, 'labels')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # Process each video
    for annotation in annotations:
        video_id = annotation['video_id']
        video_path = os.path.join(dataset_path, 'samples', video_id)
        object_images_path = os.path.join(video_path, 'object_images')
        
        # Process each frame with bounding boxes
        for bbox_data in annotation['annotations']:
            for bbox in bbox_data['bboxes']:
                frame_num = bbox['frame']
                img_file = f"img_{frame_num}.jpg"
                src_img = os.path.join(object_images_path, img_file)
                
                if not os.path.exists(src_img):
                    continue
                
                # Read image to get dimensions
                img = cv2.imread(src_img)
                if img is None:
                    continue
                
                h, w = img.shape[:2]
                
                # Copy image
                dst_img = os.path.join(images_dir, f"{video_id}_frame_{frame_num}.jpg")
                cv2.imwrite(dst_img, img)
                
                # Convert bbox to YOLO format (normalized center_x, center_y, width, height)
                x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
                
                center_x = ((x1 + x2) / 2) / w
                center_y = ((y1 + y2) / 2) / h
                bbox_w = (x2 - x1) / w
                bbox_h = (y2 - y1) / h
                
                # Write label file (class_id center_x center_y width height)
                # Assuming single class (drone) with class_id = 0
                label_file = os.path.join(labels_dir, f"{video_id}_frame_{frame_num}.txt")
                with open(label_file, 'a') as lf:
                    lf.write(f"0 {center_x} {center_y} {bbox_w} {bbox_h}\n")
    
    # Create data.yaml
    data_yaml = f"""
train: {images_dir}
val: {images_dir}
nc: 1
names: ['drone']
"""
    
    with open(os.path.join(output_path, 'data.yaml'), 'w') as f:
        f.write(data_yaml)
    
    print(f"Dataset converted to YOLO format at {output_path}")


# ==============================================================================
# YOLO-NAS Training
# ==============================================================================

def train_yolo_nas(data_yaml_path: str, output_dir: str, num_epochs: int = 100):
    """
    Train YOLO-NAS model on the drone detection dataset.
    """
    # Initialize trainer
    trainer = Trainer(experiment_name='drone_detection', ckpt_root_dir=output_dir)
    
    # Load model - you can choose from yolo_nas_s, yolo_nas_m, yolo_nas_l
    model = models.get('yolo_nas_s', num_classes=1, pretrained_weights="coco")
    
    # Training parameters
    train_params = {
        'max_epochs': num_epochs,
        'lr_mode': 'cosine',
        'initial_lr': 5e-4,
        'lr_warmup_epochs': 3,
        'warmup_initial_lr': 1e-6,
        'optimizer': 'AdamW',
        'optimizer_params': {'weight_decay': 0.0001},
        'ema': True,
        'ema_params': {'decay': 0.9, 'decay_type': 'threshold'},
        'zero_weight_decay_on_bias_and_bn': True,
        'average_best_models': True,
        'mixed_precision': True,
        'loss': PPYoloELoss(num_classes=1, use_static_assigner=False, reg_max=16),
        'valid_metrics_list': [
            DetectionMetrics(
                post_prediction_callback=PPYoloEPostPredictionCallback(
                    score_threshold=0.01,
                    nms_top_k=1000,
                    max_predictions=300,
                    nms_threshold=0.7
                ),
                num_cls=1
            )
        ],
        'metric_to_watch': 'mAP@0.50',
    }
    
    # Load data
    train_data = coco_detection_yolo_format_train(
        dataset_params={
            'data_dir': os.path.dirname(data_yaml_path),
            'images_dir': 'images',
            'labels_dir': 'labels',
            'classes': ['drone']
        },
        dataloader_params={'batch_size': 16, 'num_workers': 2}
    )
    
    val_data = coco_detection_yolo_format_val(
        dataset_params={
            'data_dir': os.path.dirname(data_yaml_path),
            'images_dir': 'images',
            'labels_dir': 'labels',
            'classes': ['drone']
        },
        dataloader_params={'batch_size': 16, 'num_workers': 2}
    )
    
    # Train
    trainer.train(
        model=model,
        training_params=train_params,
        train_loader=train_data,
        valid_loader=val_data
    )
    
    print(f"Training complete. Model saved to {output_dir}")


# ==============================================================================
# Inference and Prediction Generation
# ==============================================================================

def generate_predictions(model_path: str, video_dir: str, output_file: str, conf_threshold: float = 0.25):
    """
    Generate predictions for all videos in the dataset.
    """
    # Load trained model
    model = models.get('yolo_nas_s', num_classes=1, checkpoint_path=model_path)
    model.eval()
    
    predictions = []
    
    # Process each video
    video_folders = [d for d in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, d))]
    
    for video_id in video_folders:
        video_path = os.path.join(video_dir, video_id)
        object_images_path = os.path.join(video_path, 'object_images')
        
        if not os.path.exists(object_images_path):
            # Add empty detection for videos without detections
            predictions.append({
                'video_id': video_id,
                'detections': []
            })
            continue
        
        # Get all images
        images = sorted([f for f in os.listdir(object_images_path) if f.endswith('.jpg')])
        
        detections = []
        
        for img_file in images:
            # Extract frame number
            frame_num = int(img_file.split('_')[1].split('.')[0])
            
            img_path = os.path.join(object_images_path, img_file)
            img = cv2.imread(img_path)
            
            if img is None:
                continue
            
            # Run inference
            result = model.predict(img, conf=conf_threshold)
            
            # Extract bounding boxes
            bboxes = []
            if result.prediction.bboxes_xyxy is not None:
                for bbox, conf, cls in zip(
                    result.prediction.bboxes_xyxy,
                    result.prediction.confidence,
                    result.prediction.labels
                ):
                    x1, y1, x2, y2 = bbox
                    bboxes.append({
                        'frame': frame_num,
                        'x1': float(x1),
                        'y1': float(y1),
                        'x2': float(x2),
                        'y2': float(y2)
                    })
            
            if bboxes:
                detections.append({'bboxes': bboxes})
        
        predictions.append({
            'video_id': video_id,
            'detections': detections
        })
    
    # Save predictions
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print(f"Predictions saved to {output_file}")


# ==============================================================================
# Main Pipeline
# ==============================================================================

def main():
    # Configuration - Windows paths
    BASE_DIR = r'D:\zalo_ai'
    TRAIN_PATH = r'D:\zalo_ai\observing\train'
    TEST_PATH = r'D:\zalo_ai\public_test'
    
    # Create output directories
    OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
    YOLO_FORMAT_PATH = os.path.join(OUTPUT_DIR, 'yolo_dataset')
    CHECKPOINTS_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
    PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, 'predictions')
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(YOLO_FORMAT_PATH, exist_ok=True)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    
    PREDICTIONS_FILE = os.path.join(PREDICTIONS_DIR, 'predictions.json')
    GROUND_TRUTH_FILE = os.path.join(TRAIN_PATH, 'annotations', 'annotations.json')
    
    # Step 1: Convert dataset to YOLO format
    print("Converting training dataset to YOLO format...")
    convert_to_yolo_format(TRAIN_PATH, YOLO_FORMAT_PATH)
    
    # Step 2: Train YOLO-NAS
    print("\nTraining YOLO-NAS model...")
    data_yaml = os.path.join(YOLO_FORMAT_PATH, 'data.yaml')
    train_yolo_nas(data_yaml, CHECKPOINTS_DIR, num_epochs=100)
    
    # Step 3: Generate predictions on training set (for validation)
    print("\nGenerating predictions on training set...")
    model_path = os.path.join(CHECKPOINTS_DIR, 'drone_detection', 'ckpt_best.pth')
    train_video_dir = os.path.join(TRAIN_PATH, 'samples')
    train_predictions_file = os.path.join(PREDICTIONS_DIR, 'train_predictions.json')
    generate_predictions(model_path, train_video_dir, train_predictions_file)
    
    # Step 4: Calculate ST-IoU score on training set
    print("\nCalculating ST-IoU score on training set...")
    final_score = calculate_final_score(train_predictions_file, GROUND_TRUTH_FILE)
    print(f"\nTraining ST-IoU Score: {final_score:.4f}")
    
    # Step 5: Generate predictions on test set
    print("\nGenerating predictions on test set...")
    test_video_dir = os.path.join(TEST_PATH, 'samples')
    test_predictions_file = os.path.join(PREDICTIONS_DIR, 'test_predictions.json')
    generate_predictions(model_path, test_video_dir, test_predictions_file)
    
    print(f"\n✓ All outputs saved to: {OUTPUT_DIR}")
    print(f"  - YOLO dataset: {YOLO_FORMAT_PATH}")
    print(f"  - Model checkpoints: {CHECKPOINTS_DIR}")
    print(f"  - Training predictions: {train_predictions_file}")
    print(f"  - Test predictions: {test_predictions_file}")


if __name__ == '__main__':
    main()