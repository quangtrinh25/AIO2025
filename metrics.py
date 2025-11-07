import json
import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm # <--- BỔ SUNG THƯ VIỆN

# ==============================================================================
# ST-IoU Metric Implementation
# ==============================================================================

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate IoU between two bounding boxes [x1, y1, x2, y2]"""
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
    """Calculate Spatio-Temporal Intersection-over-Union (ST-IoU)"""
    gt_frames = set()
    pred_frames = set()
    
    for annotation in ground_truth.get('annotations', []):
        for box_data in annotation.get('bboxes', []):
            gt_frames.add(box_data['frame'])
    
    for detection in predictions.get('detections', []):
        for box_data in detection.get('bboxes', []):
            pred_frames.add(box_data['frame'])
    
    union_frames = gt_frames.union(pred_frames)
    
    if len(union_frames) == 0:
        return 0.0
    
    intersection_score = 0.0
    
    # --- Bọc vòng lặp này bằng tqdm ---
    for frame_id in union_frames:
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
        
        if len(gt_boxes) > 0 and len(pred_boxes) > 0:
            max_iou = 0.0
            for gt_box in gt_boxes:
                for pred_box in pred_boxes:
                    iou = calculate_iou(gt_box, pred_box)
                    if iou > max_iou:
                        max_iou = iou
            
            if max_iou >= iou_threshold:
                intersection_score += max_iou
    
    st_iou = intersection_score / len(union_frames)
    return st_iou


def calculate_final_score(predictions_file: str, ground_truth_file: str) -> float:
    """Calculate the final score as mean ST-IoU across all videos"""
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)
    
    with open(ground_truth_file, 'r') as f:
        ground_truth = json.load(f)
    
    gt_by_video = {}
    # --- Thêm tqdm vào vòng lặp này ---
    for item in tqdm(ground_truth, desc="Grouping Ground Truth", unit="item"):
        video_id = item['video_id']
        if video_id not in gt_by_video:
            gt_by_video[video_id] = {'video_id': video_id, 'annotations': []}
        gt_by_video[video_id]['annotations'].append(item)
    
    pred_by_video = {}
    # --- Thêm tqdm vào vòng lặp này ---
    for item in tqdm(predictions, desc="Grouping Predictions", unit="item"):
        video_id = item['video_id']
        if video_id not in pred_by_video:
            pred_by_video[video_id] = {'video_id': video_id, 'detections': []}
        pred_by_video[video_id]['detections'].append(item)
    
    st_iou_scores = []
    # --- Thêm tqdm vào vòng lặp chính ---
    print("Calculating ST-IoU score for each video...")
    for video_id in tqdm(gt_by_video.keys(), desc="Calculating ST-IoU", unit="video"):
        gt = gt_by_video[video_id]
        pred = pred_by_video.get(video_id, {'video_id': video_id, 'detections': []})
        
        # --- Bọc hàm calculate_st_iou bằng tqdm cho từng frame ---
        # (Đã thêm vào hàm calculate_st_iou ở trên)
        st_iou = calculate_st_iou(gt, pred)
        st_iou_scores.append(st_iou)
    
    final_score = np.mean(st_iou_scores) if st_iou_scores else 0.0
    return final_score