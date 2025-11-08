# metrics.py
import json
import numpy as np
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ==============================================================================
# Enhanced ST-IoU Metric for Drone Search-and-Rescue
# ==============================================================================

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate IoU between two bounding boxes [x1, y1, x2, y2]."""
    # Convert to absolute coordinates if needed
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0

    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def hungarian_matching(gt_boxes: List[List[float]], pred_boxes: List[List[float]], 
                      iou_threshold: float = 0.0) -> Tuple[float, int]:
    """
    Hungarian matching for multiple objects in frame.
    Returns: total_iou, num_matches
    """
    if not gt_boxes or not pred_boxes:
        return 0.0, 0
    
    # Create cost matrix (1 - IoU)
    cost_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
    for i, gt_box in enumerate(gt_boxes):
        for j, pred_box in enumerate(pred_boxes):
            iou = calculate_iou(gt_box, pred_box)
            cost_matrix[i, j] = 1 - iou  # Convert to cost
    
    # Apply Hungarian algorithm
    gt_indices, pred_indices = linear_sum_assignment(cost_matrix)
    
    total_iou = 0.0
    num_matches = 0
    
    for gt_idx, pred_idx in zip(gt_indices, pred_indices):
        iou = 1 - cost_matrix[gt_idx, pred_idx]  # Convert back to IoU
        if iou >= iou_threshold:
            total_iou += iou
            num_matches += 1
    
    return total_iou, num_matches


def _build_frame_bbox_index_from_gt(gt_item: Dict[str, Any]) -> Dict[int, List[List[float]]]:
    """
    Build mapping frame_id -> list of GT boxes for a single ground-truth video item.
    """
    frames = {}
    for ann in gt_item.get("annotations", []):
        for b in ann.get("bboxes", []):
            f = int(b["frame"])
            frames.setdefault(f, []).append([
                float(b["x1"]), float(b["y1"]), 
                float(b["x2"]), float(b["y2"])
            ])
    return frames


def _build_frame_bbox_index_from_pred(pred_item: Dict[str, Any]) -> Dict[int, List[List[float]]]:
    """
    Build mapping frame_id -> list of predicted boxes for a single prediction video item.
    """
    frames = {}
    for det in pred_item.get("detections", []):
        for b in det.get("bboxes", []):
            f = int(b["frame"])
            frames.setdefault(f, []).append([
                float(b["x1"]), float(b["y1"]), 
                float(b["x2"]), float(b["y2"])
            ])
    return frames


def calculate_st_iou_for_video(
    gt_item: Dict[str, Any], 
    pred_item: Dict[str, Any], 
    iou_threshold: float = 0.0,
    matching_strategy: str = "hungarian"  # "max" or "hungarian"
) -> float:
    """
    Compute ST-IoU for a single video with enhanced matching strategies.
    """
    gt_frames = _build_frame_bbox_index_from_gt(gt_item)
    pred_frames = _build_frame_bbox_index_from_pred(pred_item)

    # Only consider frames that have ground truth objects
    frames_with_objects = set(gt_frames.keys())
    
    if len(frames_with_objects) == 0:
        return 0.0

    total_iou = 0.0
    total_frames_considered = 0

    for frame in frames_with_objects:
        gt_boxes = gt_frames.get(frame, [])
        pred_boxes = pred_frames.get(frame, [])
        
        if not gt_boxes:
            continue
            
        if not pred_boxes:
            # No predictions for this frame -> contributes 0 to numerator
            total_frames_considered += 1
            continue

        if matching_strategy == "hungarian":
            # Use Hungarian matching for multiple objects
            frame_iou, num_matches = hungarian_matching(gt_boxes, pred_boxes, iou_threshold)
            
            # Normalize by number of GT objects in frame
            if len(gt_boxes) > 0:
                total_iou += frame_iou / len(gt_boxes)
                total_frames_considered += 1
                
        else:  # "max" strategy (original, for single object per frame)
            max_iou = 0.0
            for gt_box in gt_boxes:
                for pred_box in pred_boxes:
                    iou = calculate_iou(gt_box, pred_box)
                    if iou > max_iou:
                        max_iou = iou
            
            if max_iou >= iou_threshold:
                total_iou += max_iou
                total_frames_considered += 1

    # Avoid division by zero
    if total_frames_considered == 0:
        return 0.0
        
    st_iou = total_iou / total_frames_considered
    return float(st_iou)


def calculate_detection_metrics(predictions_file: str, ground_truth_file: str, 
                              iou_threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculate comprehensive detection metrics including mAP, precision, recall.
    """
    start_time = time.time()
    logging.info("Starting metrics calculation...")
    
    with open(predictions_file, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    with open(ground_truth_file, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    # Build mappings
    gt_by_video = {}
    pbar_build = tqdm(ground_truth, desc="Building GT index", unit="video")
    for item in pbar_build:
        vid = item.get("video_id")
        if vid not in gt_by_video:
            gt_by_video[vid] = {"video_id": vid, "annotations": []}
        if "annotations" in item and isinstance(item["annotations"], list):
            gt_by_video[vid]["annotations"].extend(item["annotations"])
    pbar_build.close()

    pred_by_video = {}
    pbar_build = tqdm(predictions, desc="Building predictions index", unit="video")
    for item in pbar_build:
        vid = item.get("video_id")
        if vid not in pred_by_video:
            pred_by_video[vid] = {"video_id": vid, "detections": []}
        if "detections" in item and isinstance(item["detections"], list):
            pred_by_video[vid]["detections"].extend(item["detections"])
    pbar_build.close()

    # Calculate metrics per video
    st_iou_scores = []
    video_ids = list(gt_by_video.keys())
    
    # Additional metrics
    total_gt_objects = 0
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0

    pbar_videos = tqdm(video_ids, desc="Calculating video metrics", unit="video")
    
    for vid in pbar_videos:
        pbar_videos.set_postfix({"video": vid})
        gt_item = gt_by_video[vid]
        pred_item = pred_by_video.get(vid, {"video_id": vid, "detections": []})
        
        # Calculate ST-IoU
        st_iou = calculate_st_iou_for_video(gt_item, pred_item, iou_threshold, "hungarian")
        st_iou_scores.append(st_iou)
        
        # Calculate detection metrics (simplified)
        gt_frames = _build_frame_bbox_index_from_gt(gt_item)
        pred_frames = _build_frame_bbox_index_from_pred(pred_item)
        
        for frame, gt_boxes in gt_frames.items():
            total_gt_objects += len(gt_boxes)
            pred_boxes = pred_frames.get(frame, [])
            
            if not pred_boxes:
                total_false_negatives += len(gt_boxes)
                continue
                
            # Simple matching: consider at least one detection as TP if IoU > threshold
            matched_gt = set()
            for pred_box in pred_boxes:
                for i, gt_box in enumerate(gt_boxes):
                    if i in matched_gt:
                        continue
                    if calculate_iou(gt_box, pred_box) >= iou_threshold:
                        matched_gt.add(i)
                        break
            
            total_true_positives += len(matched_gt)
            total_false_negatives += (len(gt_boxes) - len(matched_gt))
            total_false_positives += max(0, len(pred_boxes) - len(matched_gt))
    
    pbar_videos.close()

    # Calculate final metrics
    final_st_iou = float(np.mean(st_iou_scores)) if st_iou_scores else 0.0
    
    precision = total_true_positives / (total_true_positives + total_false_positives + 1e-8)
    recall = total_true_positives / (total_true_positives + total_false_negatives + 1e-8)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    end_time = time.time()
    metrics_time = end_time - start_time
    
    metrics = {
        "ST-IoU": final_st_iou,
        "Precision": float(precision),
        "Recall": float(recall),
        "F1-Score": float(f1_score),
        "True_Positives": total_true_positives,
        "False_Positives": total_false_positives,
        "False_Negatives": total_false_negatives,
        "Total_GT_Objects": total_gt_objects,
        "Videos_Processed": len(video_ids),
        "Evaluation_Time_Seconds": metrics_time
    }
    
    logging.info(f"Metrics calculation completed in {metrics_time:.2f} seconds")
    
    return metrics


def calculate_final_score(predictions_file: str, ground_truth_file: str, 
                         iou_threshold: float = 0.0) -> float:
    """
    Calculate the mean ST-IoU over all videos (main metric for competition).
    """
    metrics = calculate_detection_metrics(predictions_file, ground_truth_file, iou_threshold)
    return metrics["ST-IoU"]


def print_detailed_metrics(metrics: Dict[str, float]):
    """Print comprehensive metrics report."""
    print("\n" + "="*60)
    print("DRONE SEARCH-AND-RESCUE EVALUATION METRICS")
    print("="*60)
    print(f"ST-IoU (Spatio-Temporal): {metrics['ST-IoU']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall: {metrics['Recall']:.4f}")
    print(f"F1-Score: {metrics['F1-Score']:.4f}")
    print("\nDetection Statistics:")
    print(f"True Positives: {metrics['True_Positives']}")
    print(f"False Positives: {metrics['False_Positives']}")
    print(f"False Negatives: {metrics['False_Negatives']}")
    print(f"Total GT Objects: {metrics['Total_GT_Objects']}")
    print(f"Videos Processed: {metrics['Videos_Processed']}")
    print(f"Evaluation Time: {metrics['Evaluation_Time_Seconds']:.2f} seconds")
    print("="*60)


