# metrics.py
import json
import numpy as np
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
import logging
import time
import math
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

EPS = 1e-8
MAX_PRED_PER_FRAME = 200  # limit preds per frame for performance

# ==============================================================================
# Enhanced ST-IoU Metric for Drone Search-and-Rescue (patched)
# - Hungarian matching used consistently for ST-IoU and detection counts (TP/FP/FN)
# - Validation of inputs and robust handling of videos without GT
# - Limit predictions per frame to protect performance
# - Optionally returns per-video breakdown in calculate_detection_metrics
# ==============================================================================


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate IoU between two bounding boxes [x1, y1, x2, y2]."""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0

    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    box1_area = max(0.0, (box1[2] - box1[0])) * max(0.0, (box1[3] - box1[1]))
    box2_area = max(0.0, (box2[2] - box2[0])) * max(0.0, (box2[3] - box2[1]))
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def hungarian_matching(gt_boxes: List[List[float]], pred_boxes: List[List[float]],
                       iou_threshold: float = 0.0) -> Tuple[float, int, List[Tuple[int, int]]]:
    """
    Hungarian matching for multiple objects in frame.
    Returns: total_iou (sum IoU over matched pairs), num_matches, matched_pairs list.
    """
    if not gt_boxes or not pred_boxes:
        return 0.0, 0, []

    ng = len(gt_boxes)
    npred = len(pred_boxes)
    cost_matrix = np.ones((ng, npred), dtype=np.float32)  # default cost 1 (IoU 0)

    for i, gt_box in enumerate(gt_boxes):
        for j, pred_box in enumerate(pred_boxes):
            iou = calculate_iou(gt_box, pred_box)
            cost_matrix[i, j] = 1.0 - iou  # cost = 1 - IoU

    gt_idx, pred_idx = linear_sum_assignment(cost_matrix)

    total_iou = 0.0
    matched_pairs = []
    num_matches = 0

    for gi, pj in zip(gt_idx, pred_idx):
        if gi < ng and pj < npred:
            iou = 1.0 - cost_matrix[gi, pj]
            if iou >= iou_threshold:
                total_iou += float(iou)
                matched_pairs.append((int(gi), int(pj)))
                num_matches += 1

    return float(total_iou), num_matches, matched_pairs


def _build_frame_bbox_index_from_gt(gt_item: Dict[str, Any]) -> Dict[int, List[List[float]]]:
    """
    Build mapping frame_id -> list of GT boxes for a single ground-truth video item.
    """
    frames = {}
    for ann in gt_item.get("annotations", []):
        for b in ann.get("bboxes", []):
            try:
                f = int(b["frame"])
                frames.setdefault(f, []).append([
                    float(b["x1"]), float(b["y1"]),
                    float(b["x2"]), float(b["y2"])
                ])
            except Exception:
                logging.warning(f"Invalid GT bbox entry skipped: {b}")
    return frames


def _build_frame_bbox_index_from_pred(pred_item: Dict[str, Any]) -> Dict[int, List[List[float]]]:
    """
    Build mapping frame_id -> list of predicted boxes for a single prediction video item.
    Each predicted bbox is expected to have keys x1,y1,x2,y2. Confidence, sim may be present but ignored here.
    """
    frames = {}
    for det in pred_item.get("detections", []):
        for b in det.get("bboxes", []):
            try:
                f = int(b["frame"])
                frames.setdefault(f, []).append([
                    float(b["x1"]), float(b["y1"]),
                    float(b["x2"]), float(b["y2"])
                ])
            except Exception:
                logging.warning(f"Invalid pred bbox entry skipped: {b}")
    return frames


def _limit_predictions(pred_boxes: List[List[float]], max_preds: int = MAX_PRED_PER_FRAME) -> List[List[float]]:
    """Limit number of predictions per frame by simple truncation (caller can sort by confidence beforehand if desired)."""
    if not pred_boxes:
        return []
    if len(pred_boxes) <= max_preds:
        return pred_boxes
    return pred_boxes[:max_preds]


def calculate_st_iou_for_video(
    gt_item: Dict[str, Any],
    pred_item: Dict[str, Any],
    iou_threshold: float = 0.0,
    matching_strategy: str = "hungarian"
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute ST-IoU for a single video with enhanced matching strategies.
    Returns (st_iou, per_video_stats)
    per_video_stats contains per-frame matches and counts for debugging.
    """
    gt_frames = _build_frame_bbox_index_from_gt(gt_item)
    pred_frames = _build_frame_bbox_index_from_pred(pred_item)

    frames_with_objects = sorted(list(set(gt_frames.keys())))

    per_video_stats = {
        "video_id": gt_item.get("video_id"),
        "frames_considered": len(frames_with_objects),
        "frame_details": {}
    }

    if len(frames_with_objects) == 0:
        # No GT frames: define behavior -> ST-IoU is 0.0 (can be changed if needed)
        per_video_stats["note"] = "no_ground_truth_frames"
        return 0.0, per_video_stats

    total_iou = 0.0
    total_frames_considered = 0

    for frame in frames_with_objects:
        gt_boxes = gt_frames.get(frame, [])
        pred_boxes = pred_frames.get(frame, [])
        pred_boxes = _limit_predictions(pred_boxes, MAX_PRED_PER_FRAME)

        frame_info = {"gt": len(gt_boxes), "pred": len(pred_boxes), "matched": 0, "sum_iou": 0.0}

        if not gt_boxes:
            per_video_stats["frame_details"][frame] = frame_info
            continue

        if not pred_boxes:
            total_frames_considered += 1
            per_video_stats["frame_details"][frame] = frame_info
            continue

        if matching_strategy == "hungarian":
            frame_iou_sum, num_matches, matched_pairs = hungarian_matching(gt_boxes, pred_boxes, iou_threshold)
            # Normalize contribution by number of GT objects (consistent with prior implementation)
            if len(gt_boxes) > 0:
                total_iou += frame_iou_sum / float(len(gt_boxes))
            total_frames_considered += 1
            frame_info["matched"] = num_matches
            frame_info["sum_iou"] = float(frame_iou_sum)
            frame_info["matched_pairs"] = matched_pairs

        else:  # "max" strategy (legacy single-object-per-frame)
            max_iou = 0.0
            for g in gt_boxes:
                for p in pred_boxes:
                    iou = calculate_iou(g, p)
                    if iou > max_iou:
                        max_iou = iou
            total_iou += max_iou
            total_frames_considered += 1
            frame_info["matched"] = 1 if max_iou >= iou_threshold else 0
            frame_info["sum_iou"] = float(max_iou)

        per_video_stats["frame_details"][frame] = frame_info

    if total_frames_considered == 0:
        return 0.0, per_video_stats

    st_iou = float(total_iou / float(total_frames_considered + EPS))
    per_video_stats["st_iou"] = st_iou
    return st_iou, per_video_stats


def calculate_detection_metrics(predictions_file: str, ground_truth_file: str,
                                iou_threshold: float = 0.5,
                                matching_strategy: str = "hungarian",
                                return_per_video: bool = False) -> Dict[str, Any]:
    """
    Calculate comprehensive detection metrics including ST-IoU, precision, recall.
    - iou_threshold: threshold for counting matched pairs as true positives
    - matching_strategy: "hungarian" or "max"
    - return_per_video: if True, include per-video breakdown
    """
    start_time = time.time()
    logging.info("Starting metrics calculation...")

    if not os.path.exists(predictions_file):
        raise FileNotFoundError(f"Predictions file not found: {predictions_file}")
    if not os.path.exists(ground_truth_file):
        raise FileNotFoundError(f"Ground-truth file not found: {ground_truth_file}")

    with open(predictions_file, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    with open(ground_truth_file, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    # Build indices by video_id
    gt_by_video = {}
    for item in ground_truth:
        vid = item.get("video_id")
        if vid is None:
            logging.warning("GT item without video_id skipped")
            continue
        if vid not in gt_by_video:
            gt_by_video[vid] = {"video_id": vid, "annotations": []}
        if "annotations" in item and isinstance(item["annotations"], list):
            gt_by_video[vid]["annotations"].extend(item["annotations"])

    pred_by_video = {}
    for item in predictions:
        vid = item.get("video_id")
        if vid is None:
            logging.warning("Prediction item without video_id skipped")
            continue
        if vid not in pred_by_video:
            pred_by_video[vid] = {"video_id": vid, "detections": []}
        if "detections" in item and isinstance(item["detections"], list):
            pred_by_video[vid]["detections"].extend(item["detections"])

    video_ids = sorted(list(gt_by_video.keys()))
    st_iou_scores = []
    per_video_reports = {}

    # Aggregate counts
    total_gt_objects = 0
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0

    pbar = tqdm(video_ids, desc="Calculating video metrics", unit="video")
    for vid in pbar:
        pbar.set_postfix({"video": vid})
        gt_item = gt_by_video.get(vid, {"video_id": vid, "annotations": []})
        pred_item = pred_by_video.get(vid, {"video_id": vid, "detections": []})

        st_iou, per_stats = calculate_st_iou_for_video(gt_item, pred_item, iou_threshold=0.0, matching_strategy=matching_strategy)
        st_iou_scores.append(st_iou)
        per_video_reports[vid] = per_stats

        # Count TP/FP/FN using Hungarian matching per frame for consistency
        gt_frames = _build_frame_bbox_index_from_gt(gt_item)
        pred_frames = _build_frame_bbox_index_from_pred(pred_item)

        for frame, gt_boxes in gt_frames.items():
            pred_boxes = pred_frames.get(frame, [])
            pred_boxes = _limit_predictions(pred_boxes, MAX_PRED_PER_FRAME)

            total_gt_objects += len(gt_boxes)

            if not pred_boxes:
                total_false_negatives += len(gt_boxes)
                continue

            # Hungarian to determine matched pairs at the chosen iou_threshold
            frame_iou_sum, num_matches, matched_pairs = hungarian_matching(gt_boxes, pred_boxes, iou_threshold)
            tp = num_matches
            fp = max(0, len(pred_boxes) - num_matches)
            fn = max(0, len(gt_boxes) - num_matches)

            total_true_positives += tp
            total_false_positives += fp
            total_false_negatives += fn

    pbar.close()

    final_st_iou = float(np.mean(st_iou_scores)) if st_iou_scores else 0.0

    precision = total_true_positives / (total_true_positives + total_false_positives + EPS)
    recall = total_true_positives / (total_true_positives + total_false_negatives + EPS)
    f1_score = 2 * (precision * recall) / (precision + recall + EPS)

    end_time = time.time()
    metrics_time = end_time - start_time

    metrics = {
        "ST-IoU": final_st_iou,
        "Precision": float(precision),
        "Recall": float(recall),
        "F1-Score": float(f1_score),
        "True_Positives": int(total_true_positives),
        "False_Positives": int(total_false_positives),
        "False_Negatives": int(total_false_negatives),
        "Total_GT_Objects": int(total_gt_objects),
        "Videos_Processed": len(video_ids),
        "Evaluation_Time_Seconds": metrics_time
    }

    logging.info(f"Metrics calculation completed in {metrics_time:.2f} seconds")

    if return_per_video:
        return {"metrics": metrics, "per_video": per_video_reports}
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
    print("\n" + "=" * 60)
    print("DRONE SEARCH-AND-RESCUE EVALUATION METRICS")
    print("=" * 60)
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
    print("=" * 60)
