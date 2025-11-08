# inferences.py
import os
import json
import cv2
import torch
import logging
import numpy as np
from pathlib import Path
from super_gradients.training import models
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

class ReferenceGuidedInference:
    def __init__(self, model_path: str, similarity_threshold: float = 0.6):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.similarity_threshold = similarity_threshold
        
        # Load detection model
        logging.info(f"Loading model from {model_path}")
        self.detector = models.get(
            "yolo_nas_s",
            num_classes=1,
            checkpoint_path=model_path
        )
        self.detector.to(self.device)
        self.detector.eval()
        
        # Post-processing với threshold thấp để capture all potential objects
        self.post_processing = PPYoloEPostPredictionCallback(
            score_threshold=0.01,
            nms_threshold=0.5,
            nms_top_k=1000,
            max_predictions=300
        )
        
        # Feature extractor
        self.feature_extractor = self._create_feature_extractor()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]),
            transforms.Resize((224, 224))
        ])
        
        logging.info(f"Reference-guided inference initialized on {self.device}")

    def _create_feature_extractor(self):
        """Tạo feature extractor từ backbone của YOLO-NAS"""
        return self.detector.backbone

    def extract_features(self, image):
        """Trích xuất features từ image"""
        if isinstance(image, np.ndarray):
            image = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.feature_extractor(image)
            if isinstance(features, (list, tuple)):
                features = features[-1]
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features.view(features.size(0), -1)
            features = F.normalize(features, p=2, dim=1)
            
        return features.cpu().numpy()

    def load_reference_features(self, reference_dir):
        """Load và trích xuất features từ reference images"""
        reference_dir = Path(reference_dir)
        reference_images = []
        
        for img_path in reference_dir.glob("*.*"):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                image = cv2.imread(str(img_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                reference_images.append(image)
        
        if not reference_images:
            logging.warning(f"No reference images found in {reference_dir}")
            return None
        
        # Trích xuất features cho tất cả reference images
        ref_features = []
        pbar = tqdm(reference_images, desc="Extracting reference features", unit="image")
        
        for ref_img in pbar:
            features = self.extract_features(ref_img)
            ref_features.append(features)
        
        pbar.close()
        
        # Tính average feature vector
        ref_features = np.concatenate(ref_features, axis=0)
        avg_features = np.mean(ref_features, axis=0, keepdims=True)
        avg_features = avg_features / np.linalg.norm(avg_features)
        
        return avg_features

    def compute_similarity(self, features1, features2):
        """Tính cosine similarity"""
        return np.dot(features1, features2.T).item()

    def process_video(self, video_path: str, reference_dir: str):
        """Process video với reference-guided detection và progress tracking"""
        video_id = Path(video_path).parent.name
        logging.info(f"Processing video: {video_id} with reference: {reference_dir}")
        
        start_time = time.time()
        
        # Load reference features
        ref_features = self.load_reference_features(reference_dir)
        if ref_features is None:
            return {"video_id": video_id, "detections": []}
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Could not open video: {video_path}")
            return {"video_id": video_id, "detections": []}
        
        # Get video properties for progress bar
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        logging.info(f"Video info: {total_frames} frames, {fps:.2f} FPS")
        
        # Dictionary để lưu detections theo frame
        frame_detections = {}
        frame_count = 0
        
        # Progress bar cho frame processing
        pbar = tqdm(total=total_frames, desc=f"Processing {video_id}", unit="frame")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = self._preprocess_frame(frame_rgb)
            
            with torch.no_grad():
                predictions = self.detector(input_tensor)
                processed_predictions = self.post_processing(predictions)
            
            # Lọc detections dựa trên similarity với reference
            detections_this_frame = self._filter_detections_by_similarity(
                processed_predictions[0], frame_rgb, ref_features, frame_count
            )
            
            if detections_this_frame:
                frame_detections[frame_count] = detections_this_frame
            
            frame_count += 1
            pbar.update(1)
            pbar.set_postfix({
                "detections": len(detections_this_frame),
                "total_detections": sum(len(d) for d in frame_detections.values())
            })
        
        cap.release()
        pbar.close()
        
        # Tạo detection groups từ frame_detections
        detection_groups = self._create_detection_groups(frame_detections)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        logging.info(f"Completed {video_id}: {len(detection_groups)} detection groups in {processing_time:.2f}s")
        
        return {
            "video_id": video_id,
            "detections": detection_groups
        }

    def _preprocess_frame(self, frame):
        """Chuẩn bị frame cho model"""
        h, w = frame.shape[:2]
        input_size = 640
        
        scale = min(input_size / w, input_size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized_frame = cv2.resize(frame, (new_w, new_h))
        padded_frame = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
        padded_frame[:new_h, :new_w] = resized_frame
        
        padded_frame = padded_frame.astype(np.float32) / 255.0
        padded_frame = padded_frame.transpose(2, 0, 1)
        frame_tensor = torch.from_numpy(padded_frame).unsqueeze(0).to(self.device)
        
        return frame_tensor

    def _filter_detections_by_similarity(self, predictions, frame, ref_features, frame_number):
        """Lọc detections dựa trên similarity với reference features"""
        filtered_detections = []
        
        if predictions is None or len(predictions.prediction) == 0:
            return filtered_detections
        
        h, w = frame.shape[:2]
        bboxes = predictions.prediction[0]
        
        for bbox in bboxes:
            x1, y1, x2, y2, conf, cls_id = bbox.cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Đảm bảo coordinates hợp lệ
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w-1, x2), min(h-1, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Crop detected region
            detection_crop = frame[y1:y2, x1:x2]
            if detection_crop.size == 0:
                continue
            
            # Trích xuất features và tính similarity
            try:
                det_features = self.extract_features(detection_crop)
                similarity = self.compute_similarity(det_features, ref_features[0])
                
                # Chỉ giữ lại nếu similarity đủ cao
                if similarity >= self.similarity_threshold:
                    filtered_detections.append({
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2
                    })
                    
            except Exception as e:
                logging.warning(f"Feature extraction failed: {e}")
                continue
        
        return filtered_detections

    def _create_detection_groups(self, frame_detections):
        """
        Tạo detection groups từ frame_detections dictionary
        Mỗi group chứa một list các bboxes cho các frame liên tiếp
        """
        if not frame_detections:
            return []
        
        # Sắp xếp frames
        sorted_frames = sorted(frame_detections.keys())
        
        # Nhóm các frame liên tiếp thành intervals
        intervals = []
        current_interval = []
        
        for frame in sorted_frames:
            if not current_interval:
                current_interval.append(frame)
            else:
                if frame == current_interval[-1] + 1:
                    current_interval.append(frame)
                else:
                    intervals.append(current_interval)
                    current_interval = [frame]
        
        if current_interval:
            intervals.append(current_interval)
        
        # Tạo detection groups
        detection_groups = []
        for interval in intervals:
            bboxes_list = []
            for frame in interval:
                for bbox in frame_detections[frame]:
                    bboxes_list.append({
                        "frame": frame,
                        "x1": bbox["x1"],
                        "y1": bbox["y1"], 
                        "x2": bbox["x2"],
                        "y2": bbox["y2"]
                    })
            
            if bboxes_list:
                detection_groups.append({
                    "bboxes": bboxes_list
                })
        
        return detection_groups

def process_all_videos(test_samples_dir: str, model_path: str, output_json_path: str,
                      similarity_threshold: float = 0.6):
    """Process tất cả videos và tạo submission file với progress tracking"""
    
    start_time = time.time()
    logging.info(f"Starting inference on {test_samples_dir}")
    
    inference_engine = ReferenceGuidedInference(
        model_path=model_path,
        similarity_threshold=similarity_threshold
    )
    
    # Tìm tất cả video directories
    test_dir = Path(test_samples_dir)
    video_dirs = [d for d in test_dir.iterdir() if d.is_dir()]
    
    if not video_dirs:
        logging.error(f"No video directories found in {test_samples_dir}")
        return
    
    submission_data = []
    
    # Progress bar cho video processing
    pbar_videos = tqdm(sorted(video_dirs), desc="Processing videos", unit="video")
    
    for video_dir in pbar_videos:
        video_path = video_dir / "drone_video.mp4"
        reference_dir = video_dir / "object_images"
        
        pbar_videos.set_postfix({"video": video_dir.name})
        
        if not video_path.exists():
            logging.warning(f"Video not found: {video_path}")
            # Vẫn thêm vào submission với detections rỗng
            submission_data.append({
                "video_id": video_dir.name,
                "detections": []
            })
            continue
        
        if not reference_dir.exists():
            logging.warning(f"Reference images not found: {reference_dir}")
            submission_data.append({
                "video_id": video_dir.name, 
                "detections": []
            })
            continue
        
        try:
            result = inference_engine.process_video(str(video_path), str(reference_dir))
            submission_data.append(result)
        except Exception as e:
            logging.error(f"Error processing {video_dir.name}: {e}")
            submission_data.append({
                "video_id": video_dir.name,
                "detections": []
            })
    
    pbar_videos.close()
    
    # Lưu submission file
    with open(output_json_path, 'w') as f:
        json.dump(submission_data, f, indent=2)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    logging.info(f"Submission file saved: {output_json_path}")
    
    # Summary statistics
    total_videos = len(submission_data)
    videos_with_detections = sum(1 for item in submission_data if item["detections"])
    total_detection_groups = sum(len(item["detections"]) for item in submission_data)
    
    logging.info(f"Inference completed in {total_time:.2f} seconds:")
    logging.info(f"  Total videos: {total_videos}")
    logging.info(f"  Videos with detections: {videos_with_detections}")
    logging.info(f"  Total detection groups: {total_detection_groups}")

