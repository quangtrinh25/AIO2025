import os
import json
import cv2
import shutil
import logging
import numpy as np
from pathlib import Path
import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time
import gc  # ✅ thêm để dọn bộ nhớ

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

class FewShotDataPreprocessor:
    def __init__(self, dataset_root: str, output_root: str):
        self.dataset_root = Path(dataset_root)
        self.output_root = Path(output_root)
        self.annotations_path = self.dataset_root / "annotations" / "annotations.json"
        
    def load_annotations(self):
        """Load và parse annotations"""
        with open(self.annotations_path, 'r') as f:
            return json.load(f)
    
    def extract_annotated_frames(self, video_path, target_video_id):
        """Trích xuất chính xác các frames có object từ annotations"""
        annotations = self.load_annotations()
        
        # Tìm annotations cho video hiện tại
        video_annotations = None
        for ann in annotations:
            if ann["video_id"] == target_video_id:
                video_annotations = ann
                break
        
        if not video_annotations:
            logging.warning(f"No annotations found for {target_video_id}")
            return []
        
        # Tập hợp tất cả frame numbers có object
        frames_with_objects = set()
        for ann_group in video_annotations.get("annotations", []):
            for bbox in ann_group.get("bboxes", []):
                frames_with_objects.add(bbox["frame"])
        
        # Extract frames từ video
        cap = cv2.VideoCapture(str(video_path))
        frames_data = []
        frame_count = 0
        
        # Progress bar cho frame extraction
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = tqdm(total=total_frames, desc=f"Extracting frames {target_video_id}", unit="frame")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # ⚙️ Nếu muốn nhẹ hơn, có thể resize
            # frame = cv2.resize(frame, (640, 360))

            if frame_count in frames_with_objects:
                frames_data.append({
                    'frame': frame_count,
                    'image': frame.copy(),
                    'bboxes': self._get_bboxes_for_frame(video_annotations, frame_count)
                })
            
            # ✅ Giải phóng frame sau mỗi vòng lặp
            del frame

            frame_count += 1
            pbar.update(1)
        
        # ✅ Giải phóng tài nguyên video
        cap.release()
        cv2.destroyAllWindows()
        gc.collect()  # ✅ Dọn RAM sau khi đọc xong video
        
        pbar.close()
        return frames_data
    
    def _get_bboxes_for_frame(self, video_annotations, frame_number):
        """Lấy tất cả bboxes cho frame cụ thể"""
        bboxes = []
        for ann_group in video_annotations.get("annotations", []):
            for bbox in ann_group.get("bboxes", []):
                if bbox["frame"] == frame_number:
                    bboxes.append(bbox)
        return bboxes
    
    def create_few_shot_dataset(self, train_ratio: float = 0.8):
        """Tạo dataset cho few-shot detection với progress tracking"""
        start_time = time.time()
        logging.info("Starting few-shot dataset creation...")
        
        annotations = self.load_annotations()
        
        # Tạo thư mục
        (self.output_root / "images/train").mkdir(parents=True, exist_ok=True)
        (self.output_root / "images/val").mkdir(parents=True, exist_ok=True)
        (self.output_root / "labels/train").mkdir(parents=True, exist_ok=True)
        (self.output_root / "labels/val").mkdir(parents=True, exist_ok=True)
        (self.output_root / "references").mkdir(parents=True, exist_ok=True)
        
        all_video_data = []
        
        # Progress bar cho video processing
        pbar_videos = tqdm(annotations, desc="Processing videos", unit="video")
        
        for video_ann in pbar_videos:
            video_id = video_ann["video_id"]
            pbar_videos.set_postfix({"video": video_id})
            
            video_dir = self.dataset_root / "samples" / video_id
            video_path = video_dir / "drone_video.mp4"
            reference_dir = video_dir / "object_images"
            
            if not video_path.exists():
                logging.warning(f"Video not found: {video_path}")
                continue
            
            # Copy reference images
            ref_output_dir = self.output_root / "references" / video_id
            ref_output_dir.mkdir(parents=True, exist_ok=True)
            
            if reference_dir.exists():
                for ref_img in reference_dir.glob("*.*"):
                    if ref_img.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        shutil.copy2(ref_img, ref_output_dir / ref_img.name)
            
            # Extract annotated frames
            frames_data = self.extract_annotated_frames(video_path, video_id)
            
            # Thêm thông tin video
            for frame_data in frames_data:
                frame_data['video_id'] = video_id
                frame_data['reference_dir'] = str(ref_output_dir)
            
            all_video_data.extend(frames_data)
            
            # ✅ Giải phóng RAM sau mỗi video
            del frames_data
            gc.collect()
        
        pbar_videos.close()
        
        # Split train/val
        if len(all_video_data) > 0:
            train_data, val_data = train_test_split(
                all_video_data, test_size=1-train_ratio, random_state=42
            )
        else:
            train_data, val_data = [], []
        
        # Lưu dataset với progress bar
        logging.info("Saving dataset splits...")
        self._save_dataset_split(train_data, "train")
        self._save_dataset_split(val_data, "val")
        
        # Tạo dataset.yaml
        self._create_dataset_yaml()
        
        end_time = time.time()
        duration = end_time - start_time
        
        logging.info(f"Few-shot dataset created in {duration:.2f} seconds:")
        logging.info(f"  Total annotated frames: {len(all_video_data)}")
        logging.info(f"  Train samples: {len(train_data)}")
        logging.info(f"  Val samples: {len(val_data)}")
        logging.info(f"  Output: {self.output_root}")
    
    def _save_dataset_split(self, data, split_name):
        """Lưu dataset split với progress bar"""
        pbar = tqdm(data, desc=f"Saving {split_name} split", unit="sample")
        
        for i, entry in enumerate(pbar):
            # Lưu image
            img_filename = f"{entry['video_id']}_{entry['frame']:06d}.jpg"
            img_path = self.output_root / "images" / split_name / img_filename
            cv2.imwrite(str(img_path), entry['image'])
            
            # Lưu label (YOLO format)
            label_filename = f"{entry['video_id']}_{entry['frame']:06d}.txt"
            label_path = self.output_root / "labels" / split_name / label_filename
            
            with open(label_path, 'w') as f:
                for bbox in entry['bboxes']:
                    # Convert to YOLO format
                    h, w = entry['image'].shape[:2]
                    x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
                    
                    cx = ((x1 + x2) / 2) / w
                    cy = ((y1 + y2) / 2) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    
                    # Clamp values
                    cx = max(0.0, min(1.0, cx))
                    cy = max(0.0, min(1.0, cy))
                    bw = max(0.0, min(1.0, bw))
                    bh = max(0.0, min(1.0, bh))
                    
                    f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            
            # ✅ Giải phóng ảnh đã lưu
            del entry['image']
            gc.collect()
            
            pbar.set_postfix({"current": entry['video_id']})
        
        pbar.close()
    
    def _create_dataset_yaml(self):
        """Tạo file cấu hình dataset"""
        dataset_yaml = {
            'path': str(self.output_root),
            'train': 'images/train',
            'val': 'images/val',
            'nc': 1,
            'names': ['target_object']
        }
        
        with open(self.output_root / 'dataset.yaml', 'w') as f:
            yaml.dump(dataset_yaml, f, default_flow_style=False)
