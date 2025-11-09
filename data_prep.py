# data_prep.py
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
import gc 

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

class FewShotDataPreprocessor:
    def __init__(self, dataset_root: str, output_root: str, resize_images: bool = False, resize_size: tuple = (640, 640)):
        self.dataset_root = Path(dataset_root)
        self.output_root = Path(output_root)
        self.annotations_path = self.dataset_root / "annotations" / "annotations.json"
        
        # Biến đếm tổng thể
        self.total_frames_processed = 0
        self.total_train_samples = 0
        self.total_val_samples = 0

        # Thống kê lỗi
        self.skipped_invalid_bboxes = 0
        self.skipped_corrupt_frames = 0

        # Resize option
        self.resize_images = resize_images
        self.resize_size = resize_size
    
    def load_annotations(self):
        """Load và parse annotations"""
        if not self.annotations_path.exists():
            logging.error(f"Annotations file not found: {self.annotations_path}")
            return []
        with open(self.annotations_path, 'r') as f:
            try:
                return json.load(f)
            except Exception as e:
                logging.error(f"Failed to load annotations.json: {e}")
                return []
    
    def extract_annotated_frames(self, video_path, target_video_id):
        """Trích xuất chính xác các frames có object từ annotations"""
        annotations = self.load_annotations()
        if not annotations:
            return []
        
        # Tìm annotations cho video hiện tại
        video_annotations = None
        for ann in annotations:
            if ann.get("video_id") == target_video_id:
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
        if not cap.isOpened():
            logging.error(f"Could not open video for extraction: {video_path}")
            return []
        
        frames_data = []
        frame_count = 0
        
        # Progress bar cho frame extraction
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
             logging.warning(f"Could not get frame count for {target_video_id}. Pbar disabled.")
             pbar = None
        else:
            pbar = tqdm(total=total_frames, desc=f"Extracting {target_video_id}", unit="frame", leave=False)
        
        while cap.isOpened():
            try:
                ret, frame = cap.read()
            except Exception as e:
                logging.error(f"Error reading frame {frame_count} from {target_video_id}: {e}")
                ret = False

            if not ret:
                break

            if frame_count in frames_with_objects:
                if frame is None or frame.size == 0:
                    self.skipped_corrupt_frames += 1
                else:
                    # Make a copy to avoid referencing the buffer that will be released
                    frames_data.append({
                        'frame': frame_count,
                        'image': frame.copy(),
                        'bboxes': self._get_bboxes_for_frame(video_annotations, frame_count)
                    })
            
            # free original frame
            del frame
            frame_count += 1
            if pbar:
                pbar.update(1)
        
        cap.release()
        gc.collect() 
        
        if pbar:
            pbar.close()
        
        self.total_frames_processed += len(frames_data)
        return frames_data
    
    def _get_bboxes_for_frame(self, video_annotations, frame_number):
        """Lấy tất cả bboxes cho frame cụ thể"""
        bboxes = []
        for ann_group in video_annotations.get("annotations", []):
            for bbox in ann_group.get("bboxes", []):
                if bbox.get("frame") == frame_number:
                    bboxes.append(bbox)
        return bboxes
    
    def create_few_shot_dataset(self, train_ratio: float = 0.8):
        """Tạo dataset cho few-shot detection với progress tracking"""
        start_time = time.time()
        logging.info("Starting few-shot dataset creation...")
        
        annotations = self.load_annotations()
        if not annotations:
            logging.error("No annotations loaded, aborting dataset creation.")
            return
        
        # Tạo thư mục
        (self.output_root / "images/train").mkdir(parents=True, exist_ok=True)
        (self.output_root / "images/val").mkdir(parents=True, exist_ok=True)
        (self.output_root / "labels/train").mkdir(parents=True, exist_ok=True)
        (self.output_root / "labels/val").mkdir(parents=True, exist_ok=True)
        (self.output_root / "references").mkdir(parents=True, exist_ok=True)
        
        
        # Chia danh sách các video, không theo các frame
        train_videos, val_videos = train_test_split(
            annotations, test_size=1-train_ratio, random_state=42, shuffle=True
        )
        
        logging.info(f"Splitting dataset by video: {len(train_videos)} train videos, {len(val_videos)} val videos.")
        
        # 1. Xử lý tập Train
        self._process_video_split(train_videos, "train")
        
        # 2. Xử lý tập Val
        self._process_video_split(val_videos, "val")
        
        # Tạo dataset.yaml
        self._create_dataset_yaml()
        
        end_time = time.time()
        duration = end_time - start_time
        
        logging.info(f"Few-shot dataset created in {duration:.2f} seconds:")
        logging.info(f"  Total annotated frames extracted: {self.total_frames_processed}")
        logging.info(f"  Train samples (frames): {self.total_train_samples}")
        logging.info(f"  Val samples (frames): {self.total_val_samples}")
        logging.info(f"  Skipped invalid bboxes: {self.skipped_invalid_bboxes}")
        logging.info(f"  Skipped corrupt frames: {self.skipped_corrupt_frames}")
        logging.info(f"  Output: {self.output_root}")

    def _process_video_split(self, video_list, split_name):
        """
        Hàm trợ giúp mới: Xử lý danh sách video cho một split (train/val)
        và lưu trực tiếp vào đĩa.
        """
        logging.info(f"Processing {split_name} split...")
        pbar_videos = tqdm(video_list, desc=f"Processing {split_name} videos", unit="video")
        
        split_sample_count = 0
        
        for video_ann in pbar_videos:
            video_id = video_ann.get("video_id")
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
            
            copied_refs = 0
            if reference_dir.exists():
                for ref_img in reference_dir.glob("*.*"):
                    if ref_img.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        try:
                            shutil.copy2(ref_img, ref_output_dir / ref_img.name)
                            copied_refs += 1
                        except Exception as e:
                            logging.warning(f"Failed to copy reference {ref_img}: {e}")
            else:
                logging.warning(f"Reference directory not found for {video_id}: {reference_dir}")
            
            if copied_refs == 0:
                logging.warning(f"No reference images copied for {video_id} (expected ~3 images).")
            
            # Extract annotated frames (chỉ cho 1 video, nhẹ nhàng)
            frames_data = self.extract_annotated_frames(video_path, video_id)
            
            # Thêm thông tin video
            for frame_data in frames_data:
                frame_data['video_id'] = video_id
            
            # Lưu các frame của video này vào đúng thư mục split
            saved_count = self._save_dataset_split(frames_data, split_name)
            
            split_sample_count += saved_count
            
            # Giải phóng RAM ngay sau khi xử lý xong 1 video
            del frames_data
            gc.collect()
        
        pbar_videos.close()
        
        # Cập nhật biến đếm tổng
        if split_name == "train":
            self.total_train_samples = split_sample_count
        else:
            self.total_val_samples = split_sample_count

    def _save_dataset_split(self, data, split_name):
        """
        Hàm này giờ sẽ xử lý một danh sách frame CỦA MỘT VIDEO
        và lưu chúng vào đĩa.
        Returns số lượng ảnh thực sự lưu được trong split.
        """
        saved_count = 0
        
        for entry in data:
            image = entry.get('image')
            if image is None:
                continue
            
            # Lưu image (resize nếu được bật)
            img_filename = f"{entry['video_id']}_{entry['frame']:06d}.jpg"
            img_path = self.output_root / "images" / split_name / img_filename

            try:
                if self.resize_images:
                    resized_image = cv2.resize(image, self.resize_size)
                    cv2.imwrite(str(img_path), resized_image)
                else:
                    cv2.imwrite(str(img_path), image)
            except Exception as e:
                logging.warning(f"Failed to write image {img_path}: {e}")
                continue
            
            # Lưu label (YOLO format)
            label_filename = f"{entry['video_id']}_{entry['frame']:06d}.txt"
            label_path = self.output_root / "labels" / split_name / label_filename
            
            h, w = image.shape[:2]
            valid_bboxes = 0
            try:
                with open(label_path, 'w') as f:
                    for bbox in entry.get('bboxes', []):
                        x1, y1, x2, y2 = bbox.get('x1'), bbox.get('y1'), bbox.get('x2'), bbox.get('y2')
                        # Kiểm tra hợp lệ
                        if x1 is None or y1 is None or x2 is None or y2 is None:
                            self.skipped_invalid_bboxes += 1
                            continue
                        if x2 <= x1 or y2 <= y1:
                            self.skipped_invalid_bboxes += 1
                            continue
                        
                        # Đảm bảo tọa độ không vượt quá ảnh
                        x1 = max(0, min(x1, w-1))
                        x2 = max(0, min(x2, w-1))
                        y1 = max(0, min(y1, h-1))
                        y2 = max(0, min(y2, h-1))
                        
                        cx = ((x1 + x2) / 2) / w
                        cy = ((y1 + y2) / 2) / h
                        bw = (x2 - x1) / w
                        bh = (y2 - y1) / h
                        
                        cx = max(0.0, min(1.0, cx))
                        cy = max(0.0, min(1.0, cy))
                        bw = max(0.0, min(1.0, bw))
                        bh = max(0.0, min(1.0, bh))
                        
                        f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                        valid_bboxes += 1
            except Exception as e:
                logging.warning(f"Failed to write label {label_path}: {e}")
                # Remove potentially partially written files
                if os.path.exists(label_path):
                    try:
                        os.remove(label_path)
                    except Exception:
                        pass
                continue
            
            # Giải phóng ảnh ngay sau khi lưu
            del entry['image']
            saved_count += 1
        
        return saved_count
        
    def _create_dataset_yaml(self):
        """Tạo file cấu hình dataset"""
        dataset_yaml = {
            'path': str(self.output_root),
            'train': 'images/train',
            'val': 'images/val',
            'nc': 1,
            'names': ['target_object']
        }
        
        try:
            with open(self.output_root / 'dataset.yaml', 'w') as f:
                yaml.dump(dataset_yaml, f, default_flow_style=False)
            logging.info(f"Created dataset.yaml at {self.output_root / 'dataset.yaml'}")
        except Exception as e:
            logging.error(f"Failed to write dataset.yaml: {e}")
