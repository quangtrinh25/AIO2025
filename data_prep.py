import os
import json
import cv2

# ==============================================================================
# Data Preparation
# ==============================================================================

def convert_to_yolo_format(annotations_path: str, samples_path: str, output_path: str):
    annotations_file = os.path.join(annotations_path, 'annotations.json')
    print(f"Reading annotations from: {annotations_file}")
    
    if not os.path.exists(annotations_file):
        print(f"⚠ annotations.json NOT FOUND: {annotations_file}")
        return
    
    with open(annotations_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    print(f"Found {len(annotations)} video annotations\n")
    
    images_dir = os.path.join(output_path, 'images')
    labels_dir = os.path.join(output_path, 'labels')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    total_images = 0
    total_boxes = 0
    missing_frames = 0
    
    for video_ann in annotations:
        video_id = video_ann['video_id']
        
        # Hiệu chỉnh video_id nếu cần (ví dụ drone_video_01 -> drone_video_001)
        corrected_video_id = video_id.replace("01", "001")
        
        video_path = os.path.join(samples_path, corrected_video_id)
        video_file = os.path.join(video_path, 'drone_video.mp4')
        
        if not os.path.exists(video_file):
            print(f"⚠ Video file NOT FOUND: {video_file}")
            continue
        
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print(f"⚠ Could not open video: {video_file}")
            continue
        
        # Gom bboxes theo frame
        frames_dict = {}
        for frame_ann in video_ann.get('annotations', []):
            for bbox in frame_ann.get('bboxes', []):
                frame_num = bbox['frame']
                if frame_num not in frames_dict:
                    frames_dict[frame_num] = []
                frames_dict[frame_num].append(bbox)
        
        for frame_num, bboxes in sorted(frames_dict.items()):
            # Đặt cap đọc đúng frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if not ret or frame is None:
                print(f"⚠ Could not read frame {frame_num} in video {corrected_video_id}")
                missing_frames += 1
                continue
            
            h, w = frame.shape[:2]
            
            dst_name = f"{corrected_video_id}_frame_{frame_num}.jpg"
            dst_img = os.path.join(images_dir, dst_name)
            cv2.imwrite(dst_img, frame)
            
            label_file = os.path.join(labels_dir, f"{corrected_video_id}_frame_{frame_num}.txt")
            
            with open(label_file, 'a') as lf:
                for bbox in bboxes:
                    x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
                    center_x = ((x1 + x2) / 2) / w
                    center_y = ((y1 + y2) / 2) / h
                    bbox_w = (x2 - x1) / w
                    bbox_h = (y2 - y1) / h
                    lf.write(f"0 {center_x:.6f} {center_y:.6f} {bbox_w:.6f} {bbox_h:.6f}\n")
                    total_boxes += 1
            
            total_images += 1
        
        cap.release()
    
    print(f"\n✓ Converted videos: {len(annotations)}")
    print(f"✓ Created image files: {len(os.listdir(images_dir))}")
    print(f"✓ Created label files: {len(os.listdir(labels_dir))}")
    print(f"✓ Total bounding boxes: {total_boxes}")
    print(f"✓ Missing frames in videos: {missing_frames}")
    print(f"✓ Output directory: {output_path}")
