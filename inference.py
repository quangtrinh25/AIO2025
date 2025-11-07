import os
import json
import cv2
import torch
from super_gradients.training import models
from tqdm import tqdm # <--- BỔ SUNG THƯ VIỆN

# ==============================================================================
# Inference
# ==============================================================================

def generate_predictions(model_path: str, samples_dir: str, output_file: str, conf_threshold: float = 0.25):
    """Generate predictions for all videos"""
    
    print(f"\n{'='*70}")
    print("Generating Predictions")
    print(f"{'='*70}")
    
    # Load model
    best_model = os.path.join(os.path.dirname(model_path), 'drone_detection', 'ckpt_best.pth')
    if os.path.exists(best_model):
        model_path = best_model
    
    print(f"Loading model from: {model_path}")
    
    try:
        model = models.get('yolo_nas_s', num_classes=1, checkpoint_path=model_path)
    except:
        print("⚠ Could not load checkpoint, using base model")
        model = models.get('yolo_nas_s', num_classes=1)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    predictions = []
    
    # Process each video
    video_folders = [d for d in os.listdir(samples_dir) if os.path.isdir(os.path.join(samples_dir, d))]
    print(f"Found {len(video_folders)} videos to process")
    
    # THAY VÒNG LẶP FOR BẰNG TQDM (cho video)
    for video_id in tqdm(video_folders, desc="Processing videos", unit="video"):
        
        video_path = os.path.join(samples_dir, video_id)
        object_images_path = os.path.join(video_path, 'object_images')
        
        if not os.path.exists(object_images_path):
            predictions.append({'video_id': video_id, 'detections': []})
            continue
        
        images = sorted([f for f in os.listdir(object_images_path) if f.endswith('.jpg')])
        detections = []
        
        # THÊM TQDM CHO VÒNG LẶP BÊN TRONG (cho ảnh)
        # leave=False có nghĩa là thanh tiến trình này sẽ biến mất sau khi hoàn thành
        for img_file in tqdm(images, desc=f"Processing {video_id.ljust(20)}", leave=False, unit="frame"):
            try:
                frame_num = int(img_file.split('_')[1].split('.')[0])
            except (ValueError, IndexError):
                print(f"Skipping malformed file name: {img_file}")
                continue

            img_path = os.path.join(object_images_path, img_file)
            img = cv2.imread(img_path)
            
            if img is None:
                continue
            
            # Run inference
            result = model.predict(img, conf=conf_threshold)
            
            bboxes = []
            if hasattr(result.prediction, 'bboxes_xyxy') and result.prediction.bboxes_xyxy is not None:
                for bbox in result.prediction.bboxes_xyxy:
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
        
        predictions.append({'video_id': video_id, 'detections': detections})
    
    # Save predictions
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print(f"\n✓ Predictions saved to: {output_file}")