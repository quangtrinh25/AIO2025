import os
import numpy as np
import cv2
import torch
import gc
from torch.utils.data import DataLoader
import albumentations as A
from super_gradients.training import models
from super_gradients.training import Trainer
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback

# ==============================================================================
# Custom Collate Function (SỬA LỖI Ở ĐÂY)
# ==============================================================================
def yolo_nas_collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, 0)  # [B, 3, H, W]
    
    # Thêm batch index vào targets
    batch_targets = []
    for i, target in enumerate(targets):
        if len(target) > 0:
            # Thêm cột batch_index vào đầu: [batch_idx, class, x1, y1, x2, y2]
            batch_idx = torch.full((len(target), 1), i, dtype=target.dtype)
            target_with_batch = torch.cat([batch_idx, target], dim=1)
            batch_targets.append(target_with_batch)
    
    # Concatenate tất cả targets thành 1 tensor
    if len(batch_targets) > 0:
        batch_targets = torch.cat(batch_targets, 0)
    else:
        batch_targets = torch.zeros((0, 6))  # [batch_idx, class, x1, y1, x2, y2]
    
    return images, batch_targets                      # targets vẫn là list các bbox


# ==============================================================================
# Custom YOLO Dataset
# ==============================================================================

class DroneYOLODataset(torch.utils.data.Dataset):
    """Custom YOLO dataset for drone detection"""
    
    def __init__(self, images_dir, labels_dir, img_size=640, augment=False):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.img_size = img_size
        self.augment = augment
        
        # Get all image files
        self.image_files = []
        if os.path.exists(images_dir):
            self.image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"Found {len(self.image_files)} images in {images_dir}")
        
        # Augmentation
        if augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Blur(blur_limit=3, p=0.1),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        else:
            self.transform = None
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # (Nội dung hàm __getitem__ giữ nguyên...)
        # Load image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        h, w = image.shape[:2]
        
        # Load labels
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(self.labels_dir, label_name)
        
        boxes = []
        class_labels = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:])
                        boxes.append([cx, cy, bw, bh])
                        class_labels.append(class_id)
        
        # Apply augmentation
        if self.transform and len(boxes) > 0:
            '''boxes = np.array(boxes, dtype=np.float32)
            boxes[:, 0] = np.clip(boxes[:, 0], 0, 1)  # cx
            boxes[:, 1] = np.clip(boxes[:, 1], 0, 1)  # cy
            boxes[:, 2] = np.clip(boxes[:, 2], 0, 1)  # bw
            boxes[:, 3] = np.clip(boxes[:, 3], 0, 1)  # bh
            boxes = boxes.tolist()

            transformed = self.transform(image=image, bboxes=boxes, class_labels=class_labels)
            image = transformed['image']
            boxes = transformed['bboxes']
            class_labels = transformed['class_labels']'''
            boxes = np.array(boxes, dtype=np.float32)

    # Clip nhẹ hơn để tránh lỗi precision (vẫn giữ được bbox nhỏ)
            boxes[:, 0] = np.clip(boxes[:, 0], 1e-6, 1 - 1e-6)  # cx
            boxes[:, 1] = np.clip(boxes[:, 1], 1e-6, 1 - 1e-6)  # cy
            boxes[:, 2] = np.clip(boxes[:, 2], 1e-6, 1 - 1e-6)  # bw
            boxes[:, 3] = np.clip(boxes[:, 3], 1e-6, 1 - 1e-6)  # bh

            boxes = boxes.tolist()

            try:
                transformed = self.transform(image=image, bboxes=boxes, class_labels=class_labels)
                image = transformed['image']
                boxes = transformed['bboxes']
                class_labels = transformed['class_labels']
            except ValueError as e:
                print(f"⚠ Bỏ qua ảnh lỗi bbox: {self.image_files[idx]} ({e})")
                # Trả về ảnh trống (phải đảm bảo là Tensor)
                dummy_image = np.zeros((3, self.img_size, self.img_size), dtype=np.float32)
                return torch.from_numpy(dummy_image), torch.zeros((0, 5), dtype=torch.float32)

                
        # Resize image
        image = cv2.resize(image, (self.img_size, self.img_size))
        image = image.transpose(2, 0, 1) # HWC to CHW
        image = image.astype(np.float32) / 255.0
        
        # Convert to tensors
        image = torch.from_numpy(image)
        
        # Convert YOLO format to xyxy for training
        targets = []
        for (cx, cy, bw, bh), cls in zip(boxes, class_labels):
            x1 = (cx - bw/2) * self.img_size
            y1 = (cy - bh/2) * self.img_size
            x2 = (cx + bw/2) * self.img_size
            y2 = (cy + bh/2) * self.img_size
            targets.append([cls, x1, y1, x2, y2])

        if len(targets) == 0:
            targets = torch.zeros((0, 5), dtype=torch.float32)
        else:
            targets = torch.tensor(targets, dtype=torch.float32)

        return image, targets


# ==============================================================================
# Training Function
# ==============================================================================


def train_yolo_nas(yolo_data_path: str, output_dir: str, num_epochs: int = 50, use_pretrained: bool = False):

    torch.cuda.empty_cache()

    images_dir = os.path.join(yolo_data_path, 'images')
    labels_dir = os.path.join(yolo_data_path, 'labels')

    if not os.path.exists(images_dir) or len(os.listdir(images_dir)) == 0:
        raise ValueError(f"No images found in {images_dir}")

    print(f"\n{'='*70}")
    print("Starting YOLO-NAS Training (optimized for low VRAM GPUs)")
    print(f"{'='*70}")
    print(f"Images directory: {images_dir}")
    print(f"Labels directory: {labels_dir}")
    print(f"Number of images: {len(os.listdir(images_dir))}")
    print(f"Number of labels: {len(os.listdir(labels_dir))}")

    # ✅ Datasets
    train_dataset = DroneYOLODataset(images_dir, labels_dir, img_size=512, augment=True)
    val_dataset = DroneYOLODataset(images_dir, labels_dir, img_size=512, augment=False)

    # ✅ Dataloader (batch_size = 1, num_workers = 0 để giảm VRAM)
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        collate_fn=yolo_nas_collate_fn,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        collate_fn=yolo_nas_collate_fn,
        pin_memory=True
    )

    trainer = Trainer(experiment_name='drone_detection', ckpt_root_dir=output_dir)

    # ✅ Model nhỏ nhất
    try:
        if use_pretrained:
            print("Loading pretrained COCO weights...")
            model = models.get('yolo_nas_s', num_classes=1, pretrained_weights="coco")
            print("✓ Pretrained weights loaded")
        else:
            print("Training from scratch (no pretrained weights)...")
            model = models.get('yolo_nas_s', num_classes=1)
    except Exception as e:
        print(f"⚠ Could not load pretrained weights: {e}")
        model = models.get('yolo_nas_s', num_classes=1)
        
    latest_ckpt = os.path.join(output_dir, "checkpoints",'drone_detection', "ckpt_latest.pth")
    if os.path.exists(latest_ckpt):
        print(f"🔄 Found existing checkpoint: {latest_ckpt}")
        try:
            state = torch.load(latest_ckpt, map_location="cuda" if torch.cuda.is_available() else "cpu")
            model.load_state_dict(state['net'])
            print("✅ Checkpoint loaded successfully. Continue training...")
        except Exception as e:
            print(f"⚠ Could not load checkpoint: {e}")
    else:
        print("ℹ No checkpoint found, training from scratch.")


    # ✅ Training parameters — giảm tải GPU
    train_params = {
        'max_epochs': num_epochs,
        'lr_mode': 'cosine',
        'initial_lr': 5e-5,
        'lr_warmup_epochs': 2,
        'warmup_initial_lr': 1e-6,
        'optimizer': 'AdamW',
        'optimizer_params': {'weight_decay': 1e-4},
        'ema': False,  # ❌ tắt EMA để giảm VRAM
        'zero_weight_decay_on_bias_and_bn': True,
        'average_best_models': False,
        'mixed_precision': False,  # ❌ tránh float16 nếu GPU yếu
        'loss': PPYoloELoss(num_classes=1, use_static_assigner=False, reg_max=16),
        'valid_metrics_list': [
            DetectionMetrics(
                post_prediction_callback=PPYoloEPostPredictionCallback(
                    score_threshold=0.25,  # tăng để giảm số box giữ lại
                    nms_top_k=300,
                    max_predictions=100,
                    nms_threshold=0.6
                ),
                num_cls=1
            )
        ],
        'metric_to_watch': 'mAP@0.50:0.95',
        'greater_metric_to_watch_is_better': True,

        # ✅ Rất quan trọng cho GPU yếu
        'batch_accumulate': 2,     # mô phỏng batch lớn (2 ảnh/lần update)
        'max_batch_size_per_device': 1,
        'save_ckpt_epoch_list': [num_epochs // 2, num_epochs - 1]
    }

    print("\nStarting training...")
    try:
        os.makedirs(output_dir, exist_ok=True)
        trainer.train(
            model=model,
            training_params=train_params,
            train_loader=train_loader,
            valid_loader=val_loader,
            
        )
        print(f"\n✓ Training complete!")
        print(f"✓ Model saved to: {output_dir}")
    except torch.cuda.OutOfMemoryError:
        print("\n🚫 GPU out of memory! Try:")
        print("   - batch_size = 1")
        print("   - img_size = 416 or 320")
        print("   - use smaller model (yolo_nas_s)")