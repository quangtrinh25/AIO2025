import os
import logging
import yaml
import torch
import torch.nn as nn
from super_gradients import init_trainer
from super_gradients.training import Trainer, models
from super_gradients.training.dataloaders.dataloaders import get_data_loader
from super_gradients.training.metrics import DetectionMetrics
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
from super_gradients.training.dataloaders.dataloaders import (
    coco_detection_yolo_format_train,
    coco_detection_yolo_format_val
) 
import torch.nn.functional as F
from tqdm import tqdm
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

class ReferenceAwareDetectionModel(nn.Module):
    """YOLO-NAS với cơ chế attention từ reference features"""
    def __init__(self, base_model, feature_dim=256):
        super().__init__()
        self.backbone = base_model.backbone
        self.neck = base_model.neck
        self.head = base_model.head
        
        # Reference attention mechanism
        self.reference_proj = nn.Linear(512, feature_dim)
        self.feature_proj = nn.Conv2d(256, feature_dim, 1)
        self.attention = nn.MultiheadAttention(feature_dim, num_heads=8)
        
    def forward(self, x, reference_features=None):
        features = self.backbone(x)
        
        if reference_features is not None:
            # Áp dụng reference-guided attention
            features = self._apply_reference_attention(features, reference_features)
        
        features = self.neck(features)
        outputs = self.head(features)
        return outputs
    
    def _apply_reference_attention(self, features, reference_features):
        """Áp dụng attention dựa trên reference features"""
        # features: [B, C, H, W], reference_features: [B, D]
        b, c, h, w = features.shape
        
        # Project features to same dimension
        spatial_features = self.feature_proj(features)  # [B, D, H, W]
        spatial_features = spatial_features.view(b, -1, h*w).transpose(1, 2)  # [B, H*W, D]
        
        # Project reference features
        ref_features = self.reference_proj(reference_features).unsqueeze(1)  # [B, 1, D]
        
        # Apply attention
        attended_features, _ = self.attention(
            query=spatial_features, 
            key=ref_features, 
            value=ref_features
        )
        
        attended_features = attended_features.transpose(1, 2).view(b, -1, h, w)
        return attended_features

class SearchAndRescueTrainer:
    def __init__(self, config_path: str = "search_rescue_config.yaml"):
        self.config = self._load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup_directories()
        
    def _load_config(self, config_path: str) -> dict:
        """Load training configuration với error handling tốt hơn"""
        default_config = {
            'data': {
                'dataset_yaml': 'few_shot_dataset/dataset.yaml',
                'batch_size': 16,
                'num_workers': 4,
                'img_size': 640
            },
            'training': {
                'epochs': 100,
                'initial_lr': 0.01,
                'warmup_epochs': 3,
                'cosine_final_lr_ratio': 0.01,
                'optimizer': 'AdamW',
                'weight_decay': 0.0005,
                'ema': True,
                'patience': 20
            },
            'model': {
                'architecture': 'yolo_nas_s',
                'num_classes': 1,
                'pretrained_weights': 'coco',
                'use_reference_attention': False
            },
            'augmentation': {
                'mosaic_prob': 0.7,
                'mixup_prob': 0.2,
                'hsv_h': 0.015,
                'hsv_s': 0.7,
                'hsv_v': 0.4,
                'degrees': 15.0,
                'translate': 0.2,
                'scale': 0.5,
                'shear': 5.0,
                'perspective': 0.0005,
                'flipud': 0.3,
                'fliplr': 0.5
            },
            'checkpoint': {
                'save_dir': 'search_rescue_checkpoints',
                'experiment_name': 'drone_search_rescue',
                'save_interval': 10
            }
        }
        
        config = default_config.copy()
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = yaml.safe_load(f)
                
                if user_config:
                    for key, value in user_config.items():
                        if key in config and isinstance(config[key], dict) and isinstance(value, dict):
                            config[key].update(value)
                        else:
                            config[key] = value
                
                logging.info(f"Loaded config from {config_path}")
            except Exception as e:
                logging.error(f"Error loading config from {config_path}: {e}")
                logging.info("Using default configuration")
        else:
            logging.info("Config file not found, using default configuration")
        
        if 'augmentation' not in config:
            config['augmentation'] = default_config['augmentation']
            logging.info("Added missing augmentation config")
            
        return config

    def _setup_directories(self):
        """Create necessary directories"""
        os.makedirs(self.config['checkpoint']['save_dir'], exist_ok=True)
        
    def _get_train_params(self):
        """Training parameters tối ưu cho drone search-and-rescue"""
        train_params = {
            'max_epochs': self.config['training']['epochs'],
            'initial_lr': self.config['training']['initial_lr'],
            'optimizer': self.config['training']['optimizer'],
            'optimizer_params': {
                'weight_decay': self.config['training']['weight_decay']
            },
            'lr_mode': 'cosine',
            'cosine_final_lr_ratio': self.config['training']['cosine_final_lr_ratio'],
            'lr_warmup_epochs': self.config['training']['warmup_epochs'],
            'warmup_initial_lr': self.config['training']['initial_lr'] * 0.1,
            'batch_size': self.config['data']['batch_size'],
            'num_workers': self.config['data']['num_workers'],
            
            # THÊM DÒNG NÀY - QUAN TRỌNG:
            'loss': PPYoloELoss(
                num_classes=self.config['model']['num_classes'],
                use_static_assigner=True,
                reg_max=16,
            ),
            
            'valid_metrics_list': [
                DetectionMetrics(
                    post_prediction_callback=PPYoloEPostPredictionCallback(
                        score_threshold=0.005,
                        nms_top_k=1500,
                        max_predictions=500,
                        nms_threshold=0.6
                    ),
                    num_cls=self.config['model']['num_classes'],
                )
            ],
            'metric_to_watch': 'mAP@0.50:0.95',
            'greater_metric_to_watch_is_better': True,
            'save_ckpt_epoch_list': list(range(
                self.config['checkpoint']['save_interval'],
                self.config['training']['epochs'],
                self.config['checkpoint']['save_interval']
            )),
            'average_best_models': True,
            'ema': self.config['training']['ema'],
            'mixed_precision': True,
            'early_stop': True,
            'early_stop_patience': self.config['training']['patience'],
            'early_stop_mode': 'max',
            'clip_grad_norm': 10.0,
            'large_scale_jitter': [0.8, 1.2],
        }
        
        return train_params

    def train(self):
        """Training với progress tracking"""
        start_time = time.time()
        
        try:
            # Initialize Super-Gradients
            init_trainer()
            
            # Setup trainer
            trainer = Trainer(
                experiment_name=self.config['checkpoint']['experiment_name'],
                ckpt_root_dir=self.config['checkpoint']['save_dir']
            )
            
            # Get data loaders với augmentation mạnh cho drone scenarios
            logging.info("Setting up data loaders...")
            
            augmentation_config = self.config.get('augmentation', {})
            
            default_augmentation = {
                'mosaic_prob': 0.0,
                'mixup_prob': 0.0,
                'hsv_h': 0.0,
                'hsv_s': 0.0,
                'hsv_v': 0.0,
                'degrees': 0.0,
                'translate': 0.0,
                'scale': 0.0,
                'shear': 0.0,
                'perspective': 0.0,
                'flipud': 0.0,
                'fliplr': 0.0
            }
            
            for key, default_value in default_augmentation.items():
                if key not in augmentation_config:
                    augmentation_config[key] = default_value
                    logging.info(f"Using default augmentation value for {key}: {default_value}")
            
            # SỬA LẠI PHẦN NÀY: Thêm dataset_cls và train parameters
            train_loader = coco_detection_yolo_format_train(
                dataset_params={
                    'data_dir': os.path.dirname(self.config['data']['dataset_yaml']),
                    'images_dir': 'images/train',
                    'labels_dir': 'labels/train',
                    'classes': ['target_object'],
                    'input_dim': [self.config['data']['img_size'], self.config['data']['img_size']],
                },
                dataloader_params={
                    'batch_size': self.config['data']['batch_size'],
                    'num_workers': self.config['data']['num_workers'],
                    'shuffle': True,
                    'pin_memory': True
                }
            )

            val_loader = coco_detection_yolo_format_val(
                dataset_params={
                    'data_dir': os.path.dirname(self.config['data']['dataset_yaml']),
                    'images_dir': 'images/val',
                    'labels_dir': 'labels/val',
                    'classes': ['target_object'],
                    'input_dim': [self.config['data']['img_size'], self.config['data']['img_size']],
                },
                dataloader_params={
                    'batch_size': self.config['data']['batch_size'],
                    'num_workers': self.config['data']['num_workers'],
                    'shuffle': False,
                    'pin_memory': True
                },
            )
            
            # Build model
            model_path = "yolo_nas_s_coco.pth"  
            if os.path.exists(model_path):
                base_model = models.get(
                    model_name=self.config['model']['architecture'],
                    num_classes=self.config['model']['num_classes'],
                    checkpoint_path=model_path  
            )
                logging.info(f"Loaded model from local: {model_path}")
            else:
                base_model = models.get(
                    model_name=self.config['model']['architecture'],
                    num_classes=self.config['model']['num_classes'],
                    pretrained_weights=self.config['model']['pretrained_weights']
                    
            )
            # Wrap với reference-aware mechanism nếu được enabled
            if self.config['model']['use_reference_attention']:
                model = ReferenceAwareDetectionModel(base_model)
                logging.info("Using reference-aware detection model")
            else:
                model = base_model
            
            # Training parameters
            train_params = self._get_train_params()
            
            # Start training với progress tracking
            logging.info("Starting Search-and-Rescue YOLO-NAS training...")
            
            trainer.train(
                model=model,
                training_params=train_params,
                train_loader=train_loader,
                valid_loader=val_loader
            )
            
            end_time = time.time()
            training_duration = end_time - start_time
            
            logging.info("Search-and-rescue training completed!")
            logging.info(f"Training took {training_duration:.2f} seconds ({training_duration/60:.2f} minutes)")
            
        except Exception as e:
            logging.error(f"Training failed: {e}")
            raise