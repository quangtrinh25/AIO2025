# train.py (fast-defaults for RTX 4060 Ti)
import os
import logging
import yaml
import torch
import torch.nn as nn
import torch.distributed as dist
from super_gradients import init_trainer
from super_gradients.training import Trainer, models
from super_gradients.training.metrics import DetectionMetrics
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
from super_gradients.training.dataloaders.dataloaders import (
    coco_detection_yolo_format_train,
    coco_detection_yolo_format_val
)
from tqdm import tqdm
import time
import gc

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Distributed utils
def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()

def get_world_size():
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1

def get_rank():
    return dist.get_rank() if is_dist_avail_and_initialized() else 0

# Reference-aware module (unchanged logic kept, optional use)
class ReferenceAwareDetectionModel(nn.Module):
    def __init__(self, base_model, feature_dim=256):
        super().__init__()
        self.backbone = base_model.backbone
        self.neck = base_model.neck
        self.head = base_model.head
        self.reference_proj = nn.Linear(512, feature_dim)
        self.feature_proj = nn.Conv2d(256, feature_dim, 1)
        self.attention = nn.MultiheadAttention(feature_dim, num_heads=8)

    def forward(self, x, reference_features=None):
        features = self.backbone(x)
        if reference_features is not None:
            features = self._apply_reference_attention(features, reference_features)
        features = self.neck(features)
        outputs = self.head(features)
        return outputs

    def _apply_reference_attention(self, features, reference_features):
        b, c, h, w = features.shape
        spatial_features = self.feature_proj(features).view(b, -1, h * w).transpose(1, 2)
        ref_features = self.reference_proj(reference_features).unsqueeze(1)
        attended_features, _ = self.attention(query=spatial_features, key=ref_features, value=ref_features)
        attended_features = attended_features.transpose(1, 2).view(b, -1, h, w)
        return attended_features

class SearchAndRescueTrainer:
    def __init__(self, config_path: str = "search_rescue_config.yaml", local_rank: int = 0):
        self.config = self._load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.local_rank = local_rank
        self.world_size = 1
        self._maybe_init_distributed()
        self._setup_directories()

    def _maybe_init_distributed(self):
        if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
            try:
                dist.init_process_group(backend="nccl", init_method="env://")
                self.world_size = dist.get_world_size()
                self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
                torch.cuda.set_device(self.local_rank)
                logging.info(f"Initialized DDP: rank {self.local_rank} / world_size {self.world_size}")
            except Exception as e:
                logging.warning(f"Failed to init distributed: {e}; continuing single-process")
                self.world_size = 1
        else:
            self.world_size = 1

    def _load_config(self, config_path: str) -> dict:
        default_config = {
            'data': {
                'dataset_yaml': 'few_shot_dataset/dataset.yaml',
                'batch_size': 8,
                'num_workers': 6,
                'img_size': 512
            },
            'training': {
                'epochs': 30,
                'initial_lr': 0.01,
                'warmup_epochs': 3,
                'cosine_final_lr_ratio': 0.01,
                'optimizer': 'AdamW',
                'weight_decay': 0.0005,
                'ema': True,
                'patience': 12,
                'gradient_accumulation_steps': 2,
                'use_torch_compile': True,
                'use_mixed_precision': True
            },
            'model': {
                'architecture': 'yolo_nas_s',
                'num_classes': 1,
                'pretrained_weights': 'coco',
                'use_reference_attention': False
            },
            'augmentation': {
                'mosaic_prob': 0.5,
                'mixup_prob': 0.1,
                'hsv_h': 0.015,
                'hsv_s': 0.7,
                'hsv_v': 0.4,
                'degrees': 10.0,
                'translate': 0.15,
                'scale': 0.4,
                'shear': 4.0,
                'perspective': 0.0005,
                'flipud': 0.25,
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
        return config

    def _setup_directories(self):
        os.makedirs(self.config['checkpoint']['save_dir'], exist_ok=True)

    def _get_train_params(self):
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
                self.config['training']['epochs'] + 1,
                self.config['checkpoint']['save_interval']
            )),
            'average_best_models': True,
            'ema': self.config['training']['ema'],
            'mixed_precision': self.config['training'].get('use_mixed_precision', True),
            'early_stop': True,
            'early_stop_patience': self.config['training']['patience'],
            'early_stop_mode': 'max',
            'clip_grad_norm': 10.0,
            'large_scale_jitter': [0.8, 1.2],
        }
        return train_params

    def train(self, resume_checkpoint: str | None = None):
        start_time = time.time()
        try:
            init_trainer()
            trainer = Trainer(
                experiment_name=self.config['checkpoint']['experiment_name'],
                ckpt_root_dir=self.config['checkpoint']['save_dir']
            )

            # Data loaders (adapted for single-GPU/limited VRAM)
            batch_per_process = max(1, int(self.config['data']['batch_size'] // max(1, self.world_size)))
            dataloader_params = {
                'batch_size': batch_per_process,
                'num_workers': self.config['data']['num_workers'],
                'shuffle': True,
                'pin_memory': True,
                'persistent_workers': True,
                'prefetch_factor': 2
            }

            train_loader = coco_detection_yolo_format_train(
                dataset_params={
                    'data_dir': os.path.dirname(self.config['data']['dataset_yaml']),
                    'images_dir': 'images/train',
                    'labels_dir': 'labels/train',
                    'classes': ['target_object'],
                    'input_dim': [self.config['data']['img_size'], self.config['data']['img_size']],
                },
                dataloader_params=dataloader_params
            )

            val_loader = coco_detection_yolo_format_val(
                dataset_params={
                    'data_dir': os.path.dirname(self.config['data']['dataset_yaml']),
                    'images_dir': 'images/val',
                    'labels_dir': 'labels/val',
                    'classes': ['target_object'],
                    'input_dim': [self.config['data']['img_size'], self.config['data']['img_size']],
                },
                dataloader_params={**dataloader_params, 'shuffle': False}
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

            if self.config['model'].get('use_reference_attention', False):
                model = ReferenceAwareDetectionModel(base_model)
                logging.info("Using reference-aware detection model (this increases VRAM usage)")
            else:
                model = base_model

            model.to(self.device)

            # Attempt torch.compile for speed (safe fallback)
            use_compile = self.config['training'].get('use_torch_compile', True)
            if use_compile and hasattr(torch, "compile"):
                try:
                    model = torch.compile(model, backend="inductor")
                    logging.info("Wrapped model with torch.compile (inductor)")
                except Exception as e:
                    logging.warning(f"torch.compile not compatible: {e}")

            train_params = self._get_train_params()

            # Resume logic (tries trainer.load_checkpoint)
            resume_path = None
            if resume_checkpoint:
                exp_name = self.config['checkpoint']['experiment_name']
                base_dir = os.path.join(trainer.ckpt_root_dir, exp_name)
                candidate = os.path.join(base_dir, resume_checkpoint)
                if os.path.exists(candidate):
                    resume_path = candidate
                    logging.info(f"Resuming from: {resume_path}")
                else:
                    for root, dirs, files in os.walk(base_dir):
                        if resume_checkpoint in files:
                            resume_path = os.path.join(root, resume_checkpoint)
                            logging.info(f"Found checkpoint in subfolder: {resume_path}")
                            break
                    if not resume_path:
                        latest = []
                        for root, dirs, files in os.walk(base_dir):
                            for f in files:
                                if f == 'ckpt_latest.pth':
                                    latest.append(os.path.join(root, f))
                        if latest:
                            resume_path = max(latest, key=os.path.getctime)
                            logging.info(f"Auto-resume from latest: {resume_path}")

            if resume_path and os.path.exists(resume_path):
                try:
                    trainer.load_checkpoint(resume_path)
                except Exception as e:
                    logging.warning(f"trainer.load_checkpoint failed: {e}")

            logging.info("Starting training (fast defaults): epochs=%d img_size=%d batch_per_proc=%d accum_steps=%d",
                         self.config['training']['epochs'],
                         self.config['data']['img_size'],
                         batch_per_process,
                         self.config['training'].get('gradient_accumulation_steps', 1))

            trainer.train(
                model=model,
                training_params=train_params,
                train_loader=train_loader,
                valid_loader=val_loader
            )

            end_time = time.time()
            logging.info("Training finished in %.2f seconds (%.2f minutes)", end_time - start_time, (end_time - start_time) / 60.0)

        except Exception as e:
            logging.error("Training failed: %s", e)
            raise
