# main.py
import os
import traceback
import time
import sys
import logging
import argparse
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Import from the updated modules
try:
    from data_prep import FewShotDataPreprocessor
    from train import SearchAndRescueTrainer
    from inferences import ReferenceGuidedInference
    from metrics import calculate_final_score, calculate_detection_metrics, print_detailed_metrics
except ImportError as e:
    logging.error(f"Import error: {e}")
    logging.error("Make sure you have the updated versions of data_prep.py, train.py, inferences.py, and metrics.py")
    sys.exit(1)

# ==============================================================================
# Main Pipeline
# ==============================================================================

def ensure_dir(path: str):
    """Create directory if it doesn't exist"""
    if path:
        os.makedirs(path, exist_ok=True)

def setup_paths(args):
    """Setup and validate all file paths"""
    
    # Base directory structure
    if args.dataset_root:
        BASE_DIR = args.dataset_root
        TRAIN_ANNOTATIONS = os.path.join(BASE_DIR, "annotations")
        TRAIN_SAMPLES = os.path.join(BASE_DIR, "samples")
    else:
        # Fallback to your original structure
        BASE_DIR = r'C:/zalo_ai'
        TRAIN_ANNOTATIONS = r'C:/zalo_ai/observing/train/annotations'
        TRAIN_SAMPLES = r'C:/zalo_ai/observing/train/samples'
    
    # Test samples
    if args.test_dir:
        TEST_SAMPLES = args.test_dir
    else:
        TEST_SAMPLES = r'C:/zalo_ai/public_test/samples'
    
    # Output directories
    if args.output_dir:
        OUTPUT_DIR = args.output_dir
    else:
        OUTPUT_DIR = r'C:/zalo_ai/output'
    
    YOLO_FORMAT_PATH = os.path.join(OUTPUT_DIR, 'few_shot_dataset')
    CHECKPOINTS_DIR = os.path.join(OUTPUT_DIR, 'search_rescue_checkpoints')
    PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, 'predictions')
    
    # Create directories
    ensure_dir(OUTPUT_DIR)
    ensure_dir(YOLO_FORMAT_PATH)
    ensure_dir(CHECKPOINTS_DIR)
    ensure_dir(PREDICTIONS_DIR)
    
    # Important files
    GROUND_TRUTH_FILE = os.path.join(TRAIN_ANNOTATIONS, 'annotations.json')
    CONFIG_FILE = "search_rescue_config.yaml"
    
    paths = {
        'base_dir': BASE_DIR,
        'train_annotations': TRAIN_ANNOTATIONS,
        'train_samples': TRAIN_SAMPLES,
        'test_samples': TEST_SAMPLES,
        'output_dir': OUTPUT_DIR,
        'yolo_format_path': YOLO_FORMAT_PATH,
        'checkpoints_dir': CHECKPOINTS_DIR,
        'predictions_dir': PREDICTIONS_DIR,
        'ground_truth_file': GROUND_TRUTH_FILE,
        'config_file': CONFIG_FILE
    }
    
    # Validate critical paths
    if not os.path.exists(TRAIN_ANNOTATIONS):
        logging.warning(f"Train annotations path does not exist: {TRAIN_ANNOTATIONS}")
    
    if not os.path.exists(TRAIN_SAMPLES):
        logging.warning(f"Train samples path does not exist: {TRAIN_SAMPLES}")
    
    if not os.path.exists(TEST_SAMPLES):
        logging.warning(f"Test samples path does not exist: {TEST_SAMPLES}")
    
    return paths

def create_config_file(config_path: str, yolo_dataset_path: str, epochs: int = 30, img_size: int = 512, batch_size: int = 8):
    """Create the training configuration file"""
    dataset_yaml_path = os.path.join(yolo_dataset_path, 'dataset.yaml')
    
    if os.name == 'nt':
        dataset_yaml_path = dataset_yaml_path.replace('\\', '/')
    
    config_content = f"""# search_rescue_config.yaml
data:
  dataset_yaml: "{dataset_yaml_path}"
  batch_size: {batch_size}
  num_workers: 6
  img_size: {img_size}

training:
  epochs: {epochs}
  initial_lr: 0.01
  warmup_epochs: 3
  cosine_final_lr_ratio: 0.01
  optimizer: "AdamW"
  weight_decay: 0.0005
  ema: true
  patience: 12
  gradient_accumulation_steps: 2
  use_torch_compile: true
  use_mixed_precision: true

model:
  architecture: "yolo_nas_s"
  num_classes: 1
  pretrained_weights: "coco"
  use_reference_attention: false

augmentation:
  mosaic_prob: 0.5
  mixup_prob: 0.1
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
  degrees: 10.0
  translate: 0.15
  scale: 0.4
  shear: 4.0
  perspective: 0.0005
  flipud: 0.25
  fliplr: 0.5

checkpoint:
  save_dir: "search_rescue_checkpoints"
  experiment_name: "drone_search_rescue"
  save_interval: 10
"""
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_content)
    logging.info(f"Created config file: {config_path}")
    
    # Verify the config file can be loaded
    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            yaml.safe_load(f)
        logging.info("Config file validation: SUCCESS")
    except Exception as e:
        logging.error(f"Config file validation failed: {e}")
        # Create a minimal safe config as fallback
        create_minimal_safe_config(config_path, yolo_dataset_path, epochs, img_size, batch_size)

def create_minimal_safe_config(config_path: str, yolo_dataset_path: str, epochs: int = 30, img_size: int = 512, batch_size: int = 8):
    """Tạo config file đơn giản, an toàn để tránh lỗi YAML"""
    safe_config = {
        'data': {
            'dataset_yaml': os.path.join(yolo_dataset_path, 'dataset.yaml').replace('\\', '/'),
            'batch_size': batch_size,
            'num_workers': 4,
            'img_size': img_size
        },
        'training': {
            'epochs': epochs,
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
        'checkpoint': {
            'save_dir': 'search_rescue_checkpoints',
            'experiment_name': 'drone_search_rescue',
            'save_interval': 10
        }
    }
    
    import yaml
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(safe_config, f, default_flow_style=False, allow_unicode=True)
    
    logging.info(f"Created minimal safe config file: {config_path}")

def main():
    parser = argparse.ArgumentParser(description='Drone Search-and-Rescue Pipeline')
    parser.add_argument('--dataset_root', type=str, help='Root directory of dataset (contains annotations/ and samples/)')
    parser.add_argument('--test_dir', type=str, help='Directory containing test samples')
    parser.add_argument('--output_dir', type=str, help='Output directory for results')
    parser.add_argument('--skip_training', action='store_true', help='Skip training and use existing model')
    parser.add_argument('--model_path', type=str, help='Path to existing model for inference')
    parser.add_argument('--similarity_threshold', type=float, default=0.4, help='Similarity threshold for reference matching')
    parser.add_argument('--iou_threshold', type=float, default=0.5, help='IoU threshold for evaluation')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint (e.g. ckpt_epoch_30.pth, ckpt_latest.pth)')
    parser.add_argument('--save_debug_videos', action='store_true', help='Save debug videos with drawn detections')
    parser.add_argument('--max_frames_per_video', type=int, default=None, help='Limit frames per video during inference for quick debug')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs to write into config')
    parser.add_argument('--img_size', type=int, default=512, help='Image size to write into config')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size to write into config')
    args = parser.parse_args()
    
    pipeline_start_time = time.time()

    # Setup paths
    paths = setup_paths(args)
    
    logging.info("\n" + "=" * 70)
    logging.info(" DRONE SEARCH-AND-RESCUE PIPELINE")
    logging.info("=" * 70)
    logging.info(f"Dataset root: {paths['train_annotations']}")
    logging.info(f"Output directory: {paths['output_dir']}")
    logging.info(f"Skip training: {args.skip_training}")
    logging.info("=" * 70 + "\n")

    # Create config file (with fast defaults tuned for RTX 4060 Ti)
    create_config_file(paths['config_file'], paths['yolo_format_path'], epochs=args.epochs, img_size=args.img_size, batch_size=args.batch_size)

    # -------------------------
    # Step 1: Convert to YOLO format (optional)
    # -------------------------
    if not args.skip_training:
        logging.info(" Step 1: Converting dataset to YOLO format...")
        step1_start = time.time()
        try:
            preprocessor = FewShotDataPreprocessor(
                dataset_root=os.path.dirname(paths['train_annotations']),
                output_root=paths['yolo_format_path'],
                resize_images=False,  # keep original resolution; change to True if you want resize
                resize_size=(args.img_size, args.img_size)
            )
            preprocessor.create_few_shot_dataset(train_ratio=0.8)
        except Exception as e:
            logging.exception(f"Data conversion failed: {e}")
            return
        step1_end = time.time()
        logging.info(f" Step 1 finished in {step1_end - step1_start:.2f} seconds")
    else:
        logging.info("  Step 1: Skipping data conversion (using existing data)")

    # -------------------------
    # Step 2: Train detector
    # -------------------------
    if not args.skip_training:
        logging.info("\n Step 2: Training Search-and-Rescue YOLO-NAS model...")
        step2_start = time.time()
        try:
            trainer = SearchAndRescueTrainer(config_path=paths['config_file'])
            trainer.train(resume_checkpoint=args.resume)
            # The model will be saved in checkpoints_dir automatically
            model_path = os.path.join(paths['checkpoints_dir'], 'drone_search_rescue', 'ckpt_best.pth')
        except Exception as e:
            logging.exception(f"Training error: {e}")
            return
        step2_end = time.time()
        logging.info(f" Step 2 finished in {step2_end - step2_start:.2f} seconds")
    else:
        if args.model_path:
            model_path = args.model_path
        else:
            # Try to find the best model in checkpoints
            model_path = os.path.join(paths['checkpoints_dir'], 'drone_search_rescue', 'ckpt_best.pth')
            if not os.path.exists(model_path):
                logging.error(f"Model not found: {model_path}. Please specify with --model_path")
                return
        logging.info(f"  Step 2: Using existing model: {model_path}")
    
    # Ensure model exists
    if not os.path.exists(model_path):
        logging.error(f"Model path not found: {model_path}")
        return

    # Instantiate inference engine once (reuse for train/test predictions)
    inference_engine = ReferenceGuidedInference(
        model_path=model_path,
        similarity_threshold=args.similarity_threshold,
        save_debug_video=args.save_debug_videos
    )

    # -------------------------
    # Step 3: Generate predictions on training set
    # -------------------------
    logging.info("\n Step 3: Generating predictions on training set...")
    step3_start = time.time()
    try:
        train_predictions_file = os.path.join(paths['predictions_dir'], 'train_predictions.json')
        
        inference_engine.process_all_videos(
            test_samples_dir=paths['train_samples'],
            model_path=model_path,
            output_json_path=train_predictions_file,
            similarity_threshold=args.similarity_threshold,
            save_debug_videos=args.save_debug_videos,
            max_frames_per_video=args.max_frames_per_video
        )
    except Exception as e:
        logging.exception(f"Error during train-set prediction: {e}")
        return
    step3_end = time.time()
    logging.info(f" Step 3 finished in {step3_end - step3_start:.2f} seconds")

    # -------------------------
    # Step 4: Calculate metrics on training set
    # -------------------------
    logging.info("\n Step 4: Calculating metrics on training set...")
    step4_start = time.time()
    try:
        if not os.path.exists(train_predictions_file):
            raise FileNotFoundError(f"Predictions file not found: {train_predictions_file}")
        if not os.path.exists(paths['ground_truth_file']):
            raise FileNotFoundError(f"Ground-truth file not found: {paths['ground_truth_file']}")
        
        # Calculate comprehensive metrics
        metrics = calculate_detection_metrics(
            train_predictions_file, 
            paths['ground_truth_file'], 
            iou_threshold=args.iou_threshold,
            matching_strategy="hungarian",
            return_per_video=False
        )
        
        print_detailed_metrics(metrics)
        
    except Exception as e:
        logging.exception(f"Error calculating metrics: {e}")
    step4_end = time.time()
    logging.info(f" Step 4 finished in {step4_end - step4_start:.2f} seconds")

    # -------------------------
    # Step 5: Generate predictions on test set
    # -------------------------
    logging.info("\n Step 5: Generating predictions on test set...")
    step5_start = time.time()
    try:
        test_predictions_file = os.path.join(paths['predictions_dir'], 'test_predictions.json')
        
        inference_engine.process_all_videos(
            test_samples_dir=paths['test_samples'],
            model_path=model_path,
            output_json_path=test_predictions_file,
            similarity_threshold=args.similarity_threshold,
            save_debug_videos=args.save_debug_videos,
            max_frames_per_video=args.max_frames_per_video
        )
        
        logging.info(f"Test predictions saved to: {test_predictions_file}")
        
    except Exception as e:
        logging.exception(f"Error during test-set prediction: {e}")
        return
    step5_end = time.time()
    logging.info(f"Step 5 finished in {step5_end - step5_start:.2f} seconds")

    # Final summary
    pipeline_end_time = time.time()
    total_duration = pipeline_end_time - pipeline_start_time
    
    logging.info("\n" + "=" * 70)
    logging.info(" PIPELINE COMPLETE!")
    logging.info("=" * 70)
    logging.info(f"  Total pipeline duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    logging.info(f" Output directory: {paths['output_dir']}")
    logging.info(f" Training predictions: {train_predictions_file}")
    logging.info(f" Test predictions: {test_predictions_file}")
    logging.info(f" Model used: {model_path}")
    logging.info("=" * 70)

if __name__ == '__main__':
    main()
