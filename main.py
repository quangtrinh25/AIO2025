import os
import traceback
import time
import sys

# Import functions from other files
from data_prep import convert_to_yolo_format
from train import train_yolo_nas
from inference import generate_predictions
from metrics import calculate_final_score

# ==============================================================================
# Main Pipeline
# ==============================================================================



def main():
    # --- Bắt đầu đo thời gian tổng ---
    pipeline_start_time = time.time()
    
    BASE_DIR = r'D:/zalo_ai'
    TRAIN_ANNOTATIONS = r'D:/zalo_ai/observing/train/annotations'
    TRAIN_SAMPLES = r'D:/zalo_ai/observing/train/samples' 
    TEST_SAMPLES = r'D:/zalo_ai/public_test/samples'
    
    OUTPUT_DIR = r'D:/zalo_ai/output'
    YOLO_FORMAT_PATH = r'D:/zalo_ai/output/yolo_dataset'
    CHECKPOINTS_DIR = r'D:/zalo_ai/output/checkpoints'
    PREDICTIONS_DIR = r'D:/zalo_ai/output/predictions'
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    GROUND_TRUTH_FILE = os.path.join(TRAIN_ANNOTATIONS, 'annotations.json')
    
    print(f"\n{'='*70}")
    print("YOLO-NAS Drone Detection Pipeline")
    print(f"{'='*70}\n")
    
    '''# Step 1: Convert to YOLO format
    print("Step 1: Converting dataset to YOLO format...")
    step1_start = time.time() # <--- THÊM VÀO
    convert_to_yolo_format(TRAIN_ANNOTATIONS, TRAIN_SAMPLES, YOLO_FORMAT_PATH)
    step1_end = time.time() # <--- THÊM VÀO
    print(f"--- Step 1 finished in {step1_end - step1_start:.2f} seconds ---")
    '''
    
    # Step 2: Train
    print("\nStep 2: Training YOLO-NAS model...")
    step2_start = time.time() 
    try:
        train_yolo_nas(YOLO_FORMAT_PATH, CHECKPOINTS_DIR, num_epochs=50, use_pretrained=False)
    except Exception as e:
        print(f"Training error: {e}")
        traceback.print_exc()
        return
    step2_end = time.time() 
    print(f"--- Step 2 finished in {step2_end - step2_start:.2f} seconds ---")
    
    # Step 3: Validation predictions
    print("\nStep 3: Generating predictions on training set...")
    step3_start = time.time() 
    model_path = CHECKPOINTS_DIR
    train_predictions_file = os.path.join(PREDICTIONS_DIR, 'train_predictions.json')
    generate_predictions(model_path, TRAIN_SAMPLES, train_predictions_file)
    step3_end = time.time() 
    print(f"--- Step 3 finished in {step3_end - step3_start:.2f} seconds ---")
    
    # Step 4: Calculate ST-IoU
    print("\nStep 4: Calculating ST-IoU score...")
    step4_start = time.time() 
    try:
        final_score = calculate_final_score(train_predictions_file, GROUND_TRUTH_FILE)
        print(f"\n{'='*70}")
        print(f"Training ST-IoU Score: {final_score:.4f}")
        print(f"{'='*70}")
    except Exception as e:
        print(f"Error calculating score: {e}")
    step4_end = time.time() 
    print(f"--- Step 4 finished in {step4_end - step4_start:.2f} seconds ---")
    
    # Step 5: Test predictions
    print("\nStep 5: Generating predictions on test set...")
    step5_start = time.time() 
    test_predictions_file = os.path.join(PREDICTIONS_DIR, 'test_predictions.json')
    generate_predictions(model_path, TEST_SAMPLES, test_predictions_file)
    step5_end = time.time() 
    print(f"--- Step 5 finished in {step5_end - step5_start:.2f} seconds ---")
    
    print(f"\n{'='*70}")
    print("Pipeline Complete!")
    print(f"{'='*70}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Training predictions: {train_predictions_file}")
    print(f"Test predictions: {test_predictions_file}")
    
    # --- In tổng thời gian ---
    pipeline_end_time = time.time() 
    total_duration = pipeline_end_time - pipeline_start_time
    print(f"\nTotal pipeline duration: {total_duration:.2f} seconds")


if __name__ == '__main__':
    main()