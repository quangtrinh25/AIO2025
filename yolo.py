from ultralytics import YOLO
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score
import tensorflow as tf
import cv2
import glob
import time

# Set random seed for reproducibility
np.random.seed(42)

# Define constants
DATA_DIR = r"D:\DATASET"
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 20

# Check if directories exist and contain images
def check_dataset(dirs, class_names=['dogs', 'cats']):
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            print(f"ERROR: Directory not found: {dir_path}")
            exit()
        for cls in class_names:
            cls_dir = os.path.join(dir_path, cls)
            if not os.path.exists(cls_dir):
                print(f"ERROR: Directory not found: {cls_dir}")
                exit()
            images = glob.glob(os.path.join(cls_dir, "*.jpg")) + glob.glob(os.path.join(cls_dir, "*.png"))
            if not images:
                print(f"ERROR: No images found in {cls_dir}")
                exit()
            print(f"Found {len(images)} images in {cls_dir}")

print("\n--- Checking dataset structure ---")
check_dataset([os.path.join(DATA_DIR, "TRAIN"), os.path.join(DATA_DIR, "VALIDATION"), os.path.join(DATA_DIR, "TEST")])

# Load pre-trained classification model
model = YOLO('yolov8n-cls.pt')

# Train the model
print("\n--- Starting training with YOLOv8 ---")
start_time = time.time()
results = model.train(
    data=DATA_DIR,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    name='dogcat_classification_yolov8n'
)
training_time = time.time() - start_time
print(f"--- Training time: {training_time:.2f} seconds ---")

# Load training results from CSV
results_csv_path = os.path.join('runs', 'classify', 'dogcat_classification_yolov8n', 'results.csv')
if os.path.exists(results_csv_path):
    history = pd.read_csv(results_csv_path)
else:
    print(f"ERROR: Results CSV not found at {results_csv_path}")
    exit()

# Remove whitespace from column names (YOLOv8 CSV columns sometimes have leading/trailing spaces)
history.columns = history.columns.str.strip()

# Visualize training history
plt.figure(figsize=(12, 5))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(history['train/loss'], label='Training loss')
plt.plot(history['val/loss'], label='Validation loss')
plt.title('Loss of YOLOv8')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(history['metrics/accuracy_top1'], label='Training accuracy')
plt.plot(history['val/accuracy_top1'], label='Validation accuracy')
plt.title('Accuracy of YOLOv8')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Evaluate model on test set
def evaluate_yolo_model(model, data_dir, split='test'):
    try:
        results = model.val(data=data_dir, split=split, imgsz=IMG_SIZE, batch=BATCH_SIZE)
        y_pred = []
        y_true = []
        
        dataset = model.trainer.test_dataloader
        for batch in dataset:
            imgs, labels = batch['img'], batch['cls']
            preds = model(imgs)
            y_pred.extend(preds.argmax(dim=1).cpu().numpy())
            y_true.extend(labels.cpu().numpy())

        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print("Confusion Matrix:")
        print(cm)

        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['dogs', 'cats'], yticklabels=['dogs', 'cats'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

        return accuracy, precision, recall, cm
    except Exception as e:
        print(f"ERROR: Evaluation failed: {e}")
        return None, None, None, None

print("\n--- Evaluating model on test set ---")
accuracy, precision, recall, cm = evaluate_yolo_model(model, DATA_DIR)

# Create comparison table
if accuracy is not None:
    comparison_results = [{
        'Model': 'YOLOv8',
        'Training Time (seconds)': training_time,
        'Test Accuracy (%)': accuracy * 100,
        'Precision (%)': precision * 100,
        'Recall (%)': recall * 100
    }]
    df_results = pd.DataFrame(comparison_results)
    print("\n--- Comparison table ---")
    print(df_results.to_string(index=False))