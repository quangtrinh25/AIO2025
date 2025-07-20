import os
import time
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tensorflow.keras.applications import ResNet50, VGG16, MobileNetV2, EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import backend as K
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import gc

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Define constants
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 16  # Reduced to avoid OOM
DATA_DIR = r"D:\DATASET"  # Use raw string for Windows paths
train_dir = os.path.join(DATA_DIR, "TRAIN")
validation_dir = os.path.join(DATA_DIR, "VALIDATION")
test_dir = os.path.join(DATA_DIR, "TEST")

# Check if directories exist
for directory in [train_dir, validation_dir, test_dir]:
    if not os.path.exists(directory):
        print(f"ERROR: Directory not found: {directory}")
        exit()

# Model-specific preprocessing functions
preprocess_functions = {
    'ResNet50': tf.keras.applications.resnet50.preprocess_input,
    'VGG16': tf.keras.applications.vgg16.preprocess_input,
    'MobileNetV2': tf.keras.applications.mobilenet_v2.preprocess_input,
    'EfficientNetB0': tf.keras.applications.efficientnet.preprocess_input
}

# Create ImageDataGenerator
def create_data_generators(base_model_name):
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_functions[base_model_name],
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest',
        brightness_range=[0.5, 1.5],
        channel_shift_range=0.2
    )
    validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_functions[base_model_name])
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_functions[base_model_name])
    return train_datagen, validation_datagen, test_datagen

# Visualize sample images
def visualize_samples(generator, class_names, num_samples=9):
    plt.figure(figsize=(10, 10))
    images, labels = next(generator)
    for i in range(min(num_samples, len(images))):
        ax = plt.subplot(3, 3, i + 1)
        img = images[i]
        # Undo preprocessing for visualization
        img = img - img.min()
        img = img / img.max()
        plt.imshow(img)
        plt.title(class_names[int(labels[i])])
        plt.axis("off")
    plt.show()

# Load data
print("\n--- Establish Train Generator ---")
train_datagen, validation_datagen, test_datagen = create_data_generators('ResNet50')  # Initial generator for visualization
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    seed=SEED
)

print("\n--- Establish Validation Generator ---")
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    seed=SEED
)

print("\n--- Establish Test Generator ---")
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False,
    seed=SEED
)

# Print data information
num_classes = train_generator.num_classes
class_names = list(train_generator.class_indices.keys())
print(f"\nNumber of classes: {num_classes}")
print(f"Class names: {class_names}")
print(f"Class indices: {train_generator.class_indices}")
print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {validation_generator.samples}")
print(f"Test samples: {test_generator.samples}")

# Visualize sample images
visualize_samples(train_generator, class_names)

# Function to evaluate precision, recall, and confusion matrix
def evaluate_metrics(model, test_generator, class_names, num_classes):
    test_generator.reset()
    y_pred = model.predict(test_generator, steps=max(1, test_generator.samples // BATCH_SIZE + 1))
    y_pred_binary = (y_pred > 0.5).astype(int).flatten()
    y_true = test_generator.classes

    precision = precision_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    cm = confusion_matrix(y_true, y_pred_binary)

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("Confusion Matrix:")
    print(cm)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    return precision, recall, cm

# Function to build and train the model with staged fine-tuning
def build_and_train_model(base_model_name, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), epochs=20, num_classes=2):
    print(f"\n--- Start training with {base_model_name} ---")

    # Create data generators for this specific model
    train_datagen, validation_datagen, test_datagen = create_data_generators(base_model_name)
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        seed=SEED
    )
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        seed=SEED
    )
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False,
        seed=SEED
    )

    # Load base model
    if base_model_name == 'ResNet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        fine_tune_layers = -50
    elif base_model_name == 'VGG16':
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
        fine_tune_layers = -50
    elif base_model_name == 'MobileNetV2':
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
        fine_tune_layers = -20
    elif base_model_name == 'EfficientNetB0':
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
        fine_tune_layers = -20
    else:
        raise ValueError("Invalid model name.")

    # Stage 1: Train only top layers
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x) if num_classes == 2 else Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile for stage 1
    loss = 'binary_crossentropy' if num_classes == 2 else 'categorical_crossentropy'
    model.compile(optimizer=Adam(learning_rate=1e-4), loss=loss, metrics=['accuracy'])

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
    checkpoint = ModelCheckpoint(f'best_{base_model_name}_stage1.h5', monitor='val_accuracy', save_best_only=True)

    # Train stage 1
    print("Training Stage 1: Top layers only")
    start_time = time.time()
    history_stage1 = model.fit(
        train_generator,
        steps_per_epoch=max(1, train_generator.samples // BATCH_SIZE),
        epochs=epochs // 2,
        validation_data=validation_generator,
        validation_steps=max(1, validation_generator.samples // BATCH_SIZE),
        callbacks=[early_stopping, reduce_lr, checkpoint]
    )

    # Stage 2: Fine-tune
    for layer in base_model.layers[fine_tune_layers:]:
        layer.trainable = True

    # Recompile with lower learning rate
    model.compile(optimizer=Adam(learning_rate=1e-5), loss=loss, metrics=['accuracy'])
    checkpoint = ModelCheckpoint(f'best_{base_model_name}_stage2.h5', monitor='val_accuracy', save_best_only=True)

    # Train stage 2
    print("Training Stage 2: Fine-tuning")
    history_stage2 = model.fit(
        train_generator,
        steps_per_epoch=max(1, train_generator.samples // BATCH_SIZE),
        epochs=epochs // 2,
        validation_data=validation_generator,
        validation_steps=max(1, validation_generator.samples // BATCH_SIZE),
        callbacks=[early_stopping, reduce_lr, checkpoint]
    )

    end_time = time.time()
    training_time = end_time - start_time
    print(f"\n--- Training time of {base_model_name}: {training_time:.2f} s ---")

    # Evaluate on test set
    eval_results = model.evaluate(test_generator, steps=max(1, test_generator.samples // BATCH_SIZE))
    test_accuracy = eval_results[1]
    print(f"Test accuracy ({base_model_name}): {test_accuracy*100:.2f}%")

    # Calculate precision, recall, and confusion matrix
    precision, recall, cm = evaluate_metrics(model, test_generator, class_names, num_classes)

    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history_stage1.history['accuracy'] + history_stage2.history['accuracy'], label='Training accuracy')
    plt.plot(history_stage1.history['val_accuracy'] + history_stage2.history['val_accuracy'], label='Validation accuracy')
    plt.title(f'Accuracy of {base_model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history_stage1.history['loss'] + history_stage2.history['loss'], label='Training loss')
    plt.plot(history_stage1.history['val_loss'] + history_stage2.history['val_loss'], label='Validation loss')
    plt.title(f'Loss of {base_model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    return model, history_stage2, training_time, test_accuracy, precision, recall, cm

# Training loop
models_to_train = [
    {'name': 'ResNet50', 'class': ResNet50},
    {'name': 'VGG16', 'class': VGG16},
    {'name': 'MobileNetV2', 'class': MobileNetV2},
    {'name': 'EfficientNetB0', 'class': EfficientNetB0}
]
trained_models = {}
comparison_results = []

for model_config in models_to_train:
    try:
        model, history, train_time, accuracy, precision, recall, cm = build_and_train_model(
            model_config['name'], epochs=20, num_classes=num_classes
        )
        trained_models[model_config['name']] = model
        comparison_results.append({
            'Model': model_config['name'],
            'Training Time (s)': train_time,
            'Test Accuracy (%)': accuracy * 100,
            'Precision (%)': precision * 100,
            'Recall (%)': recall * 100
        })
        K.clear_session()
        gc.collect()
    except Exception as e:
        print(f"Error training {model_config['name']}: {str(e)}")

# Print comparison table
print("\n--- Comparison table ---")
df_results = pd.DataFrame(comparison_results)
print(df_results.to_string(index=False))

# Plot comparison charts
plt.figure(figsize=(10, 6))
plt.bar(df_results['Model'], df_results['Test Accuracy (%)'], color='skyblue')
plt.xlabel('Model')
plt.ylabel('Test Accuracy (%)')
plt.title('Comparison of Test Accuracy for Different Models')
plt.ylim(0, 100)
for index, row in df_results.iterrows():
    plt.text(row['Model'], row['Test Accuracy (%)'] + 2, f"{row['Test Accuracy (%)']:.2f}%", ha='center')
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(df_results['Model'], df_results['Training Time (s)'], color='lightcoral')
plt.xlabel('Model')
plt.ylabel('Training Time (s)')
plt.title('Comparison of Training Time for Different Models')
for index, row in df_results.iterrows():
    plt.text(row['Model'], row['Training Time (s)'] + 0.5, f"{row['Training Time (s)']:.2f}s", ha='center')
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(df_results['Model'], df_results['Precision (%)'], color='lightgreen')
plt.xlabel('Model')
plt.ylabel('Precision (%)')
plt.title('Comparison of Precision for Different Models')
plt.ylim(0, 100)
for index, row in df_results.iterrows():
    plt.text(row['Model'], row['Precision (%)'] + 2, f"{row['Precision (%)']:.2f}%", ha='center')
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(df_results['Model'], df_results['Recall (%)'], color='lightblue')
plt.xlabel('Model')
plt.ylabel('Recall (%)')
plt.title('Comparison of Recall for Different Models')
plt.ylim(0, 100)
for index, row in df_results.iterrows():
    plt.text(row['Model'], row['Recall (%)'] + 2, f"{row['Recall (%)']:.2f}%", ha='center')
plt.show()