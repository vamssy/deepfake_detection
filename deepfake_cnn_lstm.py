import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM, TimeDistributed, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import cv2
import os
import shutil
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.utils import Sequence

# Step 1: Custom data generator for sequence data
class SequenceDataGenerator(Sequence):
    def __init__(self, X, y, batch_size, datagen, sequence_length=10):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.datagen = datagen
        self.sequence_length = sequence_length
        self.indices = np.arange(len(X))

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_X = self.X[batch_indices]
        batch_y = self.y[batch_indices]
        
        augmented_X = np.zeros_like(batch_X)
        for i in range(len(batch_X)):
            for j in range(self.sequence_length):
                augmented_X[i, j] = self.datagen.random_transform(batch_X[i, j])
        
        return augmented_X, batch_y

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

# Step 2: Extract frames from videos and preprocess
def extract_and_preprocess_videos(input_dir, output_dir, img_size=(224, 224), max_images_per_class=5000, frames_per_video=25):
    """
    Extract frames from .mp4 videos and save as images in real and fake folders.
    """
    real_dir = os.path.join(output_dir, 'real')
    fake_dir = os.path.join(output_dir, 'fake')
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    # Process real videos
    real_count = 0
    real_path = os.path.join(input_dir, 'original_sequences', 'youtube', 'c23', 'videos')
    if not os.path.exists(real_path):
        raise FileNotFoundError(f"Real videos path {real_path} does not exist.")
    
    for video_name in os.listdir(real_path):
        if real_count >= max_images_per_class:
            break
        if video_name.endswith('.mp4'):
            video_path = os.path.join(real_path, video_name)
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step = max(1, frame_count // frames_per_video)
            frame_idx = 0
            extracted = 0

            while cap.isOpened() and extracted < frames_per_video:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, img_size)
                cv2.imwrite(os.path.join(real_dir, f'real_{real_count}.jpg'), frame)
                real_count += 1
                extracted += 1
                frame_idx += step
                if real_count >= max_images_per_class:
                    break
            cap.release()

    # Process fake videos
    fake_count = 0
    fake_path = os.path.join(input_dir, 'manipulated_sequences', 'Deepfakes', 'c23', 'videos')
    if not os.path.exists(fake_path):
        raise FileNotFoundError(f"Fake videos path {fake_path} does not exist.")
    
    for video_name in os.listdir(fake_path):
        if fake_count >= max_images_per_class:
            break
        if video_name.endswith('.mp4'):
            video_path = os.path.join(fake_path, video_name)
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step = max(1, frame_count // frames_per_video)
            frame_idx = 0
            extracted = 0

            while cap.isOpened() and extracted < frames_per_video:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, img_size)
                cv2.imwrite(os.path.join(fake_dir, f'fake_{fake_count}.jpg'), frame)
                fake_count += 1
                extracted += 1
                frame_idx += step
                if fake_count >= max_images_per_class:
                    break
            cap.release()

    print(f"Extracted {real_count} real images and {fake_count} fake images.")

# Step 3: Load images and create sequences
def load_images(data_dir, img_size=(224, 224), sequence_length=10):
    """
    Load images and group into sequences for LSTM input.
    """
    images = []
    labels = []
    sequences = []
    sequence = []
    count = 0

    for label in ['real', 'fake']:
        label_dir = os.path.join(data_dir, label)
        if not os.path.exists(label_dir):
            raise FileNotFoundError(f"Label directory {label_dir} does not exist.")
        
        for img_name in sorted(os.listdir(label_dir)):
            img_path = os.path.join(label_dir, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, img_size)
                img = img / 255.0  # Normalize
                sequence.append(img)
                count += 1

                if count == sequence_length:
                    sequences.append(sequence)
                    labels.append(1 if label == 'fake' else 0)
                    sequence = []
                    count = 0

    return np.array(sequences), np.array(labels)

# Step 4: Build CNN-LSTM model with VGG16
def build_model(input_shape=(10, 224, 224, 3)):
    # Load pre-trained VGG16
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers[:-5]:  # Unfreeze last 5 layers
        layer.trainable = True
    for layer in base_model.layers[:-10]:  # Freeze first 10 layers
        layer.trainable = False

    model = Sequential([
        TimeDistributed(base_model, input_shape=input_shape),
        TimeDistributed(Flatten()),
        LSTM(128, return_sequences=False),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    # Compile with a lower learning rate
    optimizer = Adam(learning_rate=1e-5)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Step 5: Plot training history
def plot_history(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

# Main execution
if __name__ == "__main__":
    # Paths
    input_dir = '/Users/Vamshi/Downloads/NNFL_Project/faceforensics'
    output_dir = '/Users/Vamshi/Downloads/NNFL_Project/faceforensics_processed'

    # Extract and preprocess videos (re-run to get more frames)
    shutil.rmtree(output_dir, ignore_errors=True)  # Delete existing processed folder
    extract_and_preprocess_videos(input_dir, output_dir, max_images_per_class=5000, frames_per_video=25)

    # Load data
    X, y = load_images(output_dir)
    print(f"Loaded {len(X)} sequences with shape {X.shape}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2]
    )

    # Create custom generators
    train_generator = SequenceDataGenerator(X_train, y_train, batch_size=16, datagen=datagen)
    val_generator = SequenceDataGenerator(X_test, y_test, batch_size=16, datagen=datagen)

    # Build and train model with early stopping
    model = build_model()
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=50,
        callbacks=[early_stopping],
        verbose=1
    )

    # Fine-tune: Unfreeze more layers
    model.layers[0].trainable = True
    for layer in model.layers[0].layers[:-10]:  # Freeze first 10 layers
        layer.trainable = False
    
    # Recompile with a very low learning rate for fine-tuning
    optimizer = Adam(learning_rate=1e-6)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    # Fine-tune
    history_fine = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=20,
        callbacks=[early_stopping],
        verbose=1
    )

    # Combine histories for plotting
    history.history['accuracy'] += history_fine.history['accuracy']
    history.history['val_accuracy'] += history_fine.history['val_accuracy']
    history.history['loss'] += history_fine.history['loss']
    history.history['val_loss'] += history_fine.history['val_loss']

    # Evaluate model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy*100:.2f}%")

    # Plot results
    plot_history(history)

    # Save model
    model.save('cnn_lstm_deepfake.keras')