# This was trained on a T4 GPU with 15GB of VRAM. If your system has a different VRAM size, kindly adjust the parameters(for example, batch size) to fit as much training data as possible

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import gc # The Python Garbage Collector

print("Executing Memory-Optimized Data Loading...")

# 1. MEMORY MAPPING: We open the file, but DO NOT load it into RAM yet.
X_mapped = np.load("X_data.npy", mmap_mode='r')
y_full = np.load("y_data.npy")
classes = np.load("classes.npy")
num_classes = len(classes)

# 2. We split a list of "Index Numbers" instead of splitting the heavy images
indices = np.arange(len(X_mapped))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=y_full)

print(f"Total augmented dataset size: {len(X_mapped)} chunks.")
print("Compressing and pulling Train/Test sets into RAM...")

# 3. COMPRESSION: We load the exact images we need and instantly shrink
# them to float32 format, cutting their RAM footprint in half!
X_train = np.array(X_mapped[train_idx], dtype=np.float32)
y_train = y_full[train_idx]

X_test = np.array(X_mapped[test_idx], dtype=np.float32)
y_test = y_full[test_idx]

# 4. GARBAGE COLLECTION: We delete all the temporary variables and force
# the laptop to empty the trash, freeing up Gigabytes of RAM for the GPU.
del X_mapped, y_full, indices, train_idx, test_idx
gc.collect()
print("RAM successfully cleared. Proceeding to Model Training!")

# --- The rest of your exact Fine-Tuning Code ---

# Safer Class Weights
y_integers = np.argmax(y_train, axis=1)
raw_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_integers), y=y_integers)
class_weights = {i: min(weight, 5.0) for i, weight in enumerate(raw_weights)}

print("Building Fine-Tuned MobileNetV3 Architecture...")
base_model = tf.keras.applications.MobileNetV3Large(
    input_shape=(128, 157, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg'
)

# Unfreeze the top layers
base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

model = models.Sequential([
    base_model,
    layers.Dense(512, activation='relu'), # Increased to 512
    layers.Dropout(0.4),
    layers.Dense(256, activation='relu'), # Increased to 256
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
)

callbacks = [
    ModelCheckpoint("best_pulmosense_model.keras", save_best_only=True, monitor="val_loss"),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1),
    EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
]

print("Starting GPU Fine-Tuning...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32, # Lowered from 64 to 32 to prevent GPU memory crashes!
    class_weight=class_weights,
    callbacks=callbacks
)

# Evaluate all metrics
results = model.evaluate(X_test, y_test, verbose=0)
print(f"Final Test Accuracy: {results[1] * 100:.2f}%")
print(f"Final Precision: {results[2] * 100:.2f}%")
print(f"Final Sensitivity (Recall): {results[3] * 100:.2f}%")

print("Converting model to TensorFlow Lite for mobile deployment...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('pulmosense.tflite', 'wb') as f:
    f.write(tflite_model)

print("Success! pulmosense.tflite is ready.")
