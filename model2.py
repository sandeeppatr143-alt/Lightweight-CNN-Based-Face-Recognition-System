import os
import json
import numpy as np
import face_recognition
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, ReLU
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# ===========================
# CONFIGURATION
# ===========================
DATASET_PATH = r"C:\Users\sande\OneDrive\Desktop\New folder\dataset_new"
FACE_MATCH_THRESHOLD = 0.6
EPOCHS = 50
BATCH_SIZE = 8
IMG_ENCODING_SIZE = 128    # face_recognition generates 128 values

# ===========================
# STEP 1: LOAD embedding DATA
# ===========================

X = []
y = []
label_map = {}
label_counter = 0

print("\n🔍 Extracting face embeddings using face_recognition...\n")

for person in os.listdir(DATASET_PATH):
    person_dir = os.path.join(DATASET_PATH, person)

    if not os.path.isdir(person_dir):
        continue

    print(f"→ Processing: {person}")

    for root, _, files in os.walk(person_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(root, file)
                try:
                    img = face_recognition.load_image_file(path)
                    enc = face_recognition.face_encodings(img)

                    if len(enc) > 0:
                        X.append(enc[0])
                        if person not in label_map:
                            label_map[person] = label_counter
                            label_counter += 1

                        y.append(label_map[person])

                except Exception as e:
                    print(f"⚠ Error loading {path}: {e}")

# convert list → numpy arrays
X = np.array(X)
y = np.array(y)

print(f"\n✅ Loaded {len(X)} face embeddings")

# ===========================
# STEP 2: TRAIN-TEST-SPLIT
# ===========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# reshape for CNN  (128,1,1)
X_train = X_train.reshape(-1, IMG_ENCODING_SIZE, 1, 1)
X_test = X_test.reshape(-1, IMG_ENCODING_SIZE, 1, 1)

y_train_cat = to_categorical(y_train, num_classes=len(label_map))
y_test_cat = to_categorical(y_test, num_classes=len(label_map))

# ===========================
# STEP 3: BUILD CNN MODEL
# ===========================

model = Sequential([
    Conv2D(32, (3, 1), input_shape=(128, 1, 1)), ReLU(), MaxPooling2D((2, 1)),
    Conv2D(64, (3, 1)), ReLU(), MaxPooling2D((2, 1)),
    Flatten(),
    Dense(64), ReLU(),
    Dropout(0.3),
    Dense(len(label_map), activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("\n🚀 Training CNN classifier...\n")
model.fit(X_train, y_train_cat, validation_data=(X_test, y_test_cat),
          epochs=EPOCHS, batch_size=BATCH_SIZE)

# ===========================
# STEP 4: EXPORT TFLITE
# ===========================

print("\n📦 Converting to TensorFlow Lite...\n")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("face_embedding_cnn.tflite", "wb") as f:
    f.write(tflite_model)

json.dump(label_map, open("label_map.json", "w"))

print("\n✅ Saved:")
print("➡ face_embedding_cnn.tflite")
print("➡ label_map.json")