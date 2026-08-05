import os
import cv2
import numpy as np
import face_recognition
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, ReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# ==============================
# CONFIGURATION
# ==============================
DATASET_PATH = r"C:\Users\sande\OneDrive\Desktop\New folder\dataset_new"
IMG_SIZE = (100, 100)
EPOCHS = 50
BATCH_SIZE = 8
FACE_MATCH_THRESHOLD = 0.6

# ==============================
# STEP 1: LOAD DATA
# ==============================
X, y, labels = [], [], []
label_map = {}
label_counter = 0

print("🔍 Loading dataset using face_recognition...")

for person in os.listdir(DATASET_PATH):
    person_dir = os.path.join(DATASET_PATH, person)
    if not os.path.isdir(person_dir):
        continue

    print(f"→ Processing {person}...")
    for subdir, _, files in os.walk(person_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(subdir, file)
                try:
                    image = face_recognition.load_image_file(path)
                    encodings = face_recognition.face_encodings(image)

                    if len(encodings) > 0:
                        face_encoding = encodings[0]
                        X.append(face_encoding)

                        if person not in label_map:
                            label_map[person] = label_counter
                            labels.append(person)
                            label_counter += 1
                        y.append(label_map[person])

                except Exception as e:
                    print(f"   ⚠ Error loading {path}: {e}")

X = np.array(X)
y = np.array(y)

# ==============================
# STEP 2: TRAIN-TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train_cat = to_categorical(y_train, num_classes=len(labels))
y_test_cat = to_categorical(y_test, num_classes=len(labels))

# Reshape encodings for CNN input
X_train = X_train.reshape(-1, 128, 1, 1)
X_test = X_test.reshape(-1, 128, 1, 1)

# ==============================
# STEP 3: BUILD LIGHT CNN (<75k parameters)
# ==============================
model = Sequential([
    Conv2D(16, (3, 1), input_shape=(128, 1, 1)), ReLU(), MaxPooling2D((2, 1)),
    Conv2D(32, (3, 1)), ReLU(), MaxPooling2D((2, 1)),
    Flatten(),
    Dense(64), ReLU(),
    Dropout(0.3),
    Dense(len(labels), activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ==============================
# STEP 4: TRAIN MODEL
# ==============================
print("\n🚀 Training CNN classifier...\n")
history = model.fit(X_train, y_train_cat,
                    validation_data=(X_test, y_test_cat),
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    verbose=1)

# ==============================
# STEP 5: EVALUATION + CONFUSION MATRIX
# ==============================
print("\n📊 Evaluating on test set...")

test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"✅ Final Test Accuracy: {test_acc * 100:.2f}%")

# Predict class labels
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test_cat, axis=1)

# Generate confusion matrix
cm = confusion_matrix(y_true_labels, y_pred_labels)
print("\n📌 Classification Report:")
print(classification_report(y_true_labels, y_pred_labels, target_names=labels))

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Parameter count
param_count = np.sum([np.prod(v.shape) for v in model.trainable_weights])
print(f"🧠 Total Trainable Parameters: {param_count}")

# Plot accuracy and loss
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title('Accuracy over Epochs')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss over Epochs')
plt.show()

# ==============================
# STEP 6: REAL-TIME RECOGNITION
# ==============================
print("\n🎥 Starting real-time recognition. Press 'q' to quit.\n")
video_capture = cv2.VideoCapture(0)
reverse_label_map = {v: k for k, v in label_map.items()}

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("⚠ Could not access webcam.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        face_input = face_encoding.reshape(-1, 128, 1, 1)
        predictions = model.predict(face_input, verbose=0)
        class_id = np.argmax(predictions)
        confidence = np.max(predictions)

        if confidence > 0.55:
            name = reverse_label_map[class_id]
            color = (0, 255, 0)
        else:
            name = "Unknown"
            color = (0, 0, 255)

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, f"{name} ({confidence*100:.1f}%)", (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

    cv2.imshow('Real-Time Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
