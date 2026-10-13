# Lightweight CNN Face Recognition Prototype

A research prototype that extracts 128-dimensional face embeddings with face_recognition and trains a small Keras CNN classifier. The repository also contains a TensorFlow Lite export script and an optional helper for downloading Linux ARM64 wheels for a PYNQ environment.

> **Status:** Educational prototype. It has not been independently evaluated for accuracy, fairness, security, or suitability for real-world identity decisions. Do not use it for surveillance, access control, law enforcement, or other consequential decisions.

## Repository contents

| File | Purpose |
| --- | --- |
| model1.py | Train and evaluate the CNN classifier; display a confusion matrix and training curves; includes a webcam recognition loop. |
| model2.py | Train a classifier from face embeddings and export a TensorFlow Lite model plus a label map. |
| model3.py | Optional downloader for prebuilt Python wheels targeting a specific PYNQ/Linux ARM64 setup. Review source URLs and platform compatibility before running. |
| Generated outputs | The model and label map are produced locally by model2.py and should not be committed when they encode or identify real people. |

## How it works

The training scripts expect a private image dataset arranged as one folder per identity, with image files inside each folder (subfolders are also searched). They extract face encodings, split the resulting samples into training and validation sets, and train a CNN over the embeddings. model2.py exports the model and label map.

Set the FACE_DATASET_PATH environment variable to your dataset directory before running a training script. Do not commit face images, embeddings, identity labels, or other biometric data to a public repository unless every affected person has explicitly agreed to that exact public use.

## Environment

The training code uses Python with TensorFlow/Keras, face_recognition (and its dlib dependency), OpenCV, NumPy, scikit-learn, Matplotlib, and Seaborn. Install versions compatible with your operating system and Python version; dlib and TensorFlow may require platform-specific setup.

The scripts do not currently provide a pinned, verified dependency lockfile. The optional model3.py helper targets CPython 3.10 on Linux ARM64 and is separate from model training.

## Running

1. Create a private dataset in the folder-per-identity layout described above.
2. Set FACE_DATASET_PATH to the dataset directory.
3. Install compatible dependencies in an isolated Python environment.
4. Run python model2.py to train and export the TensorFlow Lite model, or python model1.py for the evaluation and webcam demonstration.

The webcam path requires a working camera and a local graphical environment. Generated model files and identity labels should remain private when they encode or identify real people.

## Privacy and responsible use

Face images, face embeddings, and identity labels are sensitive biometric information. Keep datasets and generated identity artifacts private, obtain meaningful consent, and follow applicable privacy and biometric laws. Do not use this prototype to identify people without their knowledge and permission.

## Limitations

- The scripts use a random sample split and do not document a held-out person-level evaluation protocol.
- Metrics from a small or unrepresentative dataset should not be treated as evidence of general performance.
- The webcam loop and model export have not been validated across devices or deployment environments.
- This repository does not include a production API, access-control safeguards, or a model card.

## License

No license is currently specified. Unless a license is added, the default copyright rules apply and reuse permissions are not granted.
