import os
import requests

download_dir = r"C:\PYNQ_packages\pynq_pkgs"
os.makedirs(download_dir, exist_ok=True)

files = {
    "dlib-19.24.2-cp310-cp310-linux_aarch64.whl":
        "https://github.com/PINTO0309/dlib-build/raw/main/wheels/CP310/UbuntuARM64/dlib-19.24.2-cp310-cp310-linux_aarch64.whl",

    "opencv_python-4.7.0.72-cp310-cp310-linux_aarch64.whl":
        "https://www.piwheels.org/simple/opencv-python/opencv_python-4.7.0.72-cp310-cp310-linux_aarch64.whl",

    "tflite_runtime-2.14.0-cp310-cp310-linux_aarch64.whl":
        "https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.14.0-cp310-cp310-linux_aarch64.whl",

    "face_recognition-1.3.0-py2.py3-none-any.whl":
        "https://files.pythonhosted.org/packages/33/e4/1c79f8558d6c0e0ebdb761d01b7cc47c8f520ae568dcc9c888e512190289/face_recognition-1.3.0-py2.py3-none-any.whl",
}

print("\n⬇ Downloading wheel files...\n")

for filename, url in files.items():
    file_path = os.path.join(download_dir, filename)
    print(f"📥 Downloading: {filename}")
    
    response = requests.get(url, allow_redirects=True)
    if response.status_code != 200:
        print(f"❌ Failed to download {filename} ({response.status_code})\n")
        continue

    with open(file_path, "wb") as f:
        f.write(response.content)

    print(f"✅ Saved: {file_path}  ({round(os.path.getsize(file_path) / (1024*1024), 2)} MB)\n")

print("\n🎉 DONE. Copy these files to PYNQ via USB and pip install them.\n")
