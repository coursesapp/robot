import os
import requests
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

def download_file(url, dest_path):
    if dest_path.exists():
        if dest_path.stat().st_size > 1024:
            print(f"✅ {dest_path.name} already exists.")
            return
        else:
            print(f"⚠️  {dest_path.name} seems corrupted/incomplete. Re-downloading...")
    
    print(f"⬇️ Downloading {dest_path.name}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as file, tqdm(
        desc=dest_path.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def main():
    root = Path(__file__).parent.parent
    models_dir = root / "models"
    models_dir.mkdir(exist_ok=True)
    
    print("🚀 Starting model downloads...")

    # 1. YOLOv8 Nano (General Object Detection)
    print("\n--- Computervision (YOLOv8-nano) ---")
    download_file(
        "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt",
        models_dir / "yolov8n.pt"
    )

    # Face Detection (YuNet)
    download_file(
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        models_dir / "face_detection_yunet_2023mar.onnx"
    )
    # Face Recognition (SFace)
    download_file(
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        models_dir / "face_recognition_sface_2021dec.onnx"
    )

    # 4. YOLOv8 Face (Alternative for better people tracking)
    print("\n--- Face Detection (YOLOv8-face) ---")
    download_file(
        "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt",
        models_dir / "yolov8n-face.pt"
    )

    # 5. Speaker Identification (CAM++)
    print("\n--- Speaker Identification (Wespeaker CAM++) ---")
    download_file(
        "https://huggingface.co/Wespeaker/wespeaker-voxceleb-campplus/resolve/main/voxceleb_CAM++.onnx?download=true",
        models_dir / "voxceleb_CAM++.onnx"
    )

    # 3. Piper TTS (Lessac Medium)
    print("\n--- TTS (Piper - Lessac Medium) ---")
    piper_onnx = models_dir / "en_US-lessac-medium.onnx"
    piper_json = models_dir / "en_US-lessac-medium.onnx.json"
    
    # Using HuggingFace mirror for stability
    download_file(
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx?download=true",
        piper_onnx
    )
    download_file(
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json?download=true",
        piper_json
    )

    print("\n--- LLM (Ollama) ---")
    print("ℹ️  Please install Ollama from https://ollama.com/")
    print("ℹ️  Then run: `ollama pull phi3` or `ollama pull llama3`")

    print("\n🎉 All local models checked/downloaded.")
    print(f"Run 'pip install -r requirements.txt' then 'python main.py' to start.")

if __name__ == "__main__":
    main()
