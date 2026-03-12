import os
import requests
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

def download_file(url, dest_path):
    if dest_path.exists():
        print(f"✅ {dest_path.name} already exists.")
        return
    
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

    # 1. YOLOv8 Nano (Person Detection)
    print("\n--- Computervision (YOLOv8-nano) ---")
    try:
        model = YOLO("yolov8n.pt")  # Auto-downloads
        src = Path("yolov8n.pt")
        dst = models_dir / "yolov8n.pt"
        if src.exists() and not dst.exists():
            src.rename(dst)
        print("✅ YOLOv8-nano ready.")
    except Exception as e:
        print(f"❌ Error downloading YOLO: {e}")

    # 2. OpenCV Face Models (YuNet + SFace)
    print("\n--- Face Recognition (OpenCV YuNet + SFace) ---")
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
