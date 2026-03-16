import onnxruntime as ort
import os
from pathlib import Path

models = [
    "models/facenet_resnet34.onnx",
    "models/mtcnn_pnet.onnx",
    "models/mtcnn_rnet.onnx",
    "models/mtcnn_onet.onnx"
]

print("🔍 Verifying ONNX models...")
for m in models:
    path = Path(m)
    if not path.exists():
        print(f"❌ {m} is MISSING")
        continue
    
    size = path.stat().st_size
    print(f"File: {m} | Size: {size/1024:.2f} KB")
    
    if size < 1024 * 10: # Less than 10KB is definitely wrong
        print(f"⚠️ {m} is TOO SMALL. Likely a failed download or HTML page.")
        continue

    try:
        sess = ort.InferenceSession(m, providers=['CPUExecutionProvider'])
        print(f"✅ {m} LOADED successfully.")
    except Exception as e:
        print(f"❌ {m} FAILED TO LOAD: {e}")
        # Peek at the file content
        with open(m, 'r', errors='ignore') as f:
            content = f.read(100)
            if "<!DOCTYPE html>" in content or "<html>" in content:
                 print(f"   🚨 REASON: The file is an HTML page, not an ONNX model!")
