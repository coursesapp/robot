# Emotion Detection Pipeline

A dual-library facial emotion detection system using **FER** (for real-time webcam) and **DeepFace** (for static image/offline analysis). Both libraries are used intentionally — each where it performs best.

---

##  Project Structure

```
emotion-detection/
├── realtime_fer.py          # Live webcam emotion detection (FER)
├── static_deepface.py       # Static image / offline analysis (DeepFace)
├── notebook_demo.ipynb      # Comparison notebook (FER vs DeepFace on images)
├── requirements.txt         # Python dependencies
├── .gitignore
└── README.md
```

---

##  Library Choice Rationale

| Use Case | Library | Reason |
|---|---|---|
| Real-time webcam | **FER** | Faster (~100–300ms/frame), smooth on CPU |
| Static images / batch | **DeepFace** | Higher accuracy, richer output, better on varied faces |
| GPU available | Either | GPU closes the speed gap; DeepFace preferred |

---

##  Installation

**Python 3.8–3.10 recommended** (TensorFlow compatibility).

```bash
git clone https://github.com/your-username/emotion-detection.git
cd emotion-detection
pip install -r requirements.txt
```

> **Note:** First run will download model weights automatically (~500MB for DeepFace).

### Optional: GPU Acceleration

If you have a CUDA-capable GPU:

```bash
pip install tensorflow-gpu
```

Then set the environment variable before running:

```bash
export TF_FORCE_GPU_ALLOW_GROWTH=true   # Linux/macOS
set TF_FORCE_GPU_ALLOW_GROWTH=true      # Windows
```

---

##  Usage

### Real-Time Webcam (FER)

```bash
python realtime_fer.py
```

- Opens your default webcam (index `0`)
- Detects and overlays the dominant emotion on each frame
- Press **`q`** to quit

**Configuration** (edit top of file):
```python
CAMERA_INDEX = 0       # Change if using external webcam
FRAME_SKIP = 5         # Analyze every Nth frame (lower = more frequent, slower)
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480
```

---

### Static Image Analysis (DeepFace)

```bash
python static_deepface.py --image path/to/image.jpg
```

Or analyze all images in a folder:

```bash
python static_deepface.py --folder path/to/images/
```

**Output example:**
```
Image: Surprised_17.jpg
  Dominant emotion : surprise
  Confidence       : 99.16%
  All scores       : {'angry': 0.01, 'disgust': 0.0, 'fear': 0.12,
                      'happy': 0.03, 'sad': 0.01, 'surprise': 99.16, 'neutral': 0.67}
```

---

### Notebook Demo

Open `notebook_demo.ipynb` in Jupyter to run FER and DeepFace side-by-side on any image and compare their outputs.

```bash
jupyter notebook notebook_demo.ipynb
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `fer` | Real-time emotion detection |
| `deepface` | Static image emotion analysis |
| `opencv-python` | Webcam capture and frame display |
| `tensorflow` | Backend for both libraries |
| `mtcnn` | Face detector used by FER |
| `numpy` | Array handling |

Full list with pinned versions in `requirements.txt`.

---

##  Known Issues & Limitations

- **First-run delay:** DeepFace downloads model weights on first use. Subsequent runs are fast.
- **CPU performance:** Real-time DeepFace with RetinaFace is slow on CPU — use FER for webcam on CPU-only machines.
- **Single face:** `ferr_model.py` uses `top_emotion()` which returns only the dominant face result. For multi-face scenes, use DeepFace.
- **Lighting sensitivity:** Both libraries perform best with good frontal lighting.

---

##  License

MIT License. See `LICENSE` for details.
