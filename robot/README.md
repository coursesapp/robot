# Local-First Social AI Agent

A fully offline, modular AI agent that perceives the environment via camera, recognizes people, and interacts socially through voice.

**Updated Version:** Now uses **OpenCV** (Face Recognition) and **Ollama** (LLM) for easy installation on Windows without complex build tools.

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- A webcam
- A microphone and speakers
- **[Ollama](https://ollama.com/)** installed and running.

### Installation

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Models**
   Run the helper script to fetch YOLO, OpenCV Face Models, and Piper TTS:
   ```bash
   python models/download_models.py
   ```

3. **Setup LLM (Ollama)**
   - Install Ollama from [ollama.com](https://ollama.com).
   - Open a terminal and pull a model (e.g., Phi-3 or Llama3):
     ```bash
     ollama pull phi3
     ```
   - Keep Ollama running in the background.

4. **Install Piper TTS (Optional)**
   Download the [Piper binary](https://github.com/rhasspy/piper/releases) and add it to your PATH. If missing, the agent will use system TTS.

### Configuration
Edit `config/settings.yaml` if needed. By default, it uses:
- Camera 0
- Ollama model `phi3`
- OpenCV Face models in `models/`

## 🏃 Running the Agent

Start the main loop:
```bash
python main.py
```

## 🧪 Verification

1. **Vision Check**: Ensure camera opens. Logs will show `[DETECTOR] person detected`.
2. **Face ID**: It will use OpenCV to detect/recognize faces. First time seeing you -> New ID.
3. **Dialogue**: Speak to it. It sends text to Ollama and speaks back the response.

## 📂 Project Structure

- `core/`: Main loop and event bus
- `vision/`: Camera, YOLO detector, OpenCV Face embedding
- `audio/`: STT (Whisper) and TTS (Piper)
- `dialogue/`: LLM Client (Ollama)
- `memory/`: SQLite + FAISS database (128-d vectors)
- `models/`: Local model files
