# Aura: The Context-Aware Social AI Agent

> **"Stopping the typing, and starting the interaction."**

Aura is a next-generation social AI companion designed to perceive the physical world, recognize identities, and build lasting relationships through social memory. It leverages a hybrid **Edge-to-Cloud** architecture, using local high-speed sensors (YOLO, OpenCV) and Gemini's multimodal reasoning to create immersive, real-time social experiences.

---

## ✨ Features

- 👁️ **Proactive Vision**: Constant person tracking and face recognition using YOLOv8 and SFace.
- 🎙️ **Audio-Visual Fusion**: Correlates visual mouth movements with voice activity to ensure context-aware addressing.
- 🧠 **Context-First Brain**: Driven by Gemini 2.0 Flash to understand emotion, spatial context, and intent.
- 💾 **Cognitive Memory Architecture**:
  - **Conversational (Short-Term)**: Real-time context management for fluid dialogue.
  - **Social (Persistent)**: Remembers personal facts and user preferences via **Google Cloud Firestore**.
  - **Knowledge (Long-Term)**: Vector-indexed history of all past interactions using **ChromaDB**.
  - **Biometric**: Local-only face/voice embeddings (FAISS) for privacy.
- 📍 **Active Spatial Awareness**: Tracks physical objects over time (e.g., "Where did I leave my keys?").
- 🗣️ **Natural Expression**: High-quality local voice synthesis via Piper TTS.

---

## 🚀 Execution Modes

Aura is designed for flexibility. Configuration is managed in `config/settings.yaml` (which should be created from the provided `.example` file).

### Option 1: Standard Multimodal (Recommended)
*Uses local perception for detection and Gemini Pro/Flash for deep reasoning.*
- **LLM**: Gemini API or Groq.
- **Vision**: Local YOLOv8 + occasional frames to Gemini for scene grounding.
- **STT/TTS**: Local (Whisper/Piper) for privacy and speed.

### Option 2: Gemini Live (Immersive)
*Full real-time multimodal streaming.*
- **Engine**: Gemini Multimodal Live API.
- **Features**: Low-latency audio/video streaming, natural interruptions, and visual grounding.

### Option 3: Privacy-First (Fully Local)
*Works completely offline without cloud dependencies.*
- **LLM**: Ollama / Llama.cpp (running locally).
- **Vision**: Local person tracking and recognition.
- **STT/TTS**: Whisper and Piper.

---

## 🛠 Prerequisites

- **Python 3.10+**
- **Hardware**: Webcam, Microphone, and Speakers.
- **Cloud Setup** (Optional for Mode 1 & 2):
  - Gemini API Key.
  - Google Cloud Project with Firestore enabled.

---

## 📥 Installation

1. **Clone and Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Local Models**
   Run the helper script to fetch YOLO benchmarks and OpenCV Face models:
   ```bash
   python models/download_models.py
   ```

3. **External Services**
   - **Ollama**: If using Local Mode, [Install Ollama](https://ollama.com) and pull a model: `ollama pull phi3`.
   - **Google Cloud**: Set your `GOOGLE_APPLICATION_CREDENTIALS` if using Firestore.

---

## ⚙️ Configuration

1. **Initialize Settings**: Copy the example configuration file to create your active settings:
   ```bash
   cp config/settings.yaml.example config/settings.yaml
   ```

2. **Edit Configuration**: Open `config/settings.yaml` and customize your experience:

```yaml
interaction:
  mode: "local" # options: local, gemini_live
  providers:
    stt: "whisper"
    tts: "piper"
    brain: "gemini" # use "gemini", "groq", or "ollama"

gemini:
  api_key: "YOUR_API_KEY"
  model: "gemini-2.0-flash"
```

---

## 🏃 Running the Agent

Start the Aura core loop:
```bash
python main.py
```

### Command Line Overrides:
- `--config path/to/config.yaml`: Use a custom settings file.
- `--headless`: Run without the GUI/Visual window.
- `--duration N`: Run for N seconds then shutdown.

---

## 📂 Project Structure

- `core/`: Event bus, interaction strategies, and main agent loop.
- `vision/`: YOLO tracking, Face Embeddings, and Emotion classification.
- `audio/`: STT (Whisper), TTS (Piper), and Speaker ID.
- `dialogue/`: Prompt engine and LLM client.
- `memory/`: Identity store, Vector DB, and Social Memory management.
- `config/`: System settings and environment tuning.
- `dashboard/`: Web-based UI (available on [localhost:5050](http://localhost:5050)).

---

## 🛡 Privacy & Ethics
Aura is built with a **Consent-First** approach. Biometric embeddings are stored locally by default, and the agent explicitly requests permission before persisting social memories.

---
*Created for the Gemini Live Agent Challenge*
