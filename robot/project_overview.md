# Local-First AI Agent: Project Documentation

This document describes the **Local-First AI Agent**, a fully offline, privacy-focused social robot designed to run on consumer-grade hardware. It perceives the world, remembers interactions, and converses naturally—all without internet connectivity.

## 🎯 Core Philosophy
- **Offline-First**: Zero dependency on cloud APIs. Works in a disconnected environment.
- **Privacy-Centric**: All data (face embeddings, conversation history) lives on the local device.
- **Modular**: Components (Vision, Audio, LLM) are loosely coupled via an event bus.
- **Resource-Efficient**: Optimized models (Quantized LLMs, INT8 Vision) for CPU/edge deployment.

---

## 🏗 System Architecture

The agent operates on a continuous **Perceive-Understand-Decide-Act** loop running at ~15 FPS.

### 1. Perception Layer (Sensors)
- **Vision (`vision/`)**:
  - **Camera**: captures video frames.
  - **Detector (YOLOv8-nano)**: Identifies people and objects in real-time.
  - **Face Recognition (InsightFace)**: Extracts 512-d embeddings to identify individuals.
  - **Emotion Recognition (AffectNet)**: Detects facial expressions (Happy, Sad, Neutral, etc.).
  - **Tracker**: Follows people across frames to maintain identity stability.

- **Hearing (`audio/stt.py`)**:
  - **Microphone**: captures audio.
  - **VAD (Voice Activity Detection)**: Ignores silence and background noise.
  - **STT (Faster-Whisper)**: Transcribes speech to text locally.

### 2. Cognition Layer (Brain)
- **Context Awareness**: Aggregates sensory data (Who is here? What are they feeling? What did they say?).
- **Prompt Engine (`dialogue/prompt_engine.py`)**: Converts raw context into a structured prompt for the LLM.
- **LLM (Phi-3-Mini / Llama-3)**: Generates intelligent, persona-driven responses.
- **Memory (`memory/`)**:
  - **Short-term**: Conversation history in RAM.
  - **Long-term (SQLite + Vector DB)**: 
    - **Identity Store**: "Person A seen on [Date]".
    - **Social Memory**: "Person A likes coding".

### 3. Action Layer (Expression)
- **TTS (Piper)**: Converts LLM text response to natural-sounding speech.
- **Physical Actions** (Future): Controlling motors or expressions on a screen.

---

## 📂 Key Components

| Module | Technology | Role |
|---|---|---|
| **Vision** | YOLOv8n, ArcFace | "Eyes" – Detects and recognizes people. |
| **Audio** | Faster-Whisper, Piper | "Ears & Mouth" – Hearing and speaking. |
| **Brain** | Phi-3-Mini (GGUF) | "Mind" – Reasoning and conversation. |
| **Memory** | FAISS, SQLite | "Memory" – Recognizing faces and facts. |
| **Infrastructure** | Python, Threading | Glues everything together. |

---

## 💾 Data Flow

1. **User walks in**: Camera sees a person → Tracker assigns ID #1.
2. **Face Check**: System extracts face embedding → Checks FAISS DB.
3. **Recognition**: DB returns "Person #1 is Mohamed".
4. **Interaction**: User says "Hello".
5. **Processing**: 
   - STT transcribes "Hello".
   - Context built: `{User: Mohamed, Emotion: Happy, Text: "Hello"}`.
   - LLM generatess: "Hi Mohamed! Good to see you again."
6. **Response**: TTS speaks the response.

---

## 🚀 Future Roadmap

- [ ] **Visual Memory**: Remembering where objects were placed ("Where are my keys?").
- [ ] **RAG (Retrieval Augmented Generation)**: Searching documents/notes.
- [ ] **Gesture Control**: Recognizing hand wayes or pointing.
- [ ] **Home Assistant Integration**: Controlling smart lights via voice.

---

## 🛠 Hardware Requirements

- **Minimum**: 
  - CPU: 4 Cores (modern Intel/AMD/ARM)
  - RAM: 8 GB
  - Storage: 10 GB
- **Recommended**:
  - GPU: NVIDIA RTX 3050+ (for faster LLM/Vision)
  - RAM: 16 GB

---

## 🛡 Ethical & Privacy Standards

1. **Consent Gate**: The agent explicitly asks for permission before remembering a face.
2. **Right to Forget**: A "Forget me" command permanently wipes user data.
3. **Transparency**: Visual indicator (LED/Screen) when recording or processing.

---
*Generated for Local-First AI Agent Project*
