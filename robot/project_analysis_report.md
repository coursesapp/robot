# Project Analysis Report: Local-First Social AI Agent

## 📌 Overview
The project is a **modular, local-first social AI agent** designed to interact with users through vision and voice. It prioritizes privacy and offline capability by running all heavy models (LLM, Vision, STT/TTS) locally on the host machine.

## 🏗 System Architecture
The agent follows a **Perceive-Understand-Decide-Act** loop, implemented in `AgentLoop` (`core/agent_loop.py`).

### 1. Perception Layer (Sensors)
- **Vision (`vision/`)**:
  - **Detection & Tracking**: Uses YOLOv8 (nano) specifically optimized for people and common objects (`detector.py`).
  - **Face Identification**: Implements OpenCV's `CV_Face` (YuNet for detection, SFace for recognition) to handle 128-d embeddings (`face_embedding.py`).
  - **Emotion Detection**: Predicts facial expressions (Happy, Sad, etc.) using an emotion classifier (`emotion.py`).
- **Audio (`audio/`)**:
  - **STT (Speech-to-Text)**: Powered by `faster-whisper` for low-latency local transcription (`stt.py`).
  - **Speaker ID**: Uses CAM++ (VoxCeleb) to correlate voices with visual identities (`speaker_id.py`).

### 2. Cognition Layer (Brain)
- **LLM Client (`dialogue/llm_client.py`)**: Interfaces primarily with **Ollama** (e.g., `phi3`) or a local `llama-server`.
- **Interaction Strategies (`core/`)**:
  - **`LocalStrategy`**: Handles the sequential flow: Perception -> JSON-structured Prompt -> LLM response -> TTS.
  - **`GeminiLiveStrategy`**: A placeholder/alternative for cloud-based multimodal interaction.
- **Prompt Engineering**: The `PromptEngine` builds complex prompts that include:
  - Current detected identities and their social context.
  - Spatial information of objects (e.g., "cup is on the bottom-right").
  - Emotional state of the user.
  - Interaction history and "Deep Memory" (RAG).

### 3. Memory Layer (`memory/`)
- **Identity Store**: Manages FAISS indices for face and voice embeddings to recognize returning users.
- **Social Memory**: A SQLite database storing person-specific data (name, interests, last seen).
- **Vector Memory**: Uses **ChromaDB** for long-term storage of conversation segments, enabling Retrieval-Augmented Generation (RAG).

### 4. Action Layer
- **TTS (Text-to-Speech)**: Uses **Piper** for high-quality, local voice synthesis (`audio/tts.py`).
- **Action Library**: A framework for the LLM to trigger specific system actions via JSON declarations in its response.

## 🔄 Interaction Flow
1. **Vision Worker** (Thread) constantly tracks people and updates a shared state.
2. **STT** listens for speech. When detected:
   - It performs **Audio-Visual Fusion**: checks who is talking by correlating mouth movement (vision) with speaker identification (audio).
   - It builds a **Context Object** (Who is here? What are they feeling? What's the history?).
3. **LLM** processes the context and returns a JSON response containing:
   - `internal_thought`: Chain-of-thought processing.
   - `response`: The text for the user.
   - `actions`: List of system commands to execute.
   - `save_to_memory`: Flag to persist new facts.
4. **TTS** speaks the response while the system executes any requested actions.

## 🛠 Key Technologies
- **Logic**: Python 3.10+, Threading, EventBus.
- **AI/ML**: YOLOv8, OpenCV Face, Faster-Whisper, Piper TTS, FAISS.
- **Database**: SQLite, ChromaDB.
- **UI**: A web-based dashboard on port 5050 (`dashboard/web_dashboard.py`).

## 📋 Observations
- **Robustness**: The project includes sophisticated JSON repair logic for LLM responses and identity voting to ensure stable person tracking.
- **Scalability**: The modular strategy pattern allows switching between local LLMs and cloud providers like Gemini easily.
- **Privacy**: All biometric data (face/voice) is indexed locally.
