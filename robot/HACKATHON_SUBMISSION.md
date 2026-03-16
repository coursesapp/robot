# Gemini Live Agent Challenge Submission Draft

## Project Details

### * Project name
**Aura: The Context-Aware Social AI**

### * Elevator pitch
A next-generation social AI agent that transcends text boxes to see, hear, and interact with the physical world through an **Immersive Character-Driven UI**, leveraging Gemini Multimodal Live and Google Cloud for persistent, context-aware relationships.

---

## Project Story

## Inspiration
In a world dominated by text-based chatbots, we felt a disconnect. AI should not just live behind a keyboard; it should share our physical space, recognize our faces, and hear the nuances in our voices. We were inspired to build **Aura**—a social agent that moves beyond the text box to create **immersive, context-aware experiences**. Our goal was to create an agent that feels like a presence in the room, capable of building lasting relationships through social memory and multimodal perception.

## What it does
Aura is a multimodal AI companion that perceives its environment through a camera and microphone. It doesn't just wait for prompts; it proactively recognizes people as they enter the room, greets them by name, and remembers past interactions. By combining real-time object detection with Gemini's deep reasoning, Aura can discuss what it sees, identify emotions, and perform intelligent actions based on social context. Crucially, Aura features **Active Spatial Awareness**, allowing it to track the location of physical objects over time and answer queries like "Where did I leave my keys?" by retrieving timestamps and spatial coordinates from its memory.

## How we built it
Aura is built on a hybrid "Edge-to-Cloud" architecture:
1.  **Perception**: Aura features a flexible perception layer with two modes: a **Local-First pipeline** (using **YOLOv8** for tracking, **OpenCV SFace** for identity, and **Faster-Whisper** for STT) and a **Real-Time Streaming mode** that leverages the **Gemini Multimodal Live API** for unified, low-latency visual and auditory grounding.
2.  **Multimodal Brain**: We leverage **Gemini 2.0 Flash** as the core reasoning engine, streaming compressed visual frames and context via the Google GenAI SDK.
3.  **Unified Cognitive Memory**: Aura utilizes a three-tier memory system:
    - **Short-Term/Conversational**: Manages immediate context and recent turns.
    - **Social Memory**: Persists personal facts and relationship data (likes, names, history) via **Google Cloud Firestore**.
    - **Long-Term Knowledge**: Vector-based retrieval of past interactions using **ChromaDB**.
4.  **Deployment**: The entire backend is designed for the **Google Cloud Platform**, utilizing Vertex AI paradigms for grounded responses.

## Challenges we ran into

1. **Multimodal Synchronization & Temporal Latency**: Synchronizing high-frequency visual frames (15+ FPS) with sequential audio streams for real-time interaction is complex. We had to implement a **Dual-Buffer Perception Loop** to ensure Gemini receives a coherent "snapshot" of the environment without overloading the network or causing conversational lag. The target was to minimize the delta between perception and response:
$$ \Delta t_{\text{total}} = \Delta t_{\text{capture}} + \Delta t_{\text{local\_processing}} + \Delta t_{\text{Gemini\_inference}} < 1.5s $$

2. **Identity Fusion & Semantic Conflicts**: One of the toughest hurdles was handling "Identity Bleed"—scenarios where visual cues (a face) and auditory cues (voice ID) contradict each other. We solved this by developing an **Identity Voting Engine** that weights confidence scores from local biometrics and uses Gemini’s reasoning to "mediate" the situation (e.g., "I hear Mohamed’s voice, but I see Ahmed’s face—let me clarify who is speaking").

3. **Grounded Reasoning in Dynamic Environments**: Ensuring the AI doesn't "hallucinate" objects that are no longer there (e.g., the user picks up the cup) required a constant feedback loop between the YOLO tracker and the LLM's spatial context. We moved from static prompts to a **Dynamic Spatial Mapping** system that updates the "world view" for Gemini every few seconds.

4. **Conversational Fluidity & Interruption Handling**: In natural social settings, humans interrupt each other. Handling this without being jarring required an **Asynchronous Stream Controller**. When our VAD (Voice Activity Detection) system detects the user speaking while Aura is mid-sentence, it must instantly kill the TTS playback, flush the audio buffer, and signal Gemini that a "barge-in" has occurred to pivot the context immediately.

## Accomplishments that we're proud of
We are particularly proud of our **Cognitive Memory Architecture**. Aura distinguishes between **Social Memory** (who you are), **Conversational Memory** (what we are talking about now), and **Long-Term Knowledge** (what we have ever discussed). By managing sensitive biometrics locally (FAISS) while syncing social facts to **Google Cloud Firestore**, we achieve cross-session persistence. We also implemented **Proactive Context Compression** to distill long-term interactions into high-density insights for Gemini.

Additionally, our **Audio-Visual Fusion Engine** provides a level of conversational precision that traditional chatbots cannot match by correlating visual mouth movements with voice activity in real-time. We've also finalized a **Character-Driven Web Interface** where the agent's emotions and states are physically manifested through a reactive SVG avatar, making the AI feel presence-driven rather than just code-driven.

## What we learned
We learned that the future of AI is **grounded interaction**. When an AI can "see" the sadness in a face or "remember" a favorite topic, it stops being a tool and starts being a partner. We deeper understood how to manage complex multimodal streams and the importance of social memory in creating "immersion."

## What's next for Aura: The Context-Aware Social AI
Our roadmap includes:
- **Physical Embodiment**: Integrating Aura into smart robot platforms (e.g., TurtleBot, Unitree) to enable physical mobility and interaction.
- **Virtual Social Robots**: Deploying Aura as a 3D virtual assistant in VR/AR environments, maintaining consistent social memory across physical and digital spaces.
- **Home Integration**: Acting as a visual-first controller for smart home environments via Google Home API.
- **Collaborative Storytelling**: Leveraging Gemini's interleaved output to create visual stories with the user based on objects detected in the room.
---

## Built with

- **Languages**: Python 3.10+
- **AI/ML Frameworks**: Gemini 2.0 Flash (Multimodal), Google GenAI SDK, Ultralytics YOLOv8 (Tracking), OpenCV CV_Face (Identity), Faster-Whisper (STT), Piper TTS.
- **APIs**: Gemini Multimodal Live API, Vertex AI.
- **Databases**: Google Cloud Firestore (Social Memory), ChromaDB (Vector Knowledge Base), FAISS (Local Biometric Indexing), SQLite.
- **Cloud Services**: Google Cloud Platform (GCP), Firebase.
- **Infrastructure**: Threaded-Event-Bus, Asynchronous-WebSockets, Flask (Web-Dashboard).

