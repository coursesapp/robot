#!/usr/bin/env python3
"""
TalkNet-ASD Webcam - Fixed Version with Better Audio Detection
"""

import cv2
import torch
import numpy as np
import python_speech_features
import pyaudio
import threading
import queue
from collections import deque
from model.faceDetector.s3fd import S3FD
from talkNet import talkNet
import warnings
warnings.filterwarnings("ignore")

# ========== Optimized Settings ==========
CAMERA_ID = 0
AUDIO_RATE = 16000
AUDIO_CHUNK = 1600  # Larger size = more audio

# ========== ADJUSTABLE THRESHOLD ==========
SPEAKING_THRESHOLD = -0.5  # More sensitive threshold (try: -0.7, -0.5, -0.3, 0.0)

# Global variables
audio_queue = queue.Queue()
video_frames = deque(maxlen=75)
audio_buffer = []  # Use list instead of deque for flexibility
is_running = True
audio_level_history = deque(maxlen=100)  # Track audio levels

# ========== Loading Models ==========
print("⏳ Loading models...")
face_detector = S3FD(device='cuda')
talknet_model = talkNet()
talknet_model.loadParameters('pretrain_TalkSet.model')
talknet_model.eval()
print("✓ Models loaded\n")

# ========== Audio Recording Function ==========
def audio_capture():
    p = pyaudio.PyAudio()
    
    # Print available audio devices
    print("\n📢 Available Audio Devices:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            print(f"  [{i}] {info['name']} - Channels: {info['maxInputChannels']}")
    
    default_input = p.get_default_input_device_info()
    print(f"\n✓ Using: {default_input['name']}\n")
    
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=AUDIO_RATE,
        input=True,
        frames_per_buffer=AUDIO_CHUNK
    )
    
    print("✓ Microphone is working")
    
    while is_running:
        try:
            data = stream.read(AUDIO_CHUNK, exception_on_overflow=False)
            audio_array = np.frombuffer(data, dtype=np.int16)
            
            # Calculate audio level (RMS)
            audio_level = np.sqrt(np.mean(np.abs(audio_array.astype(np.float32))**2))
            audio_level_history.append(audio_level)
            
            audio_queue.put(audio_array)
        except Exception as e:
            print(f"Audio capture error: {e}")
            break
    
    stream.stop_stream()
    stream.close()
    p.terminate()

# Start audio recording
audio_thread = threading.Thread(target=audio_capture, daemon=True)
audio_thread.start()

# Wait to collect enough data
import time
time.sleep(2)

# ========== Opening Camera ==========
cap = cv2.VideoCapture(CAMERA_ID)
cap.set(cv2.CAP_PROP_FPS, 25)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("✓ Camera is working\n")
print("="*60)
print("🎤 Speak in front of the camera and watch the color!")
print("🟢 Green = Speaking | 🔴 Red = Silent")
print(f"📊 Current threshold: {SPEAKING_THRESHOLD}")
print("="*60)
print("\n💡 TIPS:")
print("  - Speak LOUDER and CLOSER to the microphone")
print("  - Make sure you're visible in the camera")
print("  - Watch the 'Audio Level' - it should spike when you speak")
print("  - Your audio level is VERY LOW - check Windows sound settings!")
print("  - Press 'q' to quit")
print("="*60 + "\n")

frame_count = 0
detected_faces = []
last_score = 0.0
detection_success = 0
score_history = deque(maxlen=100)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        display_frame = frame.copy()
        
        # Face detection
        if frame_count % 5 == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            bboxes = face_detector.detect_faces(rgb_frame, conf_th=0.8, scales=[0.5])
            detected_faces = bboxes
        
        # Collect audio
        while not audio_queue.empty():
            try:
                audio_data = audio_queue.get_nowait()
                audio_buffer.extend(audio_data)
                
                # Keep only the last 3 seconds
                if len(audio_buffer) > 48000:
                    audio_buffer = audio_buffer[-48000:]
            except:
                break
        
        # Process faces
        for bbox in detected_faces:
            x1, y1, x2, y2 = bbox[:4].astype(int)
            
            # Extract face
            try:
                face = frame[max(0, y1):min(frame.shape[0], y2), 
                           max(0, x1):min(frame.shape[1], x2)]
                
                if face.size > 0:
                    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    face_resized = cv2.resize(face_gray, (224, 224))
                    face_cropped = face_resized[56:168, 56:168]
                    video_frames.append(face_cropped)
            except:
                pass
            
            # Attempt detection
            score = last_score
            
            if len(video_frames) >= 25 and len(audio_buffer) >= 1600:
                try:
                    # Prepare data
                    video_array = np.array(list(video_frames)[-25:])
                    audio_array = np.array(audio_buffer[-1600:], dtype=np.float32)
                    
                    # Calculate MFCC
                    mfcc = python_speech_features.mfcc(
                        audio_array, 
                        AUDIO_RATE, 
                        numcep=13, 
                        winlen=0.025, 
                        winstep=0.010
                    )
                    
                    # Adjust size
                    if mfcc.shape[0] < 100:
                        # If audio is less than 100, repeat it
                        repeats = int(np.ceil(100 / mfcc.shape[0]))
                        mfcc = np.tile(mfcc, (repeats, 1))[:100, :]
                    else:
                        mfcc = mfcc[:100, :]
                    
                    # TalkNet
                    with torch.no_grad():
                        inputA = torch.FloatTensor(mfcc).unsqueeze(0).cuda()
                        inputV = torch.FloatTensor(video_array).unsqueeze(0).cuda()
                        
                        embedA = talknet_model.model.forward_audio_frontend(inputA)
                        embedV = talknet_model.model.forward_visual_frontend(inputV)
                        embedA, embedV = talknet_model.model.forward_cross_attention(embedA, embedV)
                        out = talknet_model.model.forward_audio_visual_backend(embedA, embedV)
                        
                        score_tensor = talknet_model.lossAV.forward(out, labels=None)
                        score = float(score_tensor[0])
                        last_score = score
                        detection_success += 1
                        score_history.append(score)
                        
                        # Print every 30 detections with statistics
                        if detection_success % 30 == 0:
                            avg_score = np.mean(list(score_history)[-30:])
                            max_score = np.max(list(score_history)[-30:])
                            min_score = np.min(list(score_history)[-30:])
                            
                            # Safe audio level calculation
                            if len(audio_level_history) > 0:
                                recent_audio = list(audio_level_history)[-30:]
                                avg_audio = np.mean([x for x in recent_audio if not np.isnan(x)])
                            else:
                                avg_audio = 0.0
                            
                            print(f"✓ Detection #{detection_success}:")
                            print(f"  Current Score: {score:.3f}")
                            print(f"  Avg Score (last 30): {avg_score:.3f}")
                            print(f"  Range: [{min_score:.3f}, {max_score:.3f}]")
                            print(f"  Avg Audio Level: {avg_audio:.1f}")
                            
                            # Warning for low audio
                            if avg_audio < 100:
                                print(f"  ⚠️  WARNING: Audio is VERY LOW! Increase microphone volume!")
                            
                            print()
                
                except Exception as e:
                    if detection_success % 50 == 0:
                        print(f"❌ Error: {e}")
            
            # Draw box with adjustable threshold
            is_speaking = score > SPEAKING_THRESHOLD
            color = (0, 255, 0) if is_speaking else (0, 0, 255)
            thickness = 5 if is_speaking else 2
            
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)
            
            status = "SPEAKING" if is_speaking else "SILENT"
            cv2.putText(
                display_frame, 
                f"{status} ({score:.2f})", 
                (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, 
                color, 
                2
            )
        
        # Calculate current audio level (safe)
        if len(audio_level_history) > 0:
            current_audio_level = list(audio_level_history)[-1]
            if np.isnan(current_audio_level):
                current_audio_level = 0.0
            avg_audio_level = np.nanmean(list(audio_level_history))
            if np.isnan(avg_audio_level):
                avg_audio_level = 0.0
        else:
            current_audio_level = 0.0
            avg_audio_level = 0.0
        
        # On-screen information
        info = [
            f"Video: {len(video_frames)}/25",
            f"Audio Buffer: {len(audio_buffer)}/1600",
            f"Detections: {detection_success}",
            f"Score: {last_score:.3f} (Threshold: {SPEAKING_THRESHOLD})",
            f"Audio Level: {current_audio_level:.0f} (Avg: {avg_audio_level:.0f})",
        ]
        
        # Add score statistics if available
        if len(score_history) > 0:
            avg_score = np.mean(list(score_history))
            info.append(f"Avg Score: {avg_score:.3f}")
        
        # Add warning for low audio
        if avg_audio_level < 100:
            info.append("WARNING: Audio LOW!")
        
        y = 25
        for line in info:
            cv2.putText(display_frame, line, (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            y += 20
        
        # Draw audio level bar (safe)
        try:
            bar_width = int((current_audio_level / 5000) * 200)  # Scale to 200px max
            bar_width = max(0, min(bar_width, 200))  # Clamp between 0-200
            cv2.rectangle(display_frame, (10, display_frame.shape[0] - 30), 
                         (10 + bar_width, display_frame.shape[0] - 10), 
                         (0, 255, 0), -1)
        except:
            pass  # Skip if calculation fails
        
        cv2.putText(display_frame, "Audio Level", (10, display_frame.shape[0] - 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('TalkNet ASD - Improved', display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

finally:
    is_running = False
    cap.release()
    cv2.destroyAllWindows()
    
    # Final statistics
    print(f"\n{'='*60}")
    print(f"✓ Completed successfully!")
    print(f"{'='*60}")
    print(f"Total Detections: {detection_success}")
    
    if len(score_history) > 0:
        scores = list(score_history)
        print(f"\n📊 Score Statistics:")
        print(f"  Average: {np.mean(scores):.3f}")
        print(f"  Min: {np.min(scores):.3f}")
        print(f"  Max: {np.max(scores):.3f}")
        print(f"  Std Dev: {np.std(scores):.3f}")
        
        # Test multiple thresholds
        print(f"\n💡 Detection Results with Different Thresholds:")
        for thresh in [-0.7, -0.5, -0.3, 0.0]:
            speaking_count = sum(1 for s in scores if s > thresh)
            percentage = (speaking_count / len(scores)) * 100
            print(f"  Threshold {thresh:+.1f}: {speaking_count}/{len(scores)} ({percentage:.1f}%)")
        
        print(f"\n📈 Score Percentiles (for finding optimal threshold):")
        percentiles = [10, 25, 50, 75, 90]
        for p in percentiles:
            val = np.percentile(scores, p)
            print(f"  {p}th percentile: {val:.3f}")
    
    if len(audio_level_history) > 0:
        levels = [x for x in audio_level_history if not np.isnan(x)]
        if levels:
            print(f"\n🔊 Audio Level Statistics:")
            print(f"  Average: {np.mean(levels):.1f}")
            print(f"  Min: {np.min(levels):.1f}")
            print(f"  Max: {np.max(levels):.1f}")
            
            if np.mean(levels) < 100:
                print(f"\n⚠️  CRITICAL: Audio levels are EXTREMELY LOW!")
                print(f"  SOLUTION:")
                print(f"  1. Right-click speaker icon in Windows taskbar")
                print(f"  2. Select 'Open Sound settings'")
                print(f"  3. Click 'Device properties' under Input")
                print(f"  4. Increase volume slider to 80-100%")
                print(f"  5. Check 'Additional device properties'")
                print(f"  6. Go to 'Levels' tab and increase microphone boost")
    
    print(f"{'='*60}")