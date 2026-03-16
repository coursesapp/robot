import vosk
import wave
import json
import os
import jiwer  

# --- SETUP ---
MODEL_LANG = "en-us"
TEST_SET_PATH = "test_set"
AUDIO_FOLDER = os.path.join(TEST_SET_PATH, "audio")
GROUND_TRUTH_FOLDER = os.path.join(TEST_SET_PATH, "ground_truth_transcripts")
PREDICTIONS_FOLDER = os.path.join(TEST_SET_PATH, "predictions")

#CUSTOM_VOCABULARY = ["mahmoud", "phone", "owner", "robot", "keys", "kitchen", "pass"]

# --- STEP 1: PROCESS AUDIO AND SAVE PREDICTIONS ---

# Create the predictions folder if it doesn't exist
os.makedirs(PREDICTIONS_FOLDER, exist_ok=True)

# Load the Vosk model
model = vosk.Model(lang=MODEL_LANG)
print("Vosk model loaded.")
print("-" * 30)

# Get a sorted list of audio files to process them in order
audio_files = sorted([f for f in os.listdir(AUDIO_FOLDER) if f.lower().endswith(".wav")])

for filename in audio_files:
    print(f"Processing {filename}...")
    
    audio_filepath = os.path.join(AUDIO_FOLDER, filename)
    
    # Open the audio file
    try:
        with wave.open(audio_filepath, "rb") as wf:
            # Verify the audio format is correct (16kHz mono)
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE" or wf.getframerate() != 16000:
                print(f"Error: Audio file {filename} is not in WAV format, 16kHz, mono, 16-bit.")
                continue

            recognizer = vosk.KaldiRecognizer(model, wf.getframerate())
            
            # Process the entire audio file
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                recognizer.AcceptWaveform(data)

            # Get the final transcribed text
            result_json = recognizer.FinalResult()
            result_dict = json.loads(result_json)
            predicted_text = result_dict.get('text', '')

            # Save the model's prediction to a corresponding text file
            prediction_filename = os.path.splitext(filename)[0] + ".txt"
            with open(os.path.join(PREDICTIONS_FOLDER, prediction_filename), "w", encoding="utf-8") as f:
                f.write(predicted_text)
                
            print(f"  -> Predicted: '{predicted_text}'")
    except wave.Error as e:
        print(f"Error opening {filename}: {e}")
        continue

print("\nAll audio files processed. Predictions saved in 'test_set/predictions'.")
print("-" * 30)
print("\n--- Calculating Word Error Rate (WER) ---")

ground_truth_sentences = []
predicted_sentences = []

# Get a sorted list of transcript files to ensure they match
transcript_files = sorted([f for f in os.listdir(GROUND_TRUTH_FOLDER) if f.lower().endswith(".txt")])

for filename in transcript_files:
    # Read the perfect ground truth transcript
    with open(os.path.join(GROUND_TRUTH_FOLDER, filename), "r", encoding="utf-8") as f:
        ground_truth_sentences.append(f.read().strip())
        
    # Read the model's predicted transcript
    prediction_filepath = os.path.join(PREDICTIONS_FOLDER, filename)
    if os.path.exists(prediction_filepath):
        with open(prediction_filepath, "r", encoding="utf-8") as f:
            predicted_sentences.append(f.read().strip())
    else:
        # If a prediction file is missing for some reason, add an empty string
        predicted_sentences.append("")

# Use jiwer to compare the two lists of sentences
# This single line calculates the overall error across all your files
error_rate = jiwer.wer(ground_truth_sentences, predicted_sentences)

print("\n--- EVALUATION COMPLETE ---")
print(f"Ground Truth Sentences:\n{ground_truth_sentences}")
print(f"\nPredicted Sentences:\n{predicted_sentences}")
print("-" * 30)
print(f"Overall Word Error Rate (WER): {error_rate * 100:.2f}%")
print("(A lower WER is better. 0% is perfect.)")