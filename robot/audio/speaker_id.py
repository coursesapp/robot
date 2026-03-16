import os
import numpy as np
import onnxruntime as ort
import logging
from typing import Optional, List
from scipy.fft import dct
from scipy.signal import lfilter

logger = logging.getLogger("SpeakerID")

class SpeakerRecognizer:
    def __init__(self, model_path: str = "models/voxceleb_CAM++.onnx"):
        self.model_path = model_path
        self.session = None
        
        if os.path.exists(model_path):
            try:
                # Use CPU for lightweight inference
                self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
                logger.info(f"Speaker recognition model loaded from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load speaker model: {e}")
        else:
            logger.warning(f"Speaker model not found at {model_path}. Identification will be disabled.")

    def extract_embedding(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """
        Extracts a 256-dim speaker embedding from raw audio (16kHz, float32).
        """
        if self.session is None:
            return None
        
        try:
            # 1. Preprocessing (Fbank 80-dim)
            # This is a simplified version of Fbank extraction
            feats = self._extract_fbank(audio)
            
            # 2. Inference
            # input shape: [batch, length, feats] -> [1, T, 80]
            inputs = {self.session.get_inputs()[0].name: feats.astype(np.float32)}
            embedding = self.session.run(None, inputs)[0]
            
            # 3. Post-processing
            embedding = embedding.flatten()
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding /= norm
                
            return embedding
        except Exception as e:
            logger.error(f"Error extracting speaker embedding: {e}")
            return None

    def _extract_fbank(self, signal: np.ndarray, samplerate=16000, winlen=0.025, winstep=0.01, nfilt=80, nfft=512):
        """
        Simplified Mel Filterbank extraction using numpy/scipy.
        """
        # Pre-emphasis
        signal = np.append(signal[0], signal[1:] - 0.97 * signal[:-1])
        
        # Framing
        frame_len = int(winlen * samplerate)
        frame_step = int(winstep * samplerate)
        signal_len = len(signal)
        num_frames = int(np.ceil(float(np.abs(signal_len - frame_len)) / frame_step))
        
        pad_signal_len = num_frames * frame_step + frame_len
        z = np.zeros((pad_signal_len - signal_len))
        pad_signal = np.append(signal, z)
        
        indices = np.tile(np.arange(0, frame_len), (num_frames, 1)) + \
                  np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_len, 1)).T
        frames = pad_signal[indices.astype(np.int32, copy=False)]
        
        # Windowing (Hamming)
        frames *= np.hamming(frame_len)
        
        # FFT and Power Spectrum
        mag_frames = np.absolute(np.fft.rfft(frames, nfft))
        pow_frames = ((1.0 / nfft) * (mag_frames ** 2))
        
        # Filterbanks
        low_mel = 0
        high_mel = (2595 * np.log10(1 + (samplerate / 2) / 700))
        mel_points = np.linspace(low_mel, high_mel, nfilt + 2)
        hz_points = (700 * (10**(mel_points / 2595) - 1))
        bin = np.floor((nfft + 1) * hz_points / samplerate)
        
        fbank = np.zeros((nfilt, int(np.floor(nfft / 2 + 1))))
        for m in range(1, nfilt + 1):
            f_m_minus = int(bin[m - 1])
            f_m = int(bin[m])
            f_m_plus = int(bin[m + 1])
            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
        
        filter_banks = np.dot(pow_frames, fbank.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
        filter_banks = 10 * np.log10(filter_banks) # Log
        
        # Mean Normalization
        filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)
        
        return filter_banks[np.newaxis, ...] # Add batch dim [1, T, 80]
