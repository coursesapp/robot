import subprocess
import threading
import queue
import logging
import os
import shutil
from typing import Optional, Callable

logger = logging.getLogger("TTS")

class TTSEngine:
    def __init__(self, model_path: str, config_path: Optional[str] = None, 
                 on_start: Optional[Callable[[], None]] = None,
                 on_end: Optional[Callable[[], None]] = None):
        self.queue = queue.Queue()
        self.running = False
        self.model_path = model_path
        self.config_path = config_path
        self.on_start = on_start
        self.on_end = on_end
        self.piper_exe = shutil.which("piper")
        
        if not self.piper_exe:
            if os.path.exists("piper/piper.exe"):
                self.piper_exe = "piper/piper.exe"
            else:
                logger.warning("Piper TTS binary not found in PATH using fallback (pyttsx3)")
                self.piper_exe = None
                self.has_pyttsx3 = True
        else:
            self.has_pyttsx3 = False
            
        self.engine = None
        self.current_process = None
        self.lock = threading.Lock()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._process_queue, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        self.interrupt()
        if self.thread.is_alive():
            self.thread.join()

    def interrupt(self):
        """Clears the queue and kills current speech."""
        logger.info("TTS: Interrupting...")
        # Clear queue
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except:
                break
        
        # Kill current process
        with self.lock:
            if self.current_process:
                try:
                    self.current_process.terminate()
                    self.current_process.kill()
                except:
                    pass
                self.current_process = None

    def speak(self, text: str):
        if not text:
            return
        self.queue.put(text)

    def _process_queue(self):
        while self.running:
            try:
                text = self.queue.get(timeout=1.0)
                logger.debug(f"Saying: {text}")
                
                if self.on_start:
                    self.on_start()

                if self.piper_exe:
                    self._speak_piper(text)
                elif hasattr(self, 'has_pyttsx3') and self.has_pyttsx3:
                    self._speak_pyttsx3(text)
                else:
                    logger.info(f"[SILENT TTS]: {text}")
                    
                if self.on_end:
                    self.on_end()
                    
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"TTS Error: {e}")

    def _speak_piper(self, text: str):
        try:
            output_file = "temp_speech.wav"
            cmd = [self.piper_exe, "--model", self.model_path, "--output_file", output_file]
            if self.config_path:
                cmd.extend(["--config", self.config_path])
                
            process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            with self.lock:
                self.current_process = process
            
            process.communicate(input=text.encode('utf-8'))
            
            with self.lock:
                self.current_process = None
                
            self._play_audio(output_file)
            
        except Exception as e:
            logger.error(f"Piper execution failed: {e}")

    def _play_audio(self, filename: str):
        try:
            cmd = ["powershell", "-c", f"(New-Object Media.SoundPlayer '{filename}').PlaySync()"]
            process = subprocess.Popen(cmd, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            with self.lock:
                self.current_process = process
            
            process.wait()
            
            with self.lock:
                self.current_process = None
            
            if os.path.exists(filename):
                os.remove(filename)
        except Exception as e:
            logger.error(f"Audio playback failed: {e}")

    def _speak_pyttsx3(self, text: str):
        try:
            import sys
            # Re-running pyttsx3 in a fresh child process circumvents the Windows COM lockup
            # Pass text via stdin to avoid command line escaping issues
            code = "import pyttsx3, sys; engine = pyttsx3.init(); engine.say(sys.stdin.read()); engine.runAndWait()"
            process = subprocess.Popen([sys.executable, "-c", code], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            with self.lock:
                self.current_process = process
                
            process.communicate(input=text.encode('utf-8'))
            
            with self.lock:
                self.current_process = None
        except Exception as e:
            logger.error(f"pyttsx3 failed: {e}")
