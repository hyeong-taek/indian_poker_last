from openai import OpenAI
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import tempfile
import os

# Placeholder for API Key - REPLACE WITH YOUR ACTUAL KEY OR LOAD SECURELY
OPENAI_API_KEY = "" 

class STTService:
    def __init__(self, api_key, node):
        self.client = OpenAI(api_key=api_key)
        self.duration = 5  # seconds
        self.samplerate = 16000  # Whisper prefers 16kHz
        self.node = node # Store node instance for logging

    def speech2text(self):
        self.node.get_logger().info("음성 녹음을 시작합니다. 5초 동안 말해주세요...")
        audio = sd.rec(
            int(self.duration * self.samplerate),
            samplerate=self.samplerate,
            channels=1,
            dtype="int16",
        )
        sd.wait()
        self.node.get_logger().info("녹음 완료. Whisper에 전송 중...")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            wav.write(temp_wav.name, self.samplerate, audio)

            with open(temp_wav.name, "rb") as f:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1", file=f
                )
        os.remove(temp_wav.name) # Clean up the temporary file

        self.node.get_logger().info(f"STT 결과: {transcript.text}")
        return transcript.text
