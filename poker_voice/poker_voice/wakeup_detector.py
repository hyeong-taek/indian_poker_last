import numpy as np
import openwakeword
from openwakeword.model import Model
from scipy.signal import resample
import os

# Path to the wakeup word model
# In a production system, this model should be installed to the package's share directory
# and loaded using ament_index_python. For now, hardcoding the path for simplicity.
MODEL_PATH = "/home/hyeonhee/ros2_ws/src/DoosanBootcamp3rd/Tutorial/VoiceProcessing/hello_rokey_8332_32.tflite"


class WakeupWord:
    def __init__(self, buffer_size, model_path=MODEL_PATH):
        openwakeword.utils.download_models()
        self.model = Model(wakeword_models=[model_path])
        self.model_name = os.path.basename(model_path).split(".", maxsplit=1)[0]
        self.stream = None
        self.buffer_size = buffer_size

    def is_wakeup(self):
        audio_chunk = np.frombuffer(
            self.stream.read(self.buffer_size, exception_on_overflow=False),
            dtype=np.int16,
        )
        audio_chunk = resample(audio_chunk, int(len(audio_chunk) * 16000 / 48000))
        outputs = self.model.predict(audio_chunk, threshold=0.1)
        confidence = outputs[self.model_name]
        # print("confidence: ", confidence) # For debugging confidence levels
        if confidence > 0.3:
            return True
        return False

    def set_stream(self, stream):
        self.stream = stream

    def reset(self):
        self.model.reset()
