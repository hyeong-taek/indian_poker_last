import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Header

import threading
import time
import numpy as np
import sounddevice as sd

from pydub import AudioSegment
from pydub.playback import play
from ament_index_python.packages import get_package_share_directory
import os

from poker_interfaces.msg import HumanIntent

from .mic_controller import MicController, MicConfig
from .wakeup_detector import WakeupWord, MODEL_PATH as WAKEUP_MODEL_PATH
from .stt_service import STTService, OPENAI_API_KEY
from .command_processor import CommandProcessor
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

voice_qos = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,  # 기본값. 의도치 않은 '지난 명령' 재전달 방지
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)
class VoiceRecognizer(Node):
    def __init__(self):
        super().__init__('voice_recognizer')
        # Change publisher to HumanIntent on /human_intent
        self.publisher_ = self.create_publisher(HumanIntent, '/voice/intent', voice_qos)
        self._first_publish_synced = False
        self.mic_controller = MicController()
        self.mic_controller.open_stream()

        self.wakeup_word_detector = WakeupWord(self.mic_controller.config.buffer_size, WAKEUP_MODEL_PATH)
        self.wakeup_word_detector.set_stream(self.mic_controller.stream)

        self.stt_service = STTService(OPENAI_API_KEY, self) # Pass node instance to STTService
        self.command_processor = CommandProcessor()

        # Load audio files
        package_share_directory = get_package_share_directory('poker_voice')
        self.ttokkang_audio_path = os.path.join(package_share_directory, 'audio', '또깡.mp3')
        self.arasseum_audio_path = os.path.join(package_share_directory, 'audio', '알았음.mp3')
        self.error_audio_path = os.path.join(package_share_directory, 'audio', '에러.mp3')

        try:
            self.ttokkang_audio = AudioSegment.from_mp3(self.ttokkang_audio_path)
            self.arasseum_audio = AudioSegment.from_mp3(self.arasseum_audio_path)
            self.error_audio = AudioSegment.from_mp3(self.error_audio_path)
            self.get_logger().info('Audio files loaded successfully.')
        except Exception as e:
            self.get_logger().error(f'Failed to load audio files: {e}')
            self.ttokkang_audio = None
            self.arasseum_audio = None
            self.error_audio = None

        self.get_logger().info('Voice Recognizer Node has been started. Listening for wakeup word...')

        # Start wakeup word listening in a separate thread
        self._wakeup_thread = threading.Thread(target=self._listen_for_wakeup_word)
        self._wakeup_thread.daemon = True # Allow the main program to exit even if thread is running
        self._wakeup_thread.start()


    def _wait_subscribers_once(self, timeout=2.0):
        if self._first_publish_synced:
            return
        t0 = time.time()
        while rclpy.ok() and self.publisher_.get_subscription_count() == 0:
            if time.time() - t0 > timeout:
                break
            time.sleep(0.05)
        self._first_publish_synced = True



    def _listen_for_wakeup_word(self):
        while rclpy.ok():
            try:
                if self.wakeup_word_detector.is_wakeup():
                    self.get_logger().info("웨이크업 워드 감지! 음성 인식을 시작합니다.")
                    if self.ttokkang_audio:
                        play(self.ttokkang_audio)

                    while True: # Loop indefinitely until a command is understood
                        text = self.stt_service.speech2text()
                        if not text:
                            self.get_logger().info("음성 입력이 없습니다. 웨이크업 워드 감지 재개.")
                            if self.error_audio:
                                play(self.error_audio)
                            break # Exit the inner loop and go back to wakeup word detection

                        intent, amount, param = self.command_processor.process_command(text)
                        
                        # If intent is found, process it and break the loop
                        if intent:
                            self.get_logger().info(f'Predicted Intent: {intent}, Amount: {amount}, Param: {param}')
                            human_intent_msg = HumanIntent()
                            human_intent_msg.header = Header()
                            human_intent_msg.header.stamp = self.get_clock().now().to_msg()
                            human_intent_msg.intent = intent
                            human_intent_msg.amount = amount
                            human_intent_msg.param = param

                            self._wait_subscribers_once(timeout=2.0)
                            self.publisher_.publish(human_intent_msg)
                            self.get_logger().info(f'Publishing HumanIntent: {human_intent_msg}')
                            if self.arasseum_audio:
                                play(self.arasseum_audio)
                            break  # Exit the infinite loop on success
                        
                        # If intent is not found, prompt for retry and continue the loop
                        else:
                            self.get_logger().info("명령을 이해하지 못했습니다. 웨이크업 워드 감지 재개.")
                            if self.error_audio:
                                play(self.error_audio)
                            break # Exit the inner loop and go back to wakeup word detection

                    # After the loop (either success or final failure)
                    self.get_logger().info("음성 인식 완료. 웨이크업 워드 감지 재개.")
                    self.wakeup_word_detector.reset()
                    time.sleep(2)
            except Exception as e:
                self.get_logger().error(f'Error in wakeup word detection or STT: {e}')

    def destroy_node(self):
        self.mic_controller.close_stream()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    voice_recognizer = VoiceRecognizer()
    rclpy.spin(voice_recognizer)
    voice_recognizer.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()