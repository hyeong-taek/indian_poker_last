# 로봇 포커 게임 음성 제어 프로젝트

이 프로젝트는 사용자의 음성 명령을 인식하여 로봇 포커 게임의 동작을 제어하는 ROS 2 패키지(`poker_voice`)입니다.

## 주요 기능

- **웨이크업 워드 감지**: "헬로 로키" 음성을 감지하여 시스템을 활성화합니다.
- **음성 인식 (STT)**: OpenAI Whisper를 사용하여 음성 명령을 텍스트로 변환합니다.
- **명령어 처리**: 변환된 텍스트를 기반으로 ChatGPT (Langchain)를 활용하여 포커 게임 명령 (예: `GAME_START`, `RAISE`, `FOLD`)을 예측하고, `poker_interfaces` 패키지에 정의된 `HumanIntent` 메시지 형태로 발행합니다.
- **로봇 제어 연동**: 발행된 `HumanIntent` 메시지를 통해 로봇의 실제 동작을 제어합니다.
- **오디오 피드백**: 시스템 상태 및 명령 처리 결과에 대한 오디오 피드백을 제공합니다.

## 파일 구조 (핵심 부분)

```
.
├── poker_interfaces/
│   └── msg/
│       └── HumanIntent.msg
└── poker_voice/
    ├── poker_voice/
    │   ├── voice_recognizer_node.py
    │   ├── mic_controller.py
    │   ├── wakeup_detector.py
    │   ├── stt_service.py
    │   └── command_processor.py
    ├── resource/
    │   ├── audio/
    │   └── models/
    └── ...
```

### 파일 구조 설명

-   **`poker_interfaces/`**: 포커 게임 관련 ROS 2 인터페이스 (메시지, 액션)를 정의합니다.
    -   **`HumanIntent.msg`**: 사용자 음성 명령을 로봇이 이해할 수 있는 형태로 정의한 메시지 파일입니다.
-   **`poker_voice/`**: 음성 인식 및 명령 처리를 담당하는 ROS 2 패키지입니다.
    -   **`poker_voice/` (Python 소스 코드 디렉토리)**:
        -   **`voice_recognizer_node.py`**: 메인 ROS 2 노드로, 음성 처리 흐름을 제어하고 ROS 2 토픽을 발행합니다.
        -   **`mic_controller.py`**: 마이크 입력을 관리합니다.
        -   **`wakeup_detector.py`**: 웨이크업 워드("헬로 로키")를 감지합니다.
        -   **`stt_service.py`**: OpenAI Whisper와 연동하여 음성을 텍스트로 변환합니다.
        -   **`command_processor.py`**: ChatGPT/Langchain을 활용하여 텍스트 명령을 포커 게임 의도로 파싱합니다.
    -   **`resource/`**: 오디오 파일 및 AI 모델과 같은 리소스 파일을 포함합니다.
        -   **`audio/`**: 음성 피드백용 MP3 파일 (예: 또깡.mp3, 알았음.mp3)이 저장됩니다.
        -   **`models/`**: 웨이크업 워드 감지 모델 파일이 저장됩니다.
    -   **`...`**: 기타 패키지 설정 파일 (setup.py, package.xml 등)을 나타냅니다.

## 구성 요소

- **`poker_voice/voice_recognizer_node.py`**: 메인 ROS 2 노드로, 마이크 입력, 웨이크업 워드 감지, STT, 명령어 처리 및 ROS 2 토픽 발행을 총괄합니다.
- **`poker_voice/mic_controller.py`**: 마이크 입력을 관리합니다.
- **`poker_voice/wakeup_detector.py`**: "헬로 로키" 웨이크업 워드를 감지합니다.
- **`poker_voice/stt_service.py`**: OpenAI Whisper API를 사용하여 음성을 텍스트로 변환합니다.
- **`poker_voice/command_processor.py`**: Langchain과 ChatGPT를 활용하여 텍스트 명령을 `HumanIntent` 메시지로 파싱합니다.
- **`poker_interfaces`**: `poker_voice` 패키지에서 사용하는 `HumanIntent` 메시지 및 로봇 액션 정의를 포함하는 ROS 2 인터페이스 패키지입니다.

## 의존성

- `openai`
- `sounddevice`
- `scipy`
- `numpy`
- `openwakeword`
- `pyaudio`
- `pydub`
- `langchain`
- `langchain-openai`

## 사용법

1.  **환경 설정**: 필요한 Python 패키지를 설치합니다.
2.  **ROS 2 빌드**: `colcon build`를 사용하여 패키지를 빌드합니다.
3.  **노드 실행**: `ros2 run poker_voice voice_recognizer_node` 명령으로 노드를 실행합니다.
4.  **음성 명령**: "헬로 로키"라고 말하여 시스템을 활성화한 후, 포커 게임 관련 명령을 내립니다.