import os
import json
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# Placeholder for API Key - User provided key

class CommandProcessor:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o", temperature=0.5, openai_api_key="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"
        )

        # Define the prompt for the LLM to extract poker game commands
        # This prompt needs to be carefully engineered to guide the LLM
        # to output specific, predefined commands.
        # Note: {{ and }} are used to escape curly braces for the PromptTemplate.
        prompt_content = """
            당신(이름: 로키)은 사용자의 문장에서 포커 게임과 관련된 명령을 추출해야 합니다.

            <목표>
            - 문장에서 다음 리스트에 포함된 명령을 최대한 정확히 추출하세요.

            <명령 리스트>
            - GAME_START: 게임 시작, 게임을 시작해줘, 게임한판할까
            - CARD_CHECK: 카드 체크, 카드 보여줘, 카드 감지, 서로 카드확인
            - PLAY_START: 게임 플레이, 플레이 시작, 베팅시작
            - CALL: 콜, 콜할게
            - RAISE: 레이즈, 돈 걸어
            - FOLD: 폴드, 다이, 나죽을게
            - SET_RESULT: 게임 셋, 결과 설정

            <출력 형식>
            - 다음 JSON 형식을 반드시 따르세요:
            ```json
            {{
              "intent": "명령어",
              "amount": 숫자,
              "param": "파라미터"
            }}
            ```
            - 명령어는 <명령 리스트>에 있는 값 중 하나여야 합니다.
            - amount는 숫자로만 표현하며, 해당하지 않으면 0으로 설정합니다.
            - param은 문자열로 표현하며, 해당하지 않으면 빈 문자열("")로 설정합니다.
            - "SET_RESULT" 명령어의 경우 amount는 1부터 10까지, param은 "ROBOT_WIN", "PLAYER_WIN", "DRAW" 중 하나여야 합니다. 

            <예시>
            - 입력: "게임 시작해줘", "게임 스타트", "게임을 시작하겠습니다", "한 판 할까"
            출력: ```json
            {{
              "intent": "GAME_START",
              "amount": 0,
              "param": ""
            }}
            ```

            - 입력: "카드 보여줘", "카드를 확인하십시오", "카드 확인해", "카드를 확인할 순서입니다", "서로의 카드를 확인하세요", "카드체크"
            출력: ```json
            {{
              "intent": "CARD_CHECK",
              "amount": 0,
              "param": ""
            }}
            ```

            - 입력: "베팅을시작하십시오", "베팅시작", "베팅하세요" 
            출력: ```json
            {{
              "intent": "PLAY_START",
              "amount": 0,
              "param": ""
            }}
            ```
            - 입력: "콜", "콜할게", "오케이 콜", "알겠어 콜", "나도 콜"
            출력: ```json
            {{
              "intent": "CALL",
              "amount": 0,
              "param": ""
            }}
            ```

            - 입력: "나는 폴드할게", "상대방이 다이했습니다", "다이", "폴드", "죽을게", "난 죽을게"
            출력: ```json
            {{
              "intent": "FOLD",
              "amount": 0,
              "param": ""
            }}
            ```

            - 입력: "레이즈 1","하나걸게", "레이즈 하나", "한 장 건다"
            출력: ```json
            {{
              "intent": "RAISE",
              "amount": 1
              "param": ""
            }}
            ```

            - 입력: "레이즈 2", "따블", "두개걸게", "레이즈투", "두장건다"
            출력: ```json
            {{
              "intent": "RAISE",
              "amount": 2
              "param": ""
            }}
            ```            
            - 입력: "레이즈 3" "세개걸게", "레이즈쓰리", "세장건다", "세개건다"
            출력: ```json
            {{
              "intent": "RAISE",
              "amount": 3
              "param": ""
            }}
            ```          
            레이즈 4 5 이런 건 없음 1,2,3 만 있음 애매하면 인식했다고 하지마
            ```             
             
            - 입력: "게임 셋 로봇 윈, 로봇의 카드는 10이었습니다." "니가 이겼어, 니카드는 10이었어", "로봇 승리, 로봇의 카드는 10"
            출력: ```json
            {{
              "intent": "SET_RESULT",
              "amount": 10,
              "param": "ROBOT_WIN"
            }}
            ```

            - 입력: "게임 셋 플레이어 윈, 로봇의 카드는 1이었습니다.", "니가 졌어, 니카드는 1이었어", "인간 승리, 로봇의 카드는 1"
            출력: ```json
            {{
              "intent": "SET_RESULT",
              "amount": 1,
              "param": "PLAYER_WIN"
            }}
            ```      

            - 입력: "게임 셋 드로우, 로봇의 카드는 5였습니다", "둘이 비겼어, 니카드는 5였어"," 비김, 로봇의 카드는 5"
            출력: ```json
            {{
              "intent": "SET_RESULT",
              "amount": 5,
              "param": "DRAW"
            }}
            그러니까 SET_RESULT는 승패를 알려주고 카드숫자를 말함

            참고로 카드는 1부터 10까지 열 장 있음. A가 1이다.
            나머지는 2,3,4,5,6,7,8,9,10 이렇게 있음. 다른 숫자는 없으니까 애매하면 보내지마.
            "니 카드는 17이야" 이런 거 나오면 걍 아무것도 보내지마.
            ```             
            <사용자 입력>
            "{user_input}"                
        """

        self.prompt_template = PromptTemplate(
            input_variables=["user_input"], template=prompt_content
        )
        self.lang_chain = self.prompt_template | self.llm

    def process_command(self, user_input: str) -> tuple[str, int, str]:
        response = self.lang_chain.invoke({"user_input": user_input})
        
        # Extract JSON string from the response content
        json_str = response.content.strip()
        if json_str.startswith("```json") and json_str.endswith("```"):
            json_str = json_str[len("```json"): -len("```")].strip()

        try:
            parsed_data = json.loads(json_str)
            intent = parsed_data.get("intent", "")
            amount = parsed_data.get("amount", 0)
            param = parsed_data.get("param", "")
            return intent, amount, param
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Received content: {json_str}")
            return "", 0, "" # Return empty values on po