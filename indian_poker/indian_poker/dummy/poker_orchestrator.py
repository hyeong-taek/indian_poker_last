#!/usr/bin/env python3
import threading, asyncio
from enum import Enum, auto
from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup

from std_msgs.msg import String
from poker_interfaces.msg import HumanIntent  # ✅ 음성노드 최종 의도만 소비

from poker_interfaces.action import (
    MoveNamedPose,
    PickCard,
    ShowCard,
    BetChips,
    DetectCardOnce,
    ObserveGestures
)

try:
    import torch
    TORCH_OK = True
except Exception:
    TORCH_OK = False


class S(Enum):
    HAND_START = auto()
    PICK_AND_SHOW = auto()
    OPP_CARD_DETECT = auto()
    OBSERVE_GESTURE = auto()
    AGENT_DECIDE_OPEN = auto()
    WAIT_HUMAN = auto()
    AGENT_RESPOND = auto()
    RESOLVE = auto()


@dataclass
class StackState:
    stack_agent: int = 10
    stack_opp: int = 10
    pot: int = 0
    to_call: int = 0
    invest_agent: int = 0
    invest_opp: int = 0


class PokerOrchestrator(Node):
    def __init__(self):
        super().__init__('poker_orchestrator')
        self.cb_group = ReentrantCallbackGroup()

        # ===== Action Clients =====
        self.ac_move   = ActionClient(self, MoveNamedPose,  'robot/move_named_pose',  callback_group=self.cb_group)
        self.ac_pick   = ActionClient(self, PickCard,       'robot/pick_card',       callback_group=self.cb_group)
        self.ac_show   = ActionClient(self, ShowCard,       'robot/show_card',       callback_group=self.cb_group)
        self.ac_bet    = ActionClient(self, BetChips,       'robot/bet_chips',       callback_group=self.cb_group)
        self.ac_detect = ActionClient(self, DetectCardOnce, 'camera/detect_card_once', callback_group=self.cb_group)
        self.ac_tells  = ActionClient(self, ObserveGestures,'camera/observe_gestures', callback_group=self.cb_group)

        # ===== Topics =====
        self.sub_intent = self.create_subscription(HumanIntent, '/voice/intent', self.on_human_intent, 10)
        self.pub_agent  = self.create_publisher(String, '/poker/agent_bet', 10)  # 디버그/표시용

        # ===== Intent Queue & State =====
        self._intent_q: asyncio.Queue = asyncio.Queue()
        self._forced_result = None  # "ROBOT_WIN"|"PLAYER_WIN"|"DRAW" or None

        self.state = S.HAND_START
        self.hand_id = 0
        self.first_actor_is_agent = True

        self.S0 = 10
        self.st = StackState(self.S0, self.S0, 0, 0, 0, 0)

        self.opp_card_rank = 0
        self.tells = (False, False, False)
        self.opp_raises = 0
        self.last_opp_action = "NONE"
        self.last_opp_raise_size = 0
        self.conf_raise = 0.0
        self.carry_pot = 0

        self.policy = None
        if TORCH_OK:
            try:
                self.policy = torch.jit.load('poker_policy.pt')
                self.policy.eval()
                self.get_logger().info('Policy loaded: poker_policy.pt')
            except Exception as e:
                self.get_logger().warn(f'Policy load failed, fallback to random: {e}')

        

    # ===================== Sub: HumanIntent =====================
    def on_human_intent(self, msg: HumanIntent):
        intent = (msg.intent or "").upper()
        amount = int(msg.amount)
        param  = msg.param or ""

        # SET_RESULT는 즉시 강제 결과로 기록
        if intent == "SET_RESULT":
            if param in ("ROBOT_WIN", "PLAYER_WIN", "DRAW"):
                self._forced_result = param
                self.get_logger().warn(f'[Admin] Forced result = {param}')
            else:
                self.get_logger().warn(f'[Admin] Invalid SET_RESULT param="{param}"')
            # 큐에도 넣어두면 대기 중인 곳에서 바로 소비 가능
            try:
                self._intent_q.put_nowait((intent, amount, param))
            except asyncio.QueueFull:
                pass
            return

        # 나머지 의도는 큐로 전달 (상태별로 소비)
        try:
            self._intent_q.put_nowait((intent, amount, param))
        except asyncio.QueueFull:
            self.get_logger().warn('Intent queue full; dropping intent')

    # ===================== Main Run Loop =======================
    async def run(self):
        '''
        모든 액션 서버 연결 완료까지 동시 대기
        hand_id : 몇번째 판인지

        '''
        self.loop = asyncio.get_running_loop()   # on_human_intent에서 thread-safe 콜에 사용
        await self._wait_all_servers()  # 최대 10초동안 서버 연결되는지 대기
        self.get_logger().info('All action servers are ready.')

        while rclpy.ok():
            self.hand_id += 1
            await self._hand_once()

    async def _hand_once(self):
        self._reset_hand_state()
        self.state = S.HAND_START
        self.get_logger().info(f'=== Hand {self.hand_id} === (say: GAME_START)')

        # 1) GAME_START 대기
        await self._wait_intent(allowed={'GAME_START'})
        self._ante()
        await self._call_action(self.ac_move, MoveNamedPose.Goal(name="home", velocity=30.0, acc=30.0))

        # 2) CARD_CHECK 대기 → Pick & Show & Detect & Tells
        self.get_logger().info('Waiting: CARD_CHECK')
        await self._wait_intent(allowed={'CARD_CHECK'})
        await self._do_card_check_sequence()

        # 3) PLAY_START 대기 → 베팅 단계 진입
        self.get_logger().info('Waiting: PLAY_START')
        await self._wait_intent(allowed={'PLAY_START'})

        # 4) 선공 처리
        self.state = S.AGENT_DECIDE_OPEN
        if self.first_actor_is_agent:
            act = self._policy_decide(to_call=0)  # 선공: to_call=0
            if not act.startswith('RAISE_'):
                self.get_logger().warn(f'Open act {act} invalid; fallback RAISE_1')
                act = 'RAISE_1'
            await self._agent_bet(act)
            self.state = S.WAIT_HUMAN
        else:
            self.state = S.WAIT_HUMAN

        # 5) 베팅 루프 (CALL/RAISE/FOLD만 소비, SET_RESULT 언제든 가능)
        winner = await self._betting_loop()

        # 6) RESOLVE
        self.state = S.RESOLVE
        self._resolve_and_log(winner)

        # 7) 다음 핸드 선공 갱신
        if winner == 'agent':
            self.first_actor_is_agent = True
        elif winner == 'opp':
            self.first_actor_is_agent = False

    # ===================== Sequences ===========================
    async def _do_card_check_sequence(self):
        # PICK & SHOW
        self.state = S.PICK_AND_SHOW
        await self._call_action(self.ac_pick, PickCard.Goal(approach_z=20.0, grip_width=600.0))

        # SHOW + DETECT 병렬
        self.state = S.OPP_CARD_DETECT
        show_gh   = await self.ac_show.send_goal_async(ShowCard.Goal(timeout_s=3.0))
        detect_gh = await self.ac_detect.send_goal_async(DetectCardOnce.Goal(timeout_s=2.5))

        detect_res = await detect_gh.get_result_async()
        if detect_res.status == 4:
            self.opp_card_rank = int(detect_res.result.rank)
            conf = float(detect_res.result.conf)
            self.get_logger().info(f'[Detect] Opp card = {self.opp_card_rank} (conf={conf:.2f})')
        else:
            self.get_logger().warn('Detect failed or timeout; opp_card=0')

        show_res = await show_gh.get_result_async()
        if show_res.status != 4:
            self.get_logger().warn('ShowCard not succeeded')

        # GESTURE
        self.state = S.OBSERVE_GESTURE
        await self._call_action(self.ac_move, MoveNamedPose.Goal(name="gesture_view", velocity=30.0, acc=30.0))
        tells_res = await self._call_action(self.ac_tells, ObserveGestures.Goal(duration_s=10.0))
        if tells_res and tells_res.result:
            r = tells_res.result
            nose = bool(getattr(r, 'nose', False))
            arms = bool(getattr(r, 'arms', False))
            ear  = bool(getattr(r, 'ear', False))
            self.tells = (nose, arms, ear)
            self.get_logger().info(f'[Tells] nose={nose} arms={arms} ear={ear}')
        await self._call_action(self.ac_move, MoveNamedPose.Goal(name="home", velocity=30.0, acc=30.0))

    # ===================== Betting Loop ========================
    async def _betting_loop(self):
        """
        반환: 'agent' | 'opp' | 'draw' | 'showdown'
        """
        while True:
            # SET_RESULT가 들어왔으면 즉시 종료
            if self._forced_result:
                return self._forced_to_winner(self._forced_result)  # agent/ opp/ draw 리턴

            # 사람 베팅 의도 대기
            self.state = S.WAIT_HUMAN
            self.get_logger().info('[WaitHuman] say: CALL | RAISE(1/2/3) | FOLD')
            intent, amount, param = await self._wait_intent(allowed={'CALL','RAISE','FOLD','SET_RESULT'})
            if intent == 'SET_RESULT':
                return self._forced_to_winner(param)

            # 사람 행동 회계 반영
            pre_need = self._need_for_opp()
            act = f'RAISE_{amount}' if intent == 'RAISE' else intent
            self._apply_human_action(act)

            if act == 'FOLD':
                return 'agent'  # 사람 폴드 → 로봇 승

            if act == 'CALL':
                if pre_need == 0:
                    # 원래부터 콜 차액이 없었음 → CHECK
                    self.get_logger().info('[Human] CHECK')
                    # 로봇 응수로 계속
                else:
                    # 베팅이 걸려 있던 상황에서 CALL을 시도함
                    if self._need_for_opp() > 0:
                        # 콜을 다 못 채운 쇼트 올인
                        self.get_logger().info('[Human] CALL (short all-in)')
                        return 'showdown'
                    else:
                        # 콜을 완전히 채움 → 라운드 종료(쇼다운)
                        self.get_logger().info('[Human] CALL')
                        return 'showdown'

            # 로봇 응수(사람 레이즈 이후)
            self.state = S.AGENT_RESPOND
            my_act = self._policy_decide(to_call=self.st.to_call)
            self.get_logger().info(f'[Agent] {my_act}')

            if my_act == 'FOLD':
                return 'opp'

            elif my_act == 'CALL':
                need = self._need_for_agent()
                pay  = min(need, self.st.stack_agent)
                self.st.invest_agent += pay
                self.st.stack_agent  -= pay
                self.st.pot          += pay
                self.st.to_call = max(0, self.st.invest_opp - self.st.invest_agent)
                self.pub_agent.publish(String(data="CALL" if need > 0 else "CHECK"))
                return 'showdown'

            elif my_act.startswith('RAISE_'):
                await self._agent_bet(my_act)
                # 루프 계속(사람 응답 대기)
            else:
                self.get_logger().warn('Invalid agent act; folding defensively')
                return 'opp'

    # ===================== Helpers =============================
    async def _wait_intent(self, allowed: set, timeout: float = None):
        """의도 큐에서 allowed에 해당하는 항목이 나올 때까지 대기"""
        while True:
            # 강제 결과가 있으면 즉시 반환
            if 'SET_RESULT' in allowed and self._forced_result:
                return ('SET_RESULT', 0, self._forced_result)
            try:
                item = await asyncio.wait_for(self._intent_q.get(), timeout=timeout)
            except asyncio.TimeoutError:
                self.get_logger().warn('Wait intent timeout')
                continue
            intent, amount, param = item
            if intent in allowed:
                return item
            else:
                self.get_logger().info(f'Ignored intent "{intent}" in state {self.state.name}')

    # 누가 이겼는지 리턴
    def _forced_to_winner(self, param: str) -> str:
        if param == 'ROBOT_WIN':
            return 'agent'
        if param == 'PLAYER_WIN':
            return 'opp'
        return 'draw'

    # 액션 콜 
    async def _call_action(self, client: ActionClient, goal, feedback_cb=None, timeout_s: float = 10.0):
        ok = await asyncio.to_thread(client.wait_for_server, timeout_sec=timeout_s)
        if not ok:
            raise RuntimeError("Action server not ready for goal")
        gh = await client.send_goal_async(goal, feedback_callback=feedback_cb)
        return await gh.get_result_async()
    
    # 액션 서버 연결
    async def _wait_all_servers(self, per_server_timeout: float = 10.0):
        async def wait(client, name: str):
            ok = await asyncio.to_thread(client.wait_for_server, timeout_sec=per_server_timeout)
            if not ok:
                raise RuntimeError(f"Action server not ready: {name}")
            return name

        await asyncio.gather(
            wait(self.ac_move,   "move"),
            wait(self.ac_pick,   "pick"),
            wait(self.ac_show,   "show"),
            wait(self.ac_bet,    "bet"),
            wait(self.ac_detect, "detect"),
            wait(self.ac_tells,  "tells"),
        )


    # ===================== Accounting ==========================
    def _reset_hand_state(self):
        self._forced_result = None
        self.st.pot = self.carry_pot
        self.carry_pot = 0 
        self.st.to_call = 0
        self.st.invest_agent = 0 # 로봇이 팟에 넣은 금액
        self.st.invest_opp = 0   # 사람이 팟에 넣은 금액
        self.opp_card_rank = 0
        self.tells = (False, False, False)
        self.opp_raises = 0
        self.last_opp_action = "NONE"
        self.last_opp_raise_size = 0
        self.conf_raise = 0.0

    # 판 시작시 보유 코인 -1 -> pot + 2
    def _ante(self):
        put = 1
        self.st.stack_agent -= put
        self.st.stack_opp   -= put
        self.st.pot         += 2

    # 사람 넣은 코인 - 로봇 넣은 코인
    def _need_for_agent(self) -> int:
        return max(0, self.st.invest_opp - self.st.invest_agent)

    # 로봇 넣은코인 - 사람 넣은 코인
    def _need_for_opp(self) -> int:
        return max(0, self.st.invest_agent - self.st.invest_opp)

    # 로봇 배팅
    async def _agent_bet(self, act: str):
        """로봇 레이즈: (콜 차액 + k). 스택 부족 시 올인."""
        self.pub_agent.publish(String(data=act))
        k = int(act.split('_')[1])
        need  = self._need_for_agent()
        total = need + k                         # 상대가 낸 코인 + 더 얻을 코인
        pay   = min(total, self.st.stack_agent)  # 내가 가지고 잇는 것보다 필요 코인이 많은 경우 올인

        self.st.invest_agent += pay              # 이번 팟에서 내가 낸 코인에 합계
        self.st.stack_agent  -= pay              # 내 보유 코인에서 감소
        self.st.pot          += pay              # 게임 팟에 낸 만큼 코인 추가
        self.st.to_call = max(0, self.st.invest_agent - self.st.invest_opp)  # 상대방이 콜시 내야할 코인 (사람이 내야할)

        await self._call_action(self.ac_bet, BetChips.Goal(amount=pay))

    # 사람의 행동 회계 계산
    def _apply_human_action(self, act: str):
        """사람 액션 회계: CALL=차액, RAISE_k=차액+k"""
        self.last_opp_action = act
        
        # RAISE
        if act.startswith('RAISE_'):
            k = int(act.split('_')[1])
            if k not in (1, 2, 3):
                self.get_logger().warn(f'Invalid raise size: {k}; clamping to 1')
                k = 1
            need  = self._need_for_opp()
            total = need + k
            pay   = min(total, self.st.stack_opp)

            self.st.invest_opp += pay
            self.st.stack_opp  -= pay
            self.st.pot        += pay
            self.st.to_call = max(0, self.st.invest_opp - self.st.invest_agent)

            # 사람 마지막 레이즈 저장 및 레이즈에 따른 자신감 결정
            self.opp_raises += 1
            self.last_opp_raise_size = k
            base_map = {1:0.3, 2:0.6, 3:1.0}
            base = base_map[k]
            self.conf_raise = min(1.0, base + 0.1 * max(0, self.opp_raises - 1))

        # CALL
        elif act == 'CALL':
            need = self._need_for_opp()
            pay  = min(need, self.st.stack_opp)

            self.st.invest_opp += pay
            self.st.stack_opp  -= pay
            self.st.pot        += pay
            self.st.to_call = max(0, self.st.invest_opp - self.st.invest_agent)

        # FOLD
        elif act == 'FOLD':
            pass

    # ===================== Policy ==============================
    # 모델에 넣어줄 입력 벡터 리턴
    def _build_obs(self, to_call_for_me: int):
        opp_card_norm = self.opp_card_rank / 10.0
        my_stack   = self.st.stack_agent / self.S0
        opp_stack  = self.st.stack_opp / self.S0
        pot_n      = self.st.pot / self.S0
        to_call_n  = to_call_for_me / self.S0
        p_win_belief = (10 - self.opp_card_rank) / 10.0 if self.opp_card_rank else 0.5

        nose_b, arms_b, ear_b = self.tells
        tells = (1.0 if nose_b else 0.0,
                 1.0 if arms_b else 0.0,
                 1.0 if ear_b else 0.0)
        is_first = 1.0 if self.first_actor_is_agent else 0.0

        a = self.last_opp_action
        onehot = [0.0]*5
        mapidx = {'FOLD':0,'CALL':1,'RAISE_1':2,'RAISE_2':3,'RAISE_3':4}
        if a in mapidx: onehot[mapidx[a]] = 1.0

        last_size = self.last_opp_raise_size / 3.0
        opp_raises = min(self.opp_raises, 3) / 3.0
        conf = self.conf_raise
        stage_flag = 1.0 if self.st.to_call > 0 else 0.0

        return [
            opp_card_norm, my_stack, opp_stack, pot_n, to_call_n,
            p_win_belief, tells[0], tells[1], tells[2], is_first,
            *onehot, last_size, opp_raises, conf, stage_flag
        ]
    # 모델 출력
    def _policy_decide(self, to_call: int) -> str:
        obs = self._build_obs(to_call)
        if TORCH_OK and self.policy is not None:
            with torch.no_grad():
                x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                logits = self.policy(x)  # [B,5]
                idx = int(torch.argmax(logits, dim=-1).item())
        else:
            import random
            if to_call <= 0:
                idx = random.choice([2,3,4])  # RAISE_1..3
            else:
                idx = random.choice([0,1,2,3,4])
        action_map = ["FOLD","CALL","RAISE_1","RAISE_2","RAISE_3"]
        act = action_map[idx]
        if to_call <= 0 and act == "CALL":
            act = "RAISE_1"
        return act

    # ===================== Resolve =============================
    def _resolve_and_log(self, winner: str):
        self.get_logger().info(f'[Resolve] winner={winner} pot={self.st.pot} '
                               f'(Agent:{self.st.invest_agent}, Opp:{self.st.invest_opp})')
        if winner == 'agent':
            self.st.stack_agent += self.st.pot
        elif winner == 'opp':
            self.st.stack_opp += self.st.pot
        elif winner in ('draw', 'showdown'):
            self.carry_pot += self.st.pot    # 👈 팟 전액을 다음 핸드로 이월
            self.get_logger().info(f'Carry-over pot = {self.carry_pot}')
        self.get_logger().info(f'End Hand {self.hand_id}: stack A={self.st.stack_agent}, '
                                f'O={self.st.stack_opp}')


def main():
    rclpy.init()
    node = PokerOrchestrator()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)

    t = threading.Thread(target=ex.spin, daemon=True)  # Executor 백그라운드 스레드
    t.start()
    try:
        asyncio.run(node.run())                        # 메인 스레드에서 asyncio
    finally:
        rclpy.shutdown()
        t.join(timeout=1.0)


if __name__ == '__main__':
    main()
