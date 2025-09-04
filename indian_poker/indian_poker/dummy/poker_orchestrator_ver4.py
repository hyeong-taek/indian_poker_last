#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Poker Orchestrator (FSM 버전)
- asyncio + rclpy 멀티스레드 실행
- HumanIntent 콜백은 thread-safe 하게 asyncio Queue에 투입
- 한 판(Hand) 진행은 정식 상태 머신(FSM)으로 구성
- 무승부(draw/showdown) 시 pot 이월(carry_pot)
- tells는 (bool,bool,bool)로 처리
- 의도 신호 맵핑:
  GAME_START  : 새 핸드 시작(앤티) + 카드 그립 위치 이동 + PickCard
  CARD_CHECK  : card_show 이동 → ShowCard & DetectCardOnce 동시 실행 → gesture_view 이동 → Observe
  PLAY_START  : 베팅 루프 진입
"""

import threading
import asyncio
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from action_msgs.msg import GoalStatus

from std_msgs.msg import String
from poker_interfaces.msg import HumanIntent  # 음성노드 최종 의도만 소비

from poker_interfaces.action import (
    MoveNamedPose,
    PickCard,
    ShowCard,
    BetChips,
    DetectCardOnce,
    ObserveGestures,
    DetectChip
)

try:
    import torch
    TORCH_OK = True
except Exception:
    TORCH_OK = False


class S(Enum):
    HAND_START = auto()
    PICK_AND_SHOW = auto()       # 현재 흐름에서는 사용하지 않지만, 호환을 위해 유지
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
        self.ac_move = ActionClient(self, MoveNamedPose, '/dsr01/robot/move_named_pose', callback_group=self.cb_group)
        self.ac_pick = ActionClient(self, PickCard, '/dsr01/robot/pick_card', callback_group=self.cb_group)
        self.ac_show = ActionClient(self, ShowCard, '/dsr01/robot/show_card', callback_group=self.cb_group)
        self.ac_bet = ActionClient(self, BetChips, '/dsr01/robot/bet_chips', callback_group=self.cb_group)
        self.ac_detect = ActionClient(self, DetectCardOnce, '/dsr01/camera/detect_card_once', callback_group=self.cb_group)
        self.ac_tells = ActionClient(self, ObserveGestures, '/dsr01/camera/observe_gestures', callback_group=self.cb_group)
        self.ac_chip_detect = ActionClient(self, DetectChip, '/dsr01/camera/detect_chip_once', callback_group=self.cb_group)
        
        # ===== Topics =====
        self.sub_intent = self.create_subscription(HumanIntent, '/voice/intent', self.on_human_intent, 10)
        self.pub_agent = self.create_publisher(String, '/poker/agent_bet', 10)  # 디버그/표시용

        # ===== Intent Queue & State =====
        self._intent_q: asyncio.Queue = asyncio.Queue()
        self._forced_result: Optional[str] = None  # "ROBOT_WIN"|"PLAYER_WIN"|"DRAW"|None

        # 이벤트 루프 준비 전 Intent 드랍 방지용 버퍼/락
        self.loop = None
        self._preloop_buf = deque()
        self._preloop_lock = threading.Lock()

        self.state = S.HAND_START
        self.hand_id = 0
        self.first_actor_is_agent = True

        self.S0 = 10
        self.st = StackState(self.S0, self.S0, 0, 0, 0, 0)

        self.opp_card_rank = 0
        self.tells = (False, False, False)  # (nose, arms, ear)
        self.opp_raises = 0
        self.last_opp_action = "NONE"
        self.last_opp_raise_size = 0
        self.conf_raise = 0.0
        self.carry_pot = 0  # 무승부 시 다음 핸드로 이월할 팟

        self.policy = None

        # ===== FSM 보조 필드 =====
        self._winner: Optional[str] = None  # 'agent'|'opp'|'draw'|'showdown'|None
        self._handlers = {
            S.HAND_START:           self._st_hand_start,
            #S.PICK_AND_SHOW:        self._st_pick_and_show,   # (현재 미사용)
            S.OPP_CARD_DETECT:      self._st_detect_and_show, # CARD_CHECK 후: show & detect
            S.OBSERVE_GESTURE:      self._st_observe_gesture, # show/detect 후: gesture_view 이동 → Observe
            S.AGENT_DECIDE_OPEN:    self._st_agent_open,
            S.WAIT_HUMAN:           self._st_wait_human,
            S.AGENT_RESPOND:        self._st_agent_respond,
            S.RESOLVE:              self._st_resolve,
        }

    # ===================== 공통 로그/유틸 ======================
    def _status_str(self, status: int) -> str:
        m = {
            GoalStatus.STATUS_UNKNOWN: "UNKNOWN",
            GoalStatus.STATUS_ACCEPTED: "ACCEPTED",
            GoalStatus.STATUS_EXECUTING: "EXECUTING",
            GoalStatus.STATUS_CANCELING: "CANCELING",
            GoalStatus.STATUS_SUCCEEDED: "SUCCEEDED",
            GoalStatus.STATUS_CANCELED: "CANCELED",
            GoalStatus.STATUS_ABORTED: "ABORTED",
        }
        return m.get(status, f"({status})")

    def _log_ok_msg(self, tag: str, ok: bool, status: int, res) -> None:
        # Result가 ok/message 필드를 가진 규격일 때
        r_ok  = getattr(res, "ok", None) if res is not None else None
        r_msg = getattr(res, "message", "") if res is not None else ""
        self.get_logger().info(
            f"[{tag}] ok={ok} status={self._status_str(status)} "
            f"res.ok={r_ok} res.message='{r_msg}'"
        )

    # ===================== Sub: HumanIntent =====================
    def on_human_intent(self, msg: HumanIntent):
        intent = (msg.intent or "").upper()
        amount = int(msg.amount)
        param = msg.param or ""

        # 관리 명령: SET_RESULT → 강제 승패
        if intent == "SET_RESULT":
            if param in ("ROBOT_WIN", "PLAYER_WIN", "DRAW"):
                self._forced_result = param
                self.get_logger().warn(f'[Admin] Forced result = {param}')
            else:
                self.get_logger().warn(f'[Admin] Invalid SET_RESULT param="{param}"')

        item = (intent, amount, param)

        # Executor 스레드 → asyncio 루프 스레드로 안전하게 투입 (루프 없으면 버퍼)
        if self.loop is not None:
            try:
                self.loop.call_soon_threadsafe(self._intent_q.put_nowait, item)
            except Exception as e:
                self.get_logger().error(f'Failed to enqueue intent via loop: {e}')
        else:
            try:
                with self._preloop_lock:
                    self._preloop_buf.append(item)
            except Exception as e:
                self.get_logger().error(f'Failed to buffer intent pre-loop: {e}')

    # ===================== Main Run Loop =======================
    async def run(self):
        """ROS 액션 서버 준비 대기 후, FSM으로 핸드를 반복 실행."""
        self.loop = asyncio.get_running_loop()

        # 루프 준비 전에 들어온 intent들을 안전하게 비움(드랍 방지)
        with self._preloop_lock:
            backlog = list(self._preloop_buf)
            self._preloop_buf.clear()
        for item in backlog:
            await self._intent_q.put(item)

        await self._wait_all_servers()
        self.get_logger().info('All action servers are ready.')

        while rclpy.ok():
            await self._run_hand_fsm()

    async def _run_hand_fsm(self):
        """한 판(Hand)을 FSM으로 수행"""
        self.hand_id += 1
        self._winner = None
        self._reset_hand_state()

        state = S.HAND_START
        while state is not None and rclpy.ok():
            self.state = state
            self.get_logger().info(f'[FSM] → {self.state.name}')
            handler = self._handlers[state]
            state = await handler()

    # ===================== FSM States ==========================
    async def _st_hand_start(self):
        """
        GAME_START 수신:
        - 앤티 처리
        - 카드 그립 위치 이동
        - 즉시 PickCard 수행
        - 이후 CARD_CHECK 대기 (→ OPP_CARD_DETECT)
        """
        self.get_logger().info(f'=== Hand {self.hand_id} === (say: GAME_START)')
        await self._wait_intent(allowed={'GAME_START'})
        self._ante()

        # 카드 집는 위치로 이동
        await self._exec_action(
            "Move(card)",
            self.ac_move,
            MoveNamedPose.Goal(name="card_grip", velocity=70.0, acc=50.0),
            retry=1
        )

        # ★ 여기서 바로 카드 집기까지 수행
        await self._exec_action(
            "PickCard",
            self.ac_pick,
            PickCard.Goal(approach_z=20.0, grip_width=600),
            retry=1
        )

        # 이제 카드 체크 신호를 기다렸다가 → '보여주기+감지' 단계로
        self.get_logger().info('Waiting: CARD_CHECK')
        await self._wait_intent(allowed={'CARD_CHECK'})
        return S.OPP_CARD_DETECT

    async def _st_detect_and_show(self):
        """
        CARD_CHECK 이후 단계:
        - '보여주기 위치'로 이동 (카드 든 채로)
        - SHOW + DETECT를 동시에 시작하고 결과를 처리
        - 처리 후 제스처 관찰 단계로 전이
        """
        # ★ 카드 보여주기 전용 자세로 이동 (포즈명은 환경에 맞게)
        await self._exec_action(
            "Move(show_card)",
            self.ac_move,
            MoveNamedPose.Goal(name="card_detect", velocity=70.0, acc=50.0),
            retry=1
        )

        # 동시에 시작
        show_task = asyncio.create_task(
            self._exec_action("ShowCard", self.ac_show, ShowCard.Goal(timeout_s=3.0), retry=0)
        )
        detect_task = asyncio.create_task(
            self._exec_action("DetectCardOnce", self.ac_detect, DetectCardOnce.Goal(timeout_s=5.0), retry=1)
        )

        # Detect 먼저 회수
        detect_ok, detect_res, _ = await detect_task
        if detect_ok and detect_res is not None:
            self.opp_card_rank = int(detect_res.rank)
            self.get_logger().info(f"[Detect] Opp card = {self.opp_card_rank}")
        else:
            self.get_logger().warn("Detect failed; opp_card_rank=0 유지")

        # Show 회수
        show_ok, show_res, _ = await show_task
        if not show_ok:
            self.get_logger().warn("ShowCard not succeeded")
        else:
            self._log_ok_msg("ShowCard", show_ok, GoalStatus.STATUS_SUCCEEDED, show_res)

        # ★ 디텍트/쇼 끝나면 제스처 단계로 전이
        return S.OBSERVE_GESTURE

    async def _st_observe_gesture(self):
        """
        - 제스처 뷰로 이동
        - ObserveGestures 실행 (nose/arms/ear)
        - 홈 복귀
        - PLAY_START 대기 → AGENT_DECIDE_OPEN
        """
        # gesture_view로 이동
        await self._exec_action(
            "Move(gesture_view)",
            self.ac_move,
            MoveNamedPose.Goal(name="gesture_view", velocity=70.0, acc=50.0),
            retry=1
        )

        # 제스처 관찰
        ok, res, _ = await self._exec_action(
            "ObserveGestures",
            self.ac_tells,
            ObserveGestures.Goal(duration_s=10.0),
            retry=0
        )
        if ok and res is not None and getattr(res, "ok", True):
            self.tells = (
                bool(getattr(res, 'nose', False)),
                bool(getattr(res, 'arms', False)),
                bool(getattr(res, 'ear', False)),
            )
            self.get_logger().info(f'[Tells] nose={self.tells[0]} arms={self.tells[1]} ear={self.tells[2]}')

        # 홈 복귀
        await self._exec_action(
            "Move(home@after-gesture)",
            self.ac_move,
            MoveNamedPose.Goal(name="home", velocity=70.0, acc=50.0),
            retry=1
        )

        # 베팅 루프 진입 신호 대기
        self.get_logger().info('Waiting: PLAY_START')
        await self._wait_intent(allowed={'PLAY_START'})
        return S.AGENT_DECIDE_OPEN

    async def _st_agent_open(self):
        if self._forced_result:
            self._winner = self._forced_to_winner(self._forced_result)
            return S.RESOLVE

        if self.first_actor_is_agent:
            act = self._policy_decide(to_call=0)
            if not act.startswith('RAISE_'):
                self.get_logger().warn(f'Open act {act} invalid; fallback RAISE_1')
                act = 'RAISE_1'
            await self._agent_bet(act)
        return S.WAIT_HUMAN  # 베팅 루프는 항상 사람부터 시작

    # 사람의 행동 대기 및 실행
    async def _st_wait_human(self):
        if self._forced_result:
            self._winner = self._forced_to_winner(self._forced_result)
            return S.RESOLVE

        self.get_logger().info('[WaitHuman] say: CALL | RAISE(1/2/3) | FOLD')
        intent, amount, param = await self._wait_intent(allowed={'CALL', 'RAISE', 'FOLD', 'SET_RESULT'})

        if intent == 'SET_RESULT':
            self._winner = self._forced_to_winner(param)
            return S.RESOLVE

        pre_need = self._need_for_opp()  # 사람 행동 직전 필요 콜 금액
        act = f'RAISE_{amount}' if intent == 'RAISE' else intent
        self._apply_human_action(act)  # 회계 반영

        if act == 'FOLD':
            self._winner = 'agent'
            return S.RESOLVE

        if act == 'CALL':
            if pre_need == 0:
                self.get_logger().info('[Human] CHECK')
                return S.AGENT_RESPOND
            else:
                if self._need_for_opp() > 0:
                    self.get_logger().info('[Human] CALL (short all-in)')
                else:
                    self.get_logger().info('[Human] CALL')
                self._winner = 'showdown'
                return S.RESOLVE

        # 사람 레이즈면 로봇 응수
        return S.AGENT_RESPOND

    async def _st_agent_respond(self):
        if self._forced_result:
            self._winner = self._forced_to_winner(self._forced_result)
            return S.RESOLVE

        my_act = self._policy_decide(to_call=self.st.to_call)
        self.get_logger().info(f'[Agent] {my_act}')

        if self.st.to_call == 0 and my_act == 'FOLD':
            # 베팅 안 걸린 상태에서 FOLD는 비정상 → CHECK로 보정
            self.get_logger().warn('Agent tried FOLD with to_call=0; treating as CHECK.')
            self.pub_agent.publish(String(data='CHECK'))
            return S.WAIT_HUMAN

        if my_act == 'FOLD':
            self.pub_agent.publish(String(data='FOLD'))
            self._winner = 'opp'
            return S.RESOLVE

        if my_act == 'CALL':
            need = self._need_for_agent()
            pay = min(need, self.st.stack_agent)
            self.st.invest_agent += pay
            self.st.stack_agent -= pay
            self.st.pot += pay
            self.st.to_call = max(0, self.st.invest_opp - self.st.invest_agent)
            self.pub_agent.publish(String(data="CALL" if pay > 0 else "CHECK"))
            self._winner = 'showdown'
            return S.RESOLVE

        if my_act.startswith('RAISE_'):
            await self._agent_bet(my_act)
            return S.WAIT_HUMAN

        # 방어적 처리
        self.get_logger().warn('Invalid agent act; folding defensively')
        self.pub_agent.publish(String(data='FOLD'))
        self._winner = 'opp'
        return S.RESOLVE

    async def _st_resolve(self):
        winner = self._winner or 'draw'
        self._resolve_and_log(winner)
        # 다음 핸드 선공 갱신
        if winner == 'agent':
            self.first_actor_is_agent = True
        elif winner == 'opp':
            self.first_actor_is_agent = False
        return None  # 핸드 종료

    # ===================== Helpers =============================
    async def _wait_intent(self, allowed: set, timeout: float = None):
        """의도 큐에서 allowed에 해당하는 항목이 나올 때까지 대기"""
        while True:
            # 강제 결과 즉시 처리(allowed에 SET_RESULT가 포함될 때)
            if self._forced_result and 'SET_RESULT' in allowed:
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

    def _forced_to_winner(self, param: str) -> str:
        if param == 'ROBOT_WIN':
            return 'agent'
        if param == 'PLAYER_WIN':
            return 'opp'
        return 'draw'

    async def _call_action(self, client, goal, feedback_cb=None,
                       timeout_s: float = 10.0, result_timeout_s: float = 20.0,
                       goal_response_timeout_s: float = 10.0):
        """
        1) send_goal_async 자체에 'goal 수락 응답' 타임아웃을 건다 (기본 10초로 상향)
        2) accepted 체크
        3) 결과 대기는 asyncio.wait_for로 rclpy Future를 그대로 await (create_task 쓰지 않음)
        """
        # --- 1) goal 전송 + 수락 응답 타임아웃 ---
        try:
            gh = await asyncio.wait_for(
                client.send_goal_async(goal, feedback_callback=feedback_cb),
                timeout=goal_response_timeout_s
            )
        except asyncio.TimeoutError:
            self.get_logger().warn("[call_action] send_goal response timeout")
            return False, None, GoalStatus.STATUS_UNKNOWN
        except Exception as e:
            self.get_logger().error(f"[call_action] send_goal error: {e}")
            return False, None, GoalStatus.STATUS_UNKNOWN

        # --- 2) accepted 여부 ---
        accepted = getattr(gh, "accepted", True)
        if not accepted:
            self.get_logger().warn("[call_action] goal REJECTED by server")
            return False, None, GoalStatus.STATUS_UNKNOWN

        # --- 3) 결과 대기 (rclpy Future를 그대로 wait_for로 감싼다) ---
        try:
            wrapped = await asyncio.wait_for(gh.get_result_async(), timeout=result_timeout_s)
        except asyncio.TimeoutError:
            # 타임아웃이면 취소 시도 후 예외 전파(위쪽에서 처리)
            try:
                await gh.cancel_goal_async()
            finally:
                raise

        ok = (wrapped.status == GoalStatus.STATUS_SUCCEEDED)
        return ok, wrapped.result, wrapped.status

    async def _exec_action(
        self,
        tag: str,
        client: ActionClient,
        goal,
        *,
        retry: int = 1,
        backoff_s: float = 0.2,
        timeout_s: float = 10.0,
        result_timeout_s: float = 20.0,
    ):
        """
        _call_action을 감싸서 에러 처리/재시도/로깅을 일관되게 수행.
        반환: (ok, res, status)
        """
        attempt = 0
        while True:
            attempt += 1
            try:
                print("get_in")
                ok, res, status = await self._call_action(
                    client, goal,
                    timeout_s=timeout_s, result_timeout_s=result_timeout_s
                )
                self._log_ok_msg(f"{tag}#{attempt}", ok, status, res)

            except RuntimeError as e:
                self.get_logger().error(f"[{tag}#{attempt}] action server not ready: {e}")
                ok, res, status = False, None, None

            except asyncio.TimeoutError:
                self.get_logger().warn(f"[{tag}#{attempt}] result timeout (goal canceled)")
                ok, res, status = False, None, None

            # 성공이거나, 더 이상 재시도 없음 → 반환
            if ok or attempt > retry:
                return ok, res, status

            # 재시도 백오프
            await asyncio.sleep(backoff_s)

    async def _wait_all_servers(self, per_server_timeout: float = 10.0):
        # (라벨, 클라이언트) 페어로 관리
        pairs = [
            ("move",   self.ac_move),
            ("pick",   self.ac_pick),
            ("show",   self.ac_show),
            ("bet",    self.ac_bet),
            ("detect", self.ac_detect),
            ("tells",  self.ac_tells),  # ObserveGestures
            ("chip_detect", self.ac_chip_detect)
        ]

        async def wait_one(label: str, client):
            # rclpy 내부 속성이긴 하지만 디버깅용으로 유용
            # (버전 따라 속성명이 다를 수 있어 안전하게 getattr)
            action_name = getattr(client, "_action_name", "<?>")
            self.get_logger().info(f"Waiting for action server: {label} (name='{action_name}') ...")
            ok = await asyncio.to_thread(client.wait_for_server, timeout_sec=per_server_timeout)
            if not ok:
                # 네임스페이스도 같이 보여주기
                ns = self.get_namespace() or "/"
                raise RuntimeError(
                    f"Action server not ready: {label}  "
                    f"(expected: resolved '{ns.rstrip('/')}/{action_name.lstrip('/')})'"
                )
            return label

        # 여러 개를 한 번에 기다리되, 어떤 게 실패했는지 모두 모아 보여줌
        results = await asyncio.gather(
            *(wait_one(label, client) for label, client in pairs),
            return_exceptions=True
        )

        failures = [e for e in results if isinstance(e, Exception)]
        if failures:
            # 첫 에러를 올리되, 로그로 전체 실패 목록을 알려줌
            for e in failures:
                self.get_logger().error(str(e))
            raise failures[0]

        self.get_logger().info("✅ All action servers are ready.")

    # ===================== Accounting ==========================
    def _reset_hand_state(self):
        self._forced_result = None
        self.st.pot = self.carry_pot  # 이전 핸드의 이월 팟 반영
        self.carry_pot = 0
        self.st.to_call = 0
        self.st.invest_agent = 0
        self.st.invest_opp = 0
        self.opp_card_rank = 0
        self.tells = (False, False, False)
        self.opp_raises = 0
        self.last_opp_action = "NONE"
        self.last_opp_raise_size = 0
        self.conf_raise = 0.0

    def _ante(self):
        """판 시작 시 앤티: 양쪽 스택 -1, 팟 +2"""
        put = 1
        self.st.stack_agent -= put
        self.st.stack_opp -= put
        self.st.pot += 2

    def _need_for_agent(self) -> int:
        """로봇이 콜하려면 필요한 금액"""
        return max(0, self.st.invest_opp - self.st.invest_agent)

    def _need_for_opp(self) -> int:
        """사람이 콜하려면 필요한 금액"""
        return max(0, self.st.invest_agent - self.st.invest_opp)

    async def _agent_bet(self, act: str):
        """로봇 레이즈: (콜 차액 + k). 스택 부족 시 올인."""
        self.pub_agent.publish(String(data=act))
        k = int(act.split('_')[1])
        if k not in (1, 2, 3):
            self.get_logger().warn(f'Invalid agent raise size: {k}; clamping to 1')
            k = 1
        need = self._need_for_agent()
        total = need + k
        pay = min(total, self.st.stack_agent)

        # 회계 반영 (기존 그대로)
        self.st.invest_agent += pay
        self.st.stack_agent  -= pay
        self.st.pot          += pay
        self.st.to_call = max(0, self.st.invest_agent - self.st.invest_opp)

        # --- 칩 좌표 요청 (카메라) ---
        points_from_cam = []
        if pay > 0:
            try:
                ok_cam, res_cam, status_cam = await self._call_action(
                    self.ac_chip_detect,
                    DetectChip.Goal(timeout_s=5.0),
                    goal_response_timeout_s=10.0,
                    result_timeout_s=7.0
                )
                self._log_ok_msg("DetectChipOnce", ok_cam, status_cam, res_cam)
                if ok_cam and res_cam is not None and getattr(res_cam, "ok", True):
                    pts = list(getattr(res_cam, "points", []))
                    if len(pts) >= 3:
                        points_from_cam = pts[:3]  # 첫 좌표만 사용(서버에서 배치 처리)
                    else:
                        self.get_logger().warn("[DetectChip] points length < 3; using empty points (server cache)")
                else:
                    self.get_logger().warn("[DetectChip] detection failed; using empty points (server cache)")
            except asyncio.TimeoutError:
                self.get_logger().warn("[DetectChip] timeout; using empty points (server cache)")

        # --- BetChips 호출 (amount=pay, points 전달) ---
        if pay > 0:
            try:
                goal = BetChips.Goal()
                goal.amount = int(pay)         # 총 칩 개수 (3 초과 가능)
                goal.points = points_from_cam  # []면 서버측 캐시 활용
                ok, res, status = await self._call_action(self.ac_bet, goal)
                self._log_ok_msg(f"BetChips({pay})", ok, status, res)
            except asyncio.TimeoutError:
                self.get_logger().warn(f"[BetChips({pay})] result timeout")


    def _apply_human_action(self, act: str):
        """사람 액션 회계: CALL=차액, RAISE_k=차액+k"""
        self.last_opp_action = act

        if act.startswith('RAISE_'):
            k = int(act.split('_')[1])
            if k not in (1, 2, 3):
                self.get_logger().warn(f'Invalid raise size: {k}; clamping to 1')
                k = 1
            need = self._need_for_opp()
            total = need + k
            pay = min(total, self.st.stack_opp)

            self.st.invest_opp += pay
            self.st.stack_opp -= pay
            self.st.pot += pay
            self.st.to_call = max(0, self.st.invest_opp - self.st.invest_agent)

            # 메타
            self.opp_raises += 1
            self.last_opp_raise_size = k
            base_map = {1: 0.3, 2: 0.6, 3: 1.0}
            base = base_map[k]
            self.conf_raise = min(1.0, base + 0.1 * max(0, self.opp_raises - 1))

        elif act == 'CALL':
            need = self._need_for_opp()
            pay = min(need, self.st.stack_opp)

            self.st.invest_opp += pay
            self.st.stack_opp -= pay
            self.st.pot += pay
            self.st.to_call = max(0, self.st.invest_opp - self.st.invest_agent)

        elif act == 'FOLD':
            # 회계 변화 없음 (정산에서 승자 처리)
            pass

    # ===================== Policy ==============================
    def _build_obs(self, to_call_for_me: int):
        opp_card_norm = self.opp_card_rank / 10.0
        my_stack = self.st.stack_agent / self.S0
        opp_stack = self.st.stack_opp / self.S0
        pot_n = self.st.pot / self.S0
        to_call_n = self.st.to_call / self.S0
        p_win_belief = (10 - self.opp_card_rank) / 10.0 if self.opp_card_rank else 0.5

        nose_b, arms_b, ear_b = self.tells
        tells = (
            1.0 if nose_b else 0.0,
            1.0 if arms_b else 0.0,
            1.0 if ear_b else 0.0,
        )
        is_first = 1.0 if self.first_actor_is_agent else 0.0

        a = self.last_opp_action
        onehot = [0.0] * 5
        mapidx = {'FOLD': 0, 'CALL': 1, 'RAISE_1': 2, 'RAISE_2': 3, 'RAISE_3': 4}
        if a in mapidx:
            onehot[mapidx[a]] = 1.0

        last_size = self.last_opp_raise_size / 3.0
        opp_raises = min(self.opp_raises, 3) / 3.0
        conf = self.conf_raise
        stage_flag = 1.0 if self.st.to_call > 0 else 0.0

        return [
            opp_card_norm, my_stack, opp_stack, pot_n, to_call_n,
            p_win_belief, tells[0], tells[1], tells[2], is_first,
            *onehot, last_size, opp_raises, conf, stage_flag,
        ]

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
                idx = random.choice([2, 3, 4])  # RAISE_1..3
            else:
                idx = random.choice([0, 1, 2, 3, 4])
        action_map = ["FOLD", "CALL", "RAISE_1", "RAISE_2", "RAISE_3"]
        act = action_map[idx]
        if to_call <= 0 and act == "CALL":
            act = "RAISE_1"
        return act

    # ===================== Resolve =============================
    def _resolve_and_log(self, winner: str):
        self.get_logger().info(
            f'[Resolve] winner={winner} pot={self.st.pot} '
            f'(Agent:{self.st.invest_agent}, Opp:{self.st.invest_opp})'
        )
        if winner == 'agent':
            self.st.stack_agent += self.st.pot
        elif winner == 'opp':
            self.st.stack_opp += self.st.pot
        elif winner in ('draw', 'showdown'):
            self.carry_pot += self.st.pot  # 팟 전액을 다음 핸드로 이월
            self.get_logger().info(f'Carry-over pot = {self.carry_pot}')
        self.get_logger().info(
            f'End Hand {self.hand_id}: stack A={self.st.stack_agent}, O={self.st.stack_opp}'
        )


def main():
    rclpy.init()
    node = PokerOrchestrator()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)

    t = threading.Thread(target=ex.spin, daemon=True)  # Executor 백그라운드 스레드
    t.start()
    try:
        asyncio.run(node.run())  # 메인 스레드에서 asyncio
    finally:
        rclpy.shutdown()
        t.join(timeout=1.0)


if __name__ == '__main__':
    main()
