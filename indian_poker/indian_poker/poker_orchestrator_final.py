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
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from action_msgs.msg import GoalStatus

from rclpy import logging as rcl_logging
from rclpy.logging import LoggingSeverity

from std_msgs.msg import String
from poker_interfaces.msg import HumanIntent  # 음성노드 최종 의도만 소비

from poker_interfaces.action import (
    MoveNamedPose,
    PickCard,
    ShowCard,
    BetChips,
    DetectCardOnce,
    ObserveGestures,
    DetectChip,
    FollowGaze

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


intent_qos = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)

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
        self.ac_gaze = ActionClient(self, FollowGaze, '/dsr01/robot/follow_gaze', callback_group=self.cb_group)

        # ===== Topics =====
        self.sub_intent = self.create_subscription(HumanIntent, '/voice/intent', self.on_human_intent, intent_qos)
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
        self.raises_total = 0        # 이 핸드에서 발생한 총 레이즈 수(오프닝 포함)
        self.last_raiser = None      # 'agent' | 'opp' | None (선택: 필요시 로깅)
        self.series_done = False
        self.pose_win  = 'pose_win'
        self.pose_lose = 'pose_lose'
        self.pose_draw = 'pose_draw'

        # ===== 하이라이트 로거 & 파라미터 =====
        self.hi_logger = rcl_logging.get_logger(f'{self.get_name()}.HI')
        self.declare_parameter('hi_only', False)
        _hi_only = bool(self.get_parameter('hi_only').get_parameter_value().bool_value)
        try:
            # 기본: 일반 INFO, 하이라이트 INFO
            rcl_logging.set_logger_level(self.get_logger().name, LoggingSeverity.INFO)
            rcl_logging.set_logger_level(self.hi_logger.name, LoggingSeverity.INFO)
            # hi_only=true 이면 일반 로그를 WARN으로 낮춰 숨김
            if _hi_only:
                rcl_logging.set_logger_level(self.get_logger().name, LoggingSeverity.WARN)
        except Exception:
            pass

        # ===== 모델 로딩 관련 파라미터 =====
        self.policy = None
        self.declare_parameter('model_path', 'best_poker_agent.pt')
        self.declare_parameter('norm_enable', False)
        self.declare_parameter('norm_mean', [])  # 길이 10
        self.declare_parameter('norm_std',  [])  # 길이 10
        self.declare_parameter('max_hands', 10)
        try:
            self.max_hands = int(self.get_parameter('max_hands').get_parameter_value().integer_value)
        except Exception:
            self.max_hands = 10

        self.norm_enable = self.get_parameter('norm_enable').get_parameter_value().bool_value
        _mean = self.get_parameter('norm_mean').get_parameter_value().double_array_value
        _std  = self.get_parameter('norm_std').get_parameter_value().double_array_value
        self._norm_mean = None
        self._norm_std  = None
        if self.norm_enable and len(_mean) == 10 and len(_std) == 10 and TORCH_OK:
            self._norm_mean = torch.tensor(list(_mean), dtype=torch.float32)
            self._norm_std  = torch.tensor(list(_std),  dtype=torch.float32)

        mpath_raw = self.get_parameter('model_path').get_parameter_value().string_value or 'models/best_poker_agent.pt'

        p_raw = Path(mpath_raw)
        here = Path(__file__).resolve().parent  # 이 파일이 있는 폴더
        candidates = [p_raw]

        # 상대경로면 몇 가지 합리적인 위치에서 찾아본다
        if not p_raw.is_absolute():
            candidates += [
                here / p_raw,                              # <pkg>/indian_poker/models/...
                here / 'models' / p_raw.name,             # <pkg>/indian_poker/models/best_poker_agent.pt
                here.parent / p_raw,                       # <pkg>/models/...
                here.parent / 'models' / p_raw.name,       # <pkg>/models/best_poker_agent.pt
                Path.cwd() / p_raw,                        # 현재 작업 디렉터리 기준
            ]

        mpath_resolved = None
        for c in candidates:
            if c.exists():
                mpath_resolved = c
                break

        if mpath_resolved is None:
            # 못 찾으면 원문을 그대로 쓰되 경고 출력
            self.get_logger().warn(
                "Model not found. Tried: " + ", ".join(str(c) for c in candidates)
            )
            mpath_resolved = p_raw

        self.get_logger().info(f"Model path resolved: '{mpath_raw}' -> '{mpath_resolved}'")
        if TORCH_OK:
            try:
                # mpath  → mpath_resolved 로 교체
                self.policy = torch.jit.load(mpath_resolved.as_posix(), map_location='cpu')
                self.policy.eval()
                self.get_logger().info(f'Policy loaded: {mpath_resolved} (obs=10D, actions=5)')
            except Exception as e:
                self.get_logger().warn(f'Policy load failed, fallback to random: {e}')
        else:
            self.get_logger().warn('PyTorch not available; using random policy.')

        # ===== FSM 보조 필드 =====
        self._winner: Optional[str] = None  # 'agent'|'opp'|'draw'|'showdown'|None
        self._handlers = {
            S.HAND_START:           self._st_hand_start,
            S.OPP_CARD_DETECT:      self._st_detect_and_show, # CARD_CHECK 후: show & detect
            S.OBSERVE_GESTURE:      self._st_observe_gesture, # show/detect 후: gesture_view 이동 → Observe
            S.AGENT_DECIDE_OPEN:    self._st_agent_open,
            S.WAIT_HUMAN:           self._st_wait_human,
            S.AGENT_RESPOND:        self._st_agent_respond,
            S.RESOLVE:              self._st_resolve,
        }

        # ===== Deck state (1~10, 각 4장 = 총 40장) =====
        self.declare_parameter('deck_init_count', 4)
        try:
            self.deck_init_count = int(self.get_parameter('deck_init_count').get_parameter_value().integer_value)
        except Exception:
            self.deck_init_count = 4
        self._deck_counts = [self.deck_init_count] * 10  # index 0→rank1 ... index 9→rank10
        self.my_card_rank = 0  # 쇼다운 때 외부로부터 입력받아 기록


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
        r_ok  = getattr(res, "ok", None) if res is not None else None
        r_msg = getattr(res, "message", "") if res is not None else ""
        self.get_logger().info(
            f"[{tag}] ok={ok} status={self._status_str(status)} "
            f"res.ok={r_ok} res.message='{r_msg}'"
        )

    # === 하이라이트 헬퍼 (하이라이트 전용 로거에 예쁜 포맷으로 기록) ===
    def _hi(self, title: str, **kv):
        parts = []
        for k, v in kv.items():
            if isinstance(v, bool):
                vstr = "✓" if v else "—"
            elif k in ('pay', 'need', 'k', 'to_call', 'pot', 'amount'):
                try:
                    vstr = f"🪙x{int(v)}"
                except Exception:
                    vstr = str(v)
            else:
                vstr = str(v)
            parts.append(f"{k}={vstr}")
        self.hi_logger.info(f"[{title}] " + " | ".join(parts))

    # ===================== Sub: HumanIntent =====================
    def on_human_intent(self, msg: HumanIntent):
        intent = (msg.intent or "").upper()
        amount = int(msg.amount)
        param = msg.param or ""
        
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
        
        # 서버 연결 대기
        await self._wait_all_servers()
        self.get_logger().info('All action servers are ready.')
        
        # ✅ 총 핸드 수 제한 루프
        while rclpy.ok() and (self.hand_id < self.max_hands) and (not self.series_done):
            await self._run_hand_fsm()

        # ✅ 시리즈 요약 로그
        self._log_match_summary()   

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

        # 사회자 게임 스타트 음성 대기
        self.get_logger().info(f'=== Hand {self.hand_id} === (say: GAME_START)')
        await self._wait_intent(allowed={'GAME_START'})
        self._ante()

        self._hi("HAND_START", hand=self.hand_id,
                 first=("agent" if self.first_actor_is_agent else "opp"),
                 pot=self.st.pot, A=self.st.stack_agent, O=self.st.stack_opp)

        # 액션 동작 [카드그립위치 이동] --> [카드 그립] 
        await self._exec_action(
            "Move(card)",
            self.ac_move,
            MoveNamedPose.Goal(name="card_grip", velocity=70.0, acc=50.0),
            retry=1
        )
        await self._exec_action(
            "PickCard",
            self.ac_pick,
            PickCard.Goal(approach_z=20.0, grip_width=600),
            retry=1
        )

        # 사회자 게임 체크 음성 대기
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

        # 액션 동작 [카드 인식위치 이동] --> [카드 보여주기] --> [카메라 카드 인식]
        await self._exec_action(
            "Move(show_card)",
            self.ac_move,
            MoveNamedPose.Goal(name="card_detect", velocity=70.0, acc=50.0),
            retry=1
        )

        # 두 액션 동시 시작
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
            self._hi("OPP_CARD", rank=self.opp_card_rank)
        else:
            self.get_logger().warn("Detect failed; opp_card_rank=0 유지")
        
        # Show 회수
        show_ok, show_res, _ = await show_task
        if not show_ok:
            self.get_logger().warn("ShowCard not succeeded")
        else:
            self._log_ok_msg("ShowCard", show_ok, GoalStatus.STATUS_SUCCEEDED, show_res)
        
        return S.OBSERVE_GESTURE

    async def _st_observe_gesture(self):
        """
        - 제스처 뷰로 이동
        - ObserveGestures 실행 (nose/arms/ear)
        - 홈 복귀
        - PLAY_START 대기 → AGENT_DECIDE_OPEN
        """

        # 액션 동작 [제스처 위치로 이동] --> [카메라 제스처 인식] & [로봇 얼굴 트래킹]  --> [홈 위치로 이동]
        await self._exec_action(
            "Move(gesture_view)",
            self.ac_move,
            MoveNamedPose.Goal(name="gesture_view", velocity=70.0, acc=50.0),
            retry=1
        )

        dur = 10.0

        # 동시에 시작
        task_tells = asyncio.create_task(
            self._exec_action("ObserveGestures", self.ac_tells, ObserveGestures.Goal(duration_s=dur), retry=0)
        )
        task_gaze  = asyncio.create_task(
            self._exec_action("FollowGaze", self.ac_gaze, FollowGaze.Goal(duration_s=dur+0.5), retry=0)
        )

        try:
            # 두 태스크 모두 기다림 (예외를 삼키지 않도록)
            (ok, res, _), (_, _, _) = await asyncio.gather(task_tells, task_gaze)
        except Exception as e:
            # 한쪽이 예외로 터지면 다른 태스크 취소 보장
            for t in (task_tells, task_gaze):
                if not t.done():
                    t.cancel()
            raise

        # tells 결과 반영
        if ok and res is not None and getattr(res, "ok", True):
            self.tells = (
                bool(getattr(res, 'nose', False)),
                bool(getattr(res, 'arms', False)),
                bool(getattr(res, 'ear', False)),
            )
            self.get_logger().info(
                f'[Tells] nose={self.tells[0]} arms={self.tells[1]} ear={self.tells[2]}')
            self._hi("GESTURE", nose=self.tells[0], arms=self.tells[1], ear=self.tells[2])


        # 카드 내려놓기
        await self._exec_action(
            "Move(card_place@after-gesture)",
            self.ac_move,
            MoveNamedPose.Goal(name="card_place", velocity=70.0, acc=50.0),
            retry=1
        )

        # 홈 복귀
        await self._exec_action(
            "Move(home@after-gesture)",
            self.ac_move,
            MoveNamedPose.Goal(name="home", velocity=70.0, acc=50.0),
            retry=1
        )

        # 사회자 게임 플레이 음성 대기
        self.get_logger().info('Waiting: PLAY_START')
        await self._wait_intent(allowed={'PLAY_START'})

        return S.AGENT_DECIDE_OPEN

    async def _st_agent_open(self):
        """
        [AGENT_OPEN]
        - 선행(오프닝) 주체가 에이전트일 때 첫 액션을 결정/수행하는 상태.
        - 강제 결과(_forced_result)가 지정된 경우 즉시 RESOLVE로 전이.
        - 에이전트가 선행자라면 정책으로부터 액션을 받아 RAISE_x 형태가 아니면 RAISE_1로 보정.
        - 베팅 수행 후 인간 응답 대기 상태로 전이.
        """
        
        act = None

        # 선행 주체가 에이전트인 경우만 오프닝 베팅 수행
        # if self.first_actor_is_agent:
        #     act = self._policy_decide(to_call=0) # 오프닝은 걸려있는 금액이 없음
        #     if not act.startswith('RAISE_'):
        #         # 오프닝은 최소 RAISE가 되도록 강제 보정
        #         self.get_logger().warn(f'Open act {act} invalid; fallback RAISE_1')
        #         act = 'RAISE_1'
            
        #     # 실제 베팅
        #     await self._agent_bet(act)

        # act_str = "(no-open)" if act is None else act
        # self.get_logger().info(
        #     f'[Agent-Open] {act_str} (opp={self.opp_card_rank}, tells={self.tells}, '
        #     f'stacks={self.st.stack_agent}/{self.st.stack_opp})')
        # return S.WAIT_HUMAN
        act = None
        if self.first_actor_is_agent:
            # 스택 0이면 오프닝 불가 → 체크로 처리
            if self.st.stack_agent <= 0:
                self.get_logger().warn('[Agent-Open] zero stack; treating as CHECK (no-open).')
            else:
                act = self._policy_decide(to_call=0)
                if not act.startswith('RAISE_'):
                    act = 'RAISE_1'
                # 합법화
                act = self._legalize_agent_action(act)
                if act.startswith('RAISE_'):
                    await self._agent_bet(act)
                else:
                    # 레이즈가 불가해서 CHECK로 떨어진 경우
                    act = None

        act_str = "(no-open)" if act is None else act
        self.get_logger().info(
            f'[Agent-Open] {act_str} (opp={self.opp_card_rank}, tells={self.tells}, '
            f'stacks={self.st.stack_agent}/{self.st.stack_opp})')
        return S.WAIT_HUMAN
    
    async def _st_wait_human(self):
        """
        [WAIT_HUMAN]
        - 사람(상대) 입력을 기다려 처리하는 상태.
        - 허용 의도: CALL, RAISE(1/2/3), FOLD, (옵션)SET_RESULT(강제결과).
        - FOLD면 에이전트 승리 즉시 종료, CALL이면 상황에 따라 CHECK 처리 또는 쇼다운 전이.
        - RAISE면 투자액/콜 액수 갱신 후 에이전트 응답 상태로 전이.
        """
        
        #  레이즈 제한: 2회에 도달하면 더 이상 RAISE 허용 안 함
        allowed = {'CALL', 'FOLD'} if self.raises_total >= 2 else {'CALL', 'RAISE', 'FOLD'}

        self.get_logger().info('[WaitHuman] say: ' + ('CALL | FOLD' if self.raises_total >= 2 else 'CALL | RAISE(1/2/3) | FOLD'))
        intent, amount, param = await self._wait_intent(allowed=allowed)
        
        # 현재 상대가 콜하기 위해 필요한 칩(금액)
        pre_need = self._need_for_opp()

        # 실제 액션 문자열로 정규화
        act = f'RAISE_{amount}' if intent == 'RAISE' else intent
        prev_invest_opp = self.st.invest_opp

        # 게임 상태(st.*: 스택/팟/투자액/to_call 등)에 인간 액션을 반영
        self._apply_human_action(act)
        pay = self.st.invest_opp - prev_invest_opp
        # 1) 인간이 폴드 → 즉시 에이전트 승리
        if act == 'FOLD':
            self._winner = 'agent'
            self._hi("HUMAN_FOLD")
            return S.RESOLVE
        
        # 2) 인간이 콜
        if act == 'CALL':
            if pre_need == 0:
                self.get_logger().info('[Human] CHECK')
                self._hi("HUMAN_CHECK")
                return S.AGENT_RESPOND
            else:
                if self._need_for_opp() > 0:
                    self.get_logger().info('[Human] CALL (short all-in)')
                else:
                    self.get_logger().info('[Human] CALL')
                self._hi("HUMAN_CALL", need=pre_need, pay=pay,
                         invest_opp=self.st.invest_opp, stack_opp=self.st.stack_opp,
                         pot=self.st.pot, to_call=self.st.to_call)
                self._winner = 'showdown'
                return S.RESOLVE

        # 사람 레이즈면 로봇 응수 (하이라이트 추가)
        if act.startswith('RAISE_'):
            try:
                k = int(act.split('_')[1])
            except Exception:
                k = 1
            self._hi("HUMAN_RAISE", k=k, need=pre_need, pay=pay,
                     invest_opp=self.st.invest_opp, stack_opp=self.st.stack_opp,
                     pot=self.st.pot, to_call=self.st.to_call, raises_total=self.raises_total)
        return S.AGENT_RESPOND

    async def _st_agent_respond(self):
        """
        [AGENT_RESPOND]
        - 인간의 액션 이후 에이전트 응답.
        - 레이즈 총 2회(cap) 이상: 모델이 FOLD면 FOLD, 나머지(CALL/RAISE_x)는 CALL.
        - cap 이전: 모델 액션 그대로(FOLD/CALL/RAISE_x).
        - CALL은 to_call 만큼 실제로 칩을 낸다(칩 포즈 이동 포함).
        - to_call=0 에서 FOLD는 CHECK로 보정.
        """

        model_act = self._policy_decide(to_call=self.st.to_call)

        if self.raises_total >= 2:
            my_act = 'FOLD' if model_act == 'FOLD' else 'CALL'
            self.get_logger().info(f'[Agent] cap>=2: model={model_act} -> act={my_act}')
            self._hi("AGENT_CAP_DECISION", model=model_act, act=my_act, to_call=self.st.to_call)
        else:
            my_act = model_act
            self.get_logger().info(f'[Agent] {my_act}')

        my_act = self._legalize_agent_action(my_act)
        
        # to_call=0 에서 FOLD는 의미 없음 → CHECK로 보정
        if self.st.to_call == 0 and my_act == 'FOLD':
            self.get_logger().warn('Agent tried FOLD with to_call=0; treating as CHECK.')
            self.pub_agent.publish(String(data='CHECK'))
            self._hi("AGENT_CHECK")
            return S.WAIT_HUMAN

        # 1) 폴드 → 상대 승리
        if my_act == 'FOLD':
            self.pub_agent.publish(String(data='FOLD'))
            self._hi("AGENT_FOLD")
            self._winner = 'opp'
            return S.RESOLVE

        # 2) 콜 → to_call 만큼 지불(물리 동작 포함) → 쇼다운
        if my_act == 'CALL':
            need = self._need_for_agent()
            pay = min(need, self.st.stack_agent)

            # 회계 갱신
            self.st.invest_agent += pay
            self.st.stack_agent  -= pay
            self.st.pot          += pay
            self.st.to_call = max(0, self.st.invest_opp - self.st.invest_agent)

            # 로그/퍼블리시
            self.pub_agent.publish(String(data="CALL" if pay > 0 else "CHECK"))
            self._hi("AGENT_CALL", need=need, pay=pay,
                    invest_agent=self.st.invest_agent, stack_agent=self.st.stack_agent,
                    pot=self.st.pot, to_call=self.st.to_call)

            # 실제 칩 동작
            if pay > 0:
                await self._agent_pay(pay)

            self._winner = 'showdown'
            return S.RESOLVE

        # 3) (cap 이전에만 가능) 레이즈 → 베팅 실행
        if my_act.startswith('RAISE_'):
            await self._agent_bet(my_act)
            return S.WAIT_HUMAN

        # 4) 방어: 이상값이면 폴드
        self.get_logger().warn('Invalid agent act; folding defensively')
        self.pub_agent.publish(String(data='FOLD'))
        self._winner = 'opp'
        return S.RESOLVE


    async def _st_resolve(self):
        """
        [RESOLVE]
        - 사회자의 SET_RESULT(param=ROBOT_WIN|PLAYER_WIN|DRAW, amount=my_card_rank)를 기다려 결과 확정
        - 결과 반영(팟 지급/이월, 덱 업데이트), 엔딩 포즈 실행 후 다음 핸드
        """
        # ⏸️ 사회자 결과 대기
        self.get_logger().info('Waiting: SET_RESULT (param=ROBOT_WIN|PLAYER_WIN|DRAW, amount=my_card_rank)')
        intent, amount, param = await self._wait_intent(allowed={'SET_RESULT'})

        # 결과 해석
        winner = self._forced_to_winner(param)  # 'agent'|'opp'|'draw'
        self.my_card_rank = amount if (1 <= amount <= 10) else 0

        # 결과 반영/로그/덱/스택
        self._resolve_and_log(winner)

         # 엔딩 포즈 → 홈
        await self._pose_after_hand(winner)
        
        # 승자에 따라 다음 핸드의 선행자 갱신
        if winner == 'agent':
            self.first_actor_is_agent = True
        elif winner == 'opp':
            self.first_actor_is_agent = False
        return None

    # ===================== Intent & helpers =====================
    async def _wait_intent(self, allowed: set, timeout: float = None):
        """
        의도 큐(self._intent_q)에서 허용된(intent in allowed) 사용자/음성 입력을 기다린다.
        - timeout: 지정 시 해당 시간 내에 입력이 없으면 경고 로그 후 재시도(루프 지속).
        - _forced_result가 걸려있고 SET_RESULT가 허용되면 즉시 이를 반환.
        반환: (intent, amount, param) 튜플
        """
        while True:
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
        """
        강제 결과 문자열을 내부 승자 표기로 변환.
        - 'ROBOT_WIN'  → 'agent'
        - 'PLAYER_WIN' → 'opp'
        - 그 외        → 'draw'
        """
        
        if param == 'ROBOT_WIN':
            return 'agent'
        if param == 'PLAYER_WIN':
            return 'opp'
        return 'draw'

    async def _pose_after_hand(self, winner: str):
        """승/패/무승부 전용 포즈 → 홈 복귀"""
        pose = self.pose_draw
        if winner == 'agent':
            pose = self.pose_win
        elif winner == 'opp':
            pose = self.pose_lose

        ok, _, _ = await self._exec_action(
            f"Move(endpose:{pose})",
            self.ac_move,
            MoveNamedPose.Goal(name=pose, velocity=70.0, acc=50.0),
            retry=0
        )
        if not ok:
            self.get_logger().warn(f'[EndPose] Failed to move to "{pose}"')

        await self._exec_action(
            "Move(home@after-endpose)",
            self.ac_move,
            MoveNamedPose.Goal(name="home", velocity=70.0, acc=50.0),
            retry=1
        )

    async def _call_action(self, client, goal, feedback_cb=None,
                       timeout_s: float = 10.0, result_timeout_s: float = 20.0,
                       goal_response_timeout_s: float = 10.0):
        """
        단일 ROS 2 액션 호출을 안전하게 감싸는 저수준 래퍼.
        - send_goal_async 응답(GoalHandle 수신) 대기: goal_response_timeout_s
        - get_result_async 결과 대기: result_timeout_s (초과 시 cancel 시도 후 Timeout 전파)
        반환: (ok: bool, result: Optional[Any], status: GoalStatus)
        - ok=True → STATUS_SUCCEEDED
        - ok=False → 타임아웃/예외/거절 등
        """
        
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
        accepted = getattr(gh, "accepted", True)
        if not accepted:
            self.get_logger().warn("[call_action] goal REJECTED by server")
            return False, None, GoalStatus.STATUS_UNKNOWN
        try:
            wrapped = await asyncio.wait_for(gh.get_result_async(), timeout=result_timeout_s)
        except asyncio.TimeoutError:
            # 결과 대기 타임아웃 → 취소 시도 후 상위에서 재시도 판단
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
        액션 호출의 상위 래퍼(재시도 포함).
        - tag: 로그 식별용 태그(예: "Move(home)" / "ShowCard")
        - retry: 실패 시 재시도 횟수 (총 시도는 retry+1)
        - backoff_s: 재시도 간 대기 시간
        - timeout_s/result_timeout_s: 하위 _call_action에 전달
        반환: (ok, res, status)
        """
        attempt = 0
        while True:
            attempt += 1
            try:
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
            if ok or attempt > retry:
                return ok, res, status
            await asyncio.sleep(backoff_s)

    async def _wait_all_servers(self, per_server_timeout: float = 10.0):
        """
        필요한 모든 액션 서버가 준비될 때까지 동시 대기.
        - 각 액션클라이언트의 wait_for_server(timeout_sec=…)를 스레드로 돌려 블로킹 방지.
        - 하나라도 준비 실패 시 예외 발생(첫 실패를 올림).
        - 모두 준비되면 '✅ All action servers are ready.' 로그 출력.
        """
        pairs = [
            ("move",   self.ac_move),
            ("pick",   self.ac_pick),
            ("show",   self.ac_show),
            ("bet",    self.ac_bet),
            ("detect", self.ac_detect),
            ("tells",  self.ac_tells),
            ("chip_detect", self.ac_chip_detect),
            ("gaze",   self.ac_gaze)

        ]
        async def wait_one(label: str, client):
            action_name = getattr(client, "_action_name", "<?>")
            self.get_logger().info(f"Waiting for action server: {label} (name='{action_name}') ...")
            ok = await asyncio.to_thread(client.wait_for_server, timeout_sec=per_server_timeout)
            if not ok:
                ns = self.get_namespace() or "/"
                raise RuntimeError(
                    f"Action server not ready: {label}  "
                    f"(expected: resolved '{ns.rstrip('/')}/{action_name.lstrip('/')})'"
                )
            return label
        
        # 모든 서버 준비를 동시에 기다림
        results = await asyncio.gather(
            *(wait_one(label, client) for label, client in pairs),
            return_exceptions=True
        )
        
        # 실패가 하나라도 있으면 첫 실패를 올림
        failures = [e for e in results if isinstance(e, Exception)]
        if failures:
            for e in failures:
                self.get_logger().error(str(e))
            raise failures[0]
        self.get_logger().info("✅ All action servers are ready.")

    # ===================== Accounting ==========================
    def _reset_hand_state(self):
        self._forced_result = None
        self.st.pot = self.carry_pot
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
        self.my_card_rank = 0

        # 레이즈 카운터 초기화
        self.raises_total = 0
        self.last_raiser = None

    def _ante(self):
        put = 1
        self.st.stack_agent -= put
        self.st.stack_opp -= put
        self.st.pot += 2

    def _need_for_agent(self) -> int:
        return max(0, self.st.invest_opp - self.st.invest_agent)

    def _need_for_opp(self) -> int:
        return max(0, self.st.invest_agent - self.st.invest_opp)

    async def _agent_pay(self, pay: int):
        """
        [공통 칩 지불]
        - pay > 0 이면 칩 포즈로 이동 → 칩 감지 → BetChips(amount=pay) → home 복귀.
        - 상태값 갱신은 호출 측에서 이미 반영된 상태여야 함(이 함수는 물리 동작만 담당).
        """
        if pay <= 0:
            return

        # 칩 포즈 이동
        await self._exec_action(
            "Move(bet_detect@pay)",
            self.ac_move,
            MoveNamedPose.Goal(name="bet_detect", velocity=70.0, acc=50.0),
            retry=1
        )
        await asyncio.sleep(0.2)  # 장면 안정화

        # 칩 감지
        points_from_cam = []
        try:
            ok_cam, res_cam, status_cam = await self._call_action(
                self.ac_chip_detect,
                DetectChip.Goal(timeout_s=5.0),
                goal_response_timeout_s=10.0,
                result_timeout_s=7.0
            )
            self._log_ok_msg("DetectChipOnce@pay", ok_cam, status_cam, res_cam)
            if ok_cam and res_cam is not None and getattr(res_cam, "ok", True):
                pts = list(getattr(res_cam, "points", []))
                if len(pts) >= 3:
                    points_from_cam = pts[:3]
                else:
                    self.get_logger().warn("[DetectChip@pay] points length < 3; using empty points (server cache)")
            else:
                self.get_logger().warn("[DetectChip@pay] detection failed; using empty points (server cache)")
        except asyncio.TimeoutError:
            self.get_logger().warn("[DetectChip@pay] timeout; using empty points (server cache)")

        # 베팅(칩 내려놓기)
        try:
            goal = BetChips.Goal()
            goal.amount = int(pay)
            goal.points = points_from_cam
            ok, res, status = await self._call_action(self.ac_bet, goal, result_timeout_s=60.0)
            self._log_ok_msg(f"BetChips(pay={pay})", ok, status, res)
        except asyncio.TimeoutError:
            self.get_logger().warn(f"[BetChips(pay={pay})] result timeout")
        finally:
            # 홈 복귀
            await self._exec_action(
                "Move(home@after-pay)",
                self.ac_move,
                MoveNamedPose.Goal(name="home", velocity=70.0, acc=50.0),
                retry=1
            )



    async def _agent_bet(self, act: str):
        """
        [에이전트 베팅 실행]
        - act: "RAISE_1|2|3"
        - 현재 to_call + k 만큼 결제/상태 갱신 후, 실제 칩 동작 수행.
        """
        self.pub_agent.publish(String(data=act))
        k = int(act.split('_')[1])
        if k not in (1, 2, 3):
            self.get_logger().warn(f'Invalid agent raise size: {k}; clamping to 1')
            k = 1

        # 레이즈 카운트 증가 (오프닝 포함)
        self.raises_total += 1
        self.last_raiser = 'agent'
        self.get_logger().info(f'[RaiseCount] total={self.raises_total} (by agent)')

        need = self._need_for_agent()
        total = need + k
        pay = min(total, self.st.stack_agent)

        # 상태 갱신
        self.st.invest_agent += pay
        self.st.stack_agent  -= pay
        self.st.pot          += pay
        self.st.to_call = max(0, self.st.invest_agent - self.st.invest_opp)

        self._hi("AGENT_BET", act=act, k=k, need=need, pay=pay,
                invest_agent=self.st.invest_agent, stack_agent=self.st.stack_agent,
                pot=self.st.pot, to_call=self.st.to_call, raises_total=self.raises_total)

        if pay > 0:
            await self._agent_pay(pay)


    def _apply_human_action(self, act: str):
        """
        [인간 액션 반영]
        - 'RAISE_k' / 'CALL' / 'FOLD'을 게임 상태(self.st: 투자액/스택/팟/to_call)에 반영.
        - raise 횟수/크기 추적(opp_raises, last_opp_raise_size) 및
        간단한 신뢰도(conf_raise) 갱신.
        """
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
            
            # 레이즈 메타(횟수/크기/신뢰도) 업데이트
            self.opp_raises += 1
            self.last_opp_raise_size = k
            
            # 간이 신뢰도 모델: 큰 레이즈일수록 base↑, 반복 레이즈 시 +0.1 누적
            base_map = {1: 0.3, 2: 0.6, 3: 1.0}
            base = base_map[k]
            self.conf_raise = min(1.0, base + 0.1 * max(0, self.opp_raises - 1))
        

            #  레이즈 카운트 증가
            self.raises_total += 1
            self.last_raiser = 'opp'
            self.get_logger().info(f'[RaiseCount] total={self.raises_total} (by human)')

        elif act == 'CALL':
            # 콜 필요액만큼 결제(스택 한도)
            need = self._need_for_opp()
            pay = min(need, self.st.stack_opp)
            self.st.invest_opp += pay
            self.st.stack_opp -= pay
            self.st.pot += pay
            self.st.to_call = max(0, self.st.invest_opp - self.st.invest_agent)
        elif act == 'FOLD':
            # 상태 갱신 없음(승패 결정은 상위 상태에서 처리)
            pass
    

    # ===================== Deck helpers =====================

    def _deck_p_win(self, opp_rank: int) -> float:
        """
        p(win | 상대카드=opp_rank, 현재까지 '확정 제거된 카드'만 제외한 덱 상태)
        - 현재 핸드에서 상대 카드 r을 '봤다'는 조건 → 분모에서 카드 1장 제거
        - 승리 장수 = r보다 큰 랭크의 남은 장수 합
        """
        if not (1 <= opp_rank <= 10):
            return 0.5
        counts = self._deck_counts
        total_remain = sum(counts)
        if total_remain <= 1:
            return 0.5  # 방어적 처리

        # 상대가 들고 있는 r 한 장은 분모에서 제외
        denom = total_remain - 1

        # r보다 큰 랭크들의 남은 장수 합 (index r..9)
        wins = sum(counts[opp_rank:])

        return wins / denom if denom > 0 else 0.5

    def _deck_remove(self, rank: int, n: int = 1):
        """해당 랭크 카드를 n장 제거(최소 0 보장). 확정된 공개 카드만 제거한다."""
        if 1 <= rank <= 10:
            i = rank - 1
            self._deck_counts[i] = max(0, self._deck_counts[i] - n)

    def _deck_update_after_hand(self, winner: str):
        """
        핸드 종료 후 확정 정보로 덱을 갱신.
        - 상대 카드는 항상 1장 제거(이미 본 카드)
        - 내 카드는 SET_RESULT.amount가 1~10 일 때만 제거(쇼다운 시 등 실제 공개된 경우)
        """
        # 상대 카드는 OPP_CARD_DETECT에서 이미 확인됨
        if 1 <= self.opp_card_rank <= 10:
            self._deck_remove(self.opp_card_rank, 1)

        # 내 카드는 SET_RESULT.amount가 1~10이면 제거 (승/패/무승부 무관)
        if 1 <= self.my_card_rank <= 10:
            self._deck_remove(self.my_card_rank, 1)

        # 남은 총 장수 로깅 (선택)
        remain = sum(self._deck_counts)
        self.get_logger().info(f'[Deck] remain={remain}, counts={self._deck_counts}')


    # ===================== Policy ==============================
    def _build_obs(self, to_call_for_me: int):
        """
        [정책 입력 관측 벡터(길이 10)]
        구성(env10):
        0: opp_card_norm   (상대 카드 추정/관측 rank/10, 미확정이면 0)
        1: my_stack        (에이전트 스택 / S0)
        2: opp_stack       (상대 스택 / S0)
        3: pot_n           (팟 / S0)
        4: to_call_n       (to_call / S0)   
        5: p_win_belief    (승률 추정치: 상대카드 기반, 미확정이면 0.5)
        6: nose            (tells: 코 만지기 여부 0/1)
        7: arms            (tells: 팔짱 여부 0/1)
        8: ear             (tells: 귀 만지기 여부 0/1)
        9: is_first        (이번 핸드 선행자가 에이전트인가 0/1)

        주의:
        - S0는 기준 스택(초기 스택 등). 0 나눗셈 방지를 위해 S0>0 전제.
        - opp_card_rank가 0이면 미확정으로 보고 p_win_belief=0.5.
        """

        # 상대 카드 정규화(미확정=0)
        opp_card_norm = self.opp_card_rank / 10.0
        
        # 스택/팟/to_call 정규화
        my_stack = self.st.stack_agent / self.S0
        opp_stack = self.st.stack_opp / self.S0
        pot_n = self.st.pot / self.S0
        to_call_n = self.st.to_call / self.S0
        p_win_belief = self._deck_p_win(self.opp_card_rank)

        nose_b, arms_b, ear_b = self.tells
        nose_n = 1.0 if nose_b else 0.0
        arms_n = 1.0 if arms_b else 0.0
        ear_n  = 1.0 if ear_b  else 0.0

        is_first = 1.0 if self.first_actor_is_agent else 0.0

        return [
            opp_card_norm, my_stack, opp_stack,
            pot_n, to_call_n, p_win_belief,
            nose_n, arms_n, ear_n, is_first
        ]

    def _policy_decide(self, to_call: int) -> str:
        obs = self._build_obs(to_call)
        if TORCH_OK and self.policy is not None:
            with torch.no_grad():
                x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # [1,10]
                # 선택적 정규화
                if self.norm_enable and self._norm_mean is not None and self._norm_std is not None:
                    if x.shape[-1] == self._norm_mean.shape[0]:
                        x = (x - self._norm_mean) / (self._norm_std + 1e-6)
                logits = self.policy(x)  # [1,5] 가정
                if logits.shape[-1] != 5:
                    self.get_logger().warn(f'Policy output dim {logits.shape[-1]} != 5; fallback to random.')
                    idx = None
                else:
                    idx = int(torch.argmax(logits, dim=-1).item())
        else:
            idx = None

        if idx is None:
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

    def _legalize_agent_action(self, act: str) -> str:
        """
        모델이 고른 act를 현재 스택/상황에 맞는 합법 액션으로 보정한다.
        - 스택이 0이면: to_call>0 → CALL(= 0칩 올인), to_call==0 → CALL(실제로는 CHECK로 퍼블리시)
        - 레이즈는 최소 need + k 가 가능할 때만 허용
        - to_call==0 에서 FOLD는 의미 없음 → 기존 로직대로 CHECK 처리되도록 FOLD 유지
        """
        need = self._need_for_agent()
        s = self.st.stack_agent

        # 칩이 하나도 없으면: 콜(=올인0) 또는 체크만 허용
        if s <= 0:
            return "CALL"  # need==0이면 아래 CALL 분기에서 CHECK로 퍼블리시됨

        # 레이즈 가능성 체크
        if act.startswith("RAISE_"):
            try:
                k = int(act.split('_')[1])
            except Exception:
                k = 1
            # 최소 레이즈 금액(need + k)을 못 채우면 레이즈 금지 → CALL/체크
            if s < (need + k):
                return "CALL"  # need==0이면 CHECK 퍼블리시

        # to_call==0에서 FOLD는 기존 분기에서 CHECK로 보정되므로 그대로 둔다
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
        elif winner == 'draw':
            self.carry_pot += self.st.pot
            self.get_logger().info(f'Carry-over pot = {self.carry_pot}')
        self.get_logger().info(
            f'End Hand {self.hand_id}: stack A={self.st.stack_agent}, O={self.st.stack_opp}'
        )
        self._hi("RESOLVE", hand=self.hand_id, winner=winner, pot=self.st.pot,
                 A=self.st.stack_agent, O=self.st.stack_opp, carry=self.carry_pot)
        
        # 핸드 결과에 따라 덱 상태 갱신 (확정된 카드만 제거)
        self._deck_update_after_hand(winner)

        # ✅ 조기 종료 조건 (한쪽이 모든 stack 소실 시)
        if self.st.stack_agent <= 0 or self.st.stack_opp <= 0:
            self.series_done = True

    # ===================== match_summary =============================
    def _log_match_summary(self):
        a, o = self.st.stack_agent, self.st.stack_opp
        if a > o: win = 'agent'
        elif o > a: win = 'opp'
        else: win = 'draw'
        self.get_logger().info(
            f'[Match] Finished: hands={self.hand_id}/{self.max_hands}, final stacks A={a}, O={o}, winner={win}'
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
