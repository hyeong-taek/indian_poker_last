#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import traceback
import rclpy
from rclpy.node import Node

# 패키지 경로
import DR_init
from DR_common2 import posx, posj
from std_msgs.msg import Float32MultiArray, String, Bool
from rclpy.executors import MultiThreadedExecutor
from rclpy.action import ActionServer, CancelResponse, GoalResponse
# 🔧 변경: ReentrantCallbackGroup 사용
from rclpy.callback_groups import ReentrantCallbackGroup

from poker_interfaces.action import MoveNamedPose, PickCard, ShowCard, BetChips
from indian_poker.onrobot import RG

# ---- 로봇 설정
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
VELOCITY, ACC = 70, 50
OFF, ON = 0, 1
GRIPPER_NAME = "rg2"
TOOLCHARGER_IP = "192.168.1.1"
TOOLCHARGER_PORT = "502"

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

# ---- DSR API를 동적으로 import (노드 주입 후에!)
def import_dsr_modules():
    global set_tool, set_tcp, movej, movel, wait, movejx, amovej
    global task_compliance_ctrl, set_desired_force, check_force_condition
    global release_force, release_compliance_ctrl
    global get_digital_input, set_digital_output
    global get_current_posx, get_current_posj, check_position_condition
    global amove_periodic, movesx, movesj, move_periodic, drl_script_stop
    global DR_FC_MOD_REL, DR_AXIS_X, DR_AXIS_Y, DR_AXIS_Z, DR_AXIS_A, DR_AXIS_B, DR_AXIS_C
    global DR_MV_MOD_ABS, DR_TOOL, DR_QSTOP, DR_BASE
    global set_robot_mode, get_robot_mode, ROBOT_MODE_MANUAL, ROBOT_MODE_AUTONOMOUS

    from DSR_ROBOT2 import (
        set_tool, set_tcp, movej, movel, wait, movejx, amovej,
        task_compliance_ctrl, set_desired_force, check_force_condition,
        release_force, release_compliance_ctrl,
        get_digital_input, set_digital_output,
        get_current_posx, get_current_posj, check_position_condition,
        amove_periodic, movesx, movesj, move_periodic, drl_script_stop,
        DR_FC_MOD_REL, DR_AXIS_X, DR_AXIS_Y, DR_AXIS_Z, DR_AXIS_A, DR_AXIS_B, DR_AXIS_C,
        DR_MV_MOD_ABS, DR_TOOL, DR_QSTOP, DR_BASE,
        set_robot_mode, get_robot_mode, ROBOT_MODE_MANUAL, ROBOT_MODE_AUTONOMOUS
    )


# -----------------------------------------------------------------------------
# 컨트롤러: rclpy.Node를 직접 상속하지 않고, 외부에서 생성된 node를 받아 사용
# -----------------------------------------------------------------------------
class PokerRobot:
    def __init__(self, node: Node):
        self.node = node
        self.logger = node.get_logger()

        self.v_joint = VELOCITY
        self.a_joint = ACC
        self.v_lin   = VELOCITY
        self.a_lin   = ACC

        # 그리퍼 설정
        try:
            self.gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)
            self.logger.info(f"[Init] Gripper RG 연결 준비: name={GRIPPER_NAME} ip={TOOLCHARGER_IP}:{TOOLCHARGER_PORT}")
        except Exception as e:
            self.logger.error(f"[Init] Gripper 초기화 실패: {e}\n{traceback.format_exc()}")

        # 🔧 변경: 액션별로 Reentrant 콜백그룹 분리
        self.cb_move = ReentrantCallbackGroup()
        self.cb_pick = ReentrantCallbackGroup()
        self.cb_show = ReentrantCallbackGroup()
        self.cb_bet  = ReentrantCallbackGroup()

        # 포즈 preset
        self.home = posj([0.0, 0.0, 90.0, 0.0, 90.0, 0.0])
        self.card_grip_pos   = posx([290.23, -221.55, -0.32 + 19.0, 101.91, 179.98, -167.75])
        self.card_detect_pos = posj([-3.53180909, -22.98944855, 112.55172729, 7.8614459, 0.45518515, 81.9596405])
        self.bluff_detect_pos= posj([-7.51381588, -74.87239075, 123.9002533, 11.15486908, 47.28850937, 80.92360687])
        self.bet_pose        = self.home  # TODO: 실제 포즈로 교체

        self.named_pose_map = {
            "home":         ('j', self.home),
            "card_grip":    ('x', self.card_grip_pos),
            "card_detect":  ('j', self.card_detect_pos),
            "card_show":    ('j', self.card_detect_pos),  # alias
            "gesture_view": ('j', self.bluff_detect_pos),
            "bet":          ('j', self.bet_pose),
        }

        # 초기 설정
        try:
            self.logger.info("[Init] DSR 모듈 로딩 OK")
            self.logger.info("[Init] 모드 전환: MANUAL → AUTONOMOUS")
            self.set_manual_mode()
            set_tool("Tool Weight")
            set_tcp("rg2")
            self.set_auto_mode()
            self.move_home()  # 현장 상황 따라 주석 가능
            self.release()
        except Exception as e:
            self.logger.error(f"[Init] 초기 시퀀스 실패: {e}\n{traceback.format_exc()}")

        # === Action 서버 등록 ===
        self.logger.info("[Action] creating servers...")

        self.as_move = ActionServer(
            self.node, MoveNamedPose, 'robot/move_named_pose',
            execute_callback=self._exec_move_named_pose,
            goal_callback=self._goal_accept('move'),
            cancel_callback=self._cancel_cb('move'),
            callback_group=self.cb_move
        )
        self.logger.info("[Action] /robot/move_named_pose OK")

        self.as_pick = ActionServer(
            self.node, PickCard, 'robot/pick_card',
            execute_callback=self._exec_pick_card,
            goal_callback=self._goal_accept('pick'),
            cancel_callback=self._cancel_cb('pick'),
            callback_group=self.cb_pick
        )
        self.logger.info("[Action] /robot/pick_card OK")

        self.as_show = ActionServer(
            self.node, ShowCard, 'robot/show_card',
            execute_callback=self._exec_show_card,
            goal_callback=self._goal_accept('show'),
            cancel_callback=self._cancel_cb('show'),
            callback_group=self.cb_show
        )
        self.logger.info("[Action] /robot/show_card OK")

        self.as_bet = ActionServer(
            self.node, BetChips, 'robot/bet_chips',
            execute_callback=self._exec_bet_chips,
            goal_callback=self._goal_accept('bet'),
            cancel_callback=self._cancel_cb('bet'),
            callback_group=self.cb_bet
        )
        self.logger.info("[Action] /robot/bet_chips OK")

        ns = self.node.get_namespace()
        self.logger.info(f"[Info] namespace='{ns}' → 액션 경로는 '{ns}/robot/...' 로 노출됨")

    # ------------------ Goal/Cancel 콜백 ------------------
    def _goal_accept(self, label: str):
        def _cb(goal_request):
            self.logger.info(f"[Goal:{label}] ACCEPT")
            return GoalResponse.ACCEPT
        return _cb

    def _cancel_cb(self, label: str):
        def _cb(goal_handle):
            self.logger.warn(f"[Cancel:{label}] 요청 수신 → drl_script_stop 시도")
            try:
                drl_script_stop()
            except Exception as e:
                self.logger.warn(f"[Cancel:{label}] drl_script_stop 실패: {e}")
            return CancelResponse.ACCEPT
        return _cb

    # ------------------ Action 실행 콜백 ------------------
    def _exec_move_named_pose(self, goal_handle):
        t0 = time.time()
        res = MoveNamedPose.Result()
        fb  = MoveNamedPose.Feedback()
        try:
            name = goal_handle.request.name
            vel  = goal_handle.request.velocity or self.v_joint
            acc  = goal_handle.request.acc      or self.a_joint
            self.logger.info(f"[MoveNamedPose] req name={name} vel={vel} acc={acc}")

            if name not in self.named_pose_map:
                res.ok=False; res.message=f"unknown pose: {name}"
                goal_handle.abort()
                self.logger.error(f"[MoveNamedPose] ABORT: {res.message}")
                return res

            kind, pose = self.named_pose_map[name]
            self.logger.info(f"[MoveNamedPose] kind={kind} pose={pose}")

            fb.progress = 0.1; goal_handle.publish_feedback(fb)
            if kind == 'x':
                self.move_linear(pose, vel=vel, acc=acc)
            else:
                self.move_joint(pose, vel=vel, acc=acc)

            fb.progress = 1.0; goal_handle.publish_feedback(fb)
            res.ok=True; res.message=f"moved to {name}"
            goal_handle.succeed()
            self.logger.info(f"[MoveNamedPose] OK in {int((time.time()-t0)*1000)} ms")
            return res

        except Exception as e:
            res.ok=False; res.message=str(e)
            goal_handle.abort()
            self.logger.error(f"[MoveNamedPose] EXC: {e}\n{traceback.format_exc()}")
            return res

    def _exec_pick_card(self, goal_handle):
        self.logger.info("[Pick][EXEC] start")
        t0 = time.time()
        res = PickCard.Result()
        fb  = PickCard.Feedback()
        try:
            dz    = goal_handle.request.approach_z or 20.0
            width = goal_handle.request.grip_width or 600.0
            self.logger.info(f"[Pick] dz={dz} width={width}")

            fb.progress=0.1; goal_handle.publish_feedback(fb)
            self.logger.info("[Pick] descend")
            self.move_linear(self.pose_trans(self.card_grip_pos, [0,0,-dz,0,0,0]),
                             vel=self.v_lin, acc=self.a_lin)

            self.logger.info(f"[Pick] grip width={width}")
            self.grip_w(width)

            self.logger.info("[Pick] retreat")
            self.move_linear(self.pose_trans(self.card_grip_pos, [0,0,100,0,0,0]),
                             vel=self.v_lin, acc=self.a_lin)

            fb.progress=1.0; goal_handle.publish_feedback(fb)
            res.ok=True; res.message="card picked"
            goal_handle.succeed()
            self.logger.info(f"[Pick] OK in {int((time.time()-t0)*1000)} ms")
            return res

        except Exception as e:
            res.ok=False; res.message=str(e)
            goal_handle.abort()
            self.logger.error(f"[Pick] EXC: {e}\n{traceback.format_exc()}")
            return res

    def _exec_show_card(self, goal_handle):
        t0 = time.time()
        res = ShowCard.Result()
        fb  = ShowCard.Feedback()
        try:
            hold_s   = goal_handle.request.timeout_s if goal_handle.request.timeout_s > 0.0 else 3.0
            settle_s = 0.5
            self.logger.info(f"[Show] start hold_s={hold_s} settle_s={settle_s}")

            fb.progress = 0.2; goal_handle.publish_feedback(fb)
            t1 = time.time()
            while time.time() - t1 < settle_s:
                if goal_handle.is_cancel_requested:
                    res.ok = False; res.message = "canceled during settle"
                    goal_handle.canceled()
                    self.logger.warn("[Show] canceled during settle")
                    return res
                time.sleep(0.05)

            fb.progress = 0.6; goal_handle.publish_feedback(fb)
            t2 = time.time()
            while time.time() - t2 < hold_s:
                if goal_handle.is_cancel_requested:
                    res.ok = False; res.message = "canceled during hold"
                    goal_handle.canceled()
                    self.logger.warn("[Show] canceled during hold")
                    return res
                time.sleep(0.05)

            fb.progress = 1.0; goal_handle.publish_feedback(fb)
            res.ok = True; res.message = f"card shown for {hold_s:.2f}s"
            goal_handle.succeed()
            self.logger.info(f"[Show] OK in {int((time.time()-t0)*1000)} ms")
            return res

        except Exception as e:
            res.ok = False; res.message = str(e)
            goal_handle.abort()
            self.logger.error(f"[Show] EXC: {e}\n{traceback.format_exc()}")
            return res

    def _exec_bet_chips(self, goal_handle):
        t0 = time.time()
        res = BetChips.Result()
        fb  = BetChips.Feedback()
        try:
            amt = int(goal_handle.request.amount)
            self.logger.info(f"[Bet] start amount={amt}")
            if amt not in (1,2,3):
                res.ok=False; res.message=f"amount must be 1|2|3, got {amt}"
                goal_handle.abort()
                self.logger.error(f"[Bet] ABORT: {res.message}")
                return res

            fb.stage="move"; fb.progress=0.2; goal_handle.publish_feedback(fb)
            self.move_joint(self.bet_pose, vel=self.v_joint, acc=self.a_joint)

            fb.stage="pick"; fb.progress=0.5; goal_handle.publish_feedback(fb)
            self.grip_w(400 + 50*amt)  # 단순 가정

            fb.stage="place"; fb.progress=0.8; goal_handle.publish_feedback(fb)
            self.release()

            fb.stage="place"; fb.progress=1.0; goal_handle.publish_feedback(fb)
            res.ok=True; res.message=f"bet {amt} chips"
            goal_handle.succeed()
            self.logger.info(f"[Bet] OK in {int((time.time()-t0)*1000)} ms")
            return res

        except Exception as e:
            res.ok=False; res.message=str(e)
            goal_handle.abort()
            self.logger.error(f"[Bet] EXC: {e}\n{traceback.format_exc()}")
            return res

    # ------------------ 모드 전환 ------------------
    def set_manual_mode(self) -> bool:
        try:
            J = get_current_posj()
            movej(J, vel=5, acc=5)
            set_robot_mode(ROBOT_MODE_MANUAL)
            ok = (get_robot_mode() == ROBOT_MODE_MANUAL)
            if not ok:
                self.logger.warn("[Mode] Manual Mode Change Failed")
            else:
                self.logger.info("[Mode] MANUAL")
            return ok
        except Exception as e:
            self.logger.error(f"[Mode] MANUAL 전환 실패: {e}\n{traceback.format_exc()}")
            return False

    def set_auto_mode(self) -> bool:
        try:
            J = get_current_posj()
            movej(J, vel=5, acc=5)
            set_robot_mode(ROBOT_MODE_AUTONOMOUS)
            ok = (get_robot_mode() == ROBOT_MODE_AUTONOMOUS)
            if not ok:
                self.logger.warn("[Mode] Auto Mode Change Failed")
            else:
                self.logger.info("[Mode] AUTONOMOUS")
            return ok
        except Exception as e:
            self.logger.error(f"[Mode] AUTO 전환 실패: {e}\n{traceback.format_exc()}")
            return False

    # ------------------ Gripper I/O ------------------
    def grip(self):
        try:
            self.gripper.close_gripper()
            self.logger.info("[Gripper] close")
        except Exception as e:
            self.logger.error(f"[Gripper] close 실패: {e}")

    def release(self):
        try:
            self.gripper.open_gripper()
            self.logger.info("[Gripper] open")
        except Exception as e:
            self.logger.error(f"[Gripper] open 실패: {e}")

    def grip_w(self, width):
        try:
            self.gripper.move_gripper(width)
            self.logger.info(f"[Gripper] move width={width}")
        except Exception as e:
            self.logger.error(f"[Gripper] move 실패: {e}")

    # ------------------ Move 래퍼 ------------------
    def move_joint(self, pose, vel=VELOCITY, acc=ACC, time=None, mod=None):
        if mod is None:
            mod = DR_MV_MOD_ABS
        self.logger.info(f"_movej vel={vel} acc={acc} mod={mod}")
        movej(pose, vel=vel, acc=acc, time=time, mod=mod)

    def move_linear(self, pose, vel=VELOCITY, acc=ACC, time=None, ref=None, mod=None):
        if mod is None:
            mod = DR_MV_MOD_ABS
        self.logger.info(f"_movel vel={vel} acc={acc} mod={mod}")
        movel(pose, vel=vel, acc=acc, time=time, ref=ref, mod=mod)

    # ------------------ 유틸 ------------------
    def pose_trans(self, pose, delta_pose):
        base = list(pose) if not isinstance(pose, list) else pose
        delta = list(delta_pose)
        if len(delta) < len(base):
            delta += [0]*(len(base)-len(delta))
        return [x + y for x, y in zip(base, delta)]

    # ------------------ 홈 ------------------
    def move_home(self):
        self.logger.info("[Home] moving...")
        self.move_joint(self.home, vel=self.v_joint, acc=self.a_joint)

    # ------------------ 데모 ------------------
    def card_grip_and_show(self):
        self.logger.info("[Demo] card_grip_and_show")
        self.move_linear(self.pose_trans(self.card_grip_pos, [0.0, 0.0, 20.0, 0.0, 0.0, 0.0]))
        self.move_linear(self.card_detect_pos)
        self.grip_w(600)
        self.move_linear(self.pose_trans(self.card_grip_pos, [0.0, 0.0, 100.0, 0.0, 0.0, 0.0]))
        self.move_joint(self.card_detect_pos)

    def card_release(self):
        self.logger.info("[Demo] card_release")
        self.move_linear(self.pose_trans(self.card_grip_pos, [0.0, 0.0, 100.0, 0.0, 0.0, 0.0]))
        self.move_linear(self.card_detect_pos)
        self.release()
        self.move_linear(self.pose_trans(self.card_grip_pos, [0.0, 0.0, 100.0, 0.0, 0.0, 0.0]))


# --------------------------------------------------------------------------------------------------
def main():
    rclpy.init()

    # ✅ 액션 서버용 노드
    node = rclpy.create_node('poker_robot', namespace=ROBOT_ID)

    # ✅ DSR 서비스/토픽용 “별도 노드” (핵심)
    dsr_node = rclpy.create_node('dsr_api', namespace=ROBOT_ID)
    DR_init.__dsr__node = dsr_node     # DSR 라이브러리가 이 노드를 사용

    # DSR 모듈 import (반드시 dsr_node 주입 후!)
    try:
        import_dsr_modules()
        node.get_logger().info("[DSR] DSR_ROBOT2 imported.")
    except Exception as e:
        node.get_logger().error(f"[DSR] import 실패: {e}\n{traceback.format_exc()}")
        rclpy.shutdown()
        return

    # 컨트롤러 생성(액션 서버는 node에 얹힘)
    try:
        ctrl = PokerRobot(node)
    except Exception as e:
        node.get_logger().error(f"[Fatal] PokerRobot init 실패: {e}\n{traceback.format_exc()}")
        rclpy.shutdown()
        return

    # 🔧 실행자: 액션서버 노드만 등록 (dsr_node는 등록하지 마세요!)
    ex = MultiThreadedExecutor(num_threads=6)
    ex.add_node(node)
    try:
        ex.spin()
    finally:
        rclpy.shutdown()

if __name__ == "__main__":
    main()
