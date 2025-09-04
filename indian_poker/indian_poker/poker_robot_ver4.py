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
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy, DurabilityPolicy

from poker_interfaces.action import MoveNamedPose, PickCard, ShowCard, BetChips, FollowGaze
from indian_poker.onrobot import RG
from indian_poker.gaze_tracking_control import GazeTracker

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

RT_NO_BACKLOG = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,                                 # 큐 1개(사실상 최신 샘플만 유지)
    reliability=ReliabilityPolicy.BEST_EFFORT,  # 따라잡지 못하면 드롭
    durability=DurabilityPolicy.VOLATILE        # 과거(라칭) 미보관
)

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


        # ====================================== Chips 픽/플레이스 파라미터 ======================================
        self.chip_height_mm   = 15                  # 칩 한 장 두께(실측 보정)
        self.pick_offsets_mm  = {1:15.0, 2:34.0, 3:50.0}  # 개수별 추가 하강 오프셋
        self.pick_approach_mm = 20.0                 # 접근 높이(+z)
        self.pick_safe_up_mm  = 60.0                 # 리프트 높이(+z)
        self._chip_cached_xyz = None                 # 칩 스택 (x,y,z_top) 캐시
        self._pick_rpy = None

        # ====================================== Gaze tracking runtime ======================================
        self.gaze_tracker = None
        self._gaze_sub = None  # subscription handle (enable/disable by create/destroy)

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
        self.cb_follow = ReentrantCallbackGroup()

        #  ====================================== 포즈 preset ======================================
        self.home = posj([0.0, 0.0, 90.0, 0.0, 90.0, 0.0])
        self.card_grip_pos   = posx([290.23, -221.55, -0.32 + 18.0, 101.91, 179.98, -167.75])
        self.card_detect_pos = posj([-3.53180909, -22.98944855, 112.55172729, 7.8614459, 0.45518515, 81.9596405])
        self.card_place_pos   = posj([-3.79808884e+01,  5.94513369e+00,  1.12330032e+02,  2.56229453e-02,
        6.17386055e+01,  5.19806252e+01])
        self.bluff_detect_pos= posj([-7.51381588, -74.87239075, 123.9002533, 11.15486908, 47.28850937, 80.92360687])
        self.bet_pos         = posx([495.55, 15.25, 142.36, 168.52, -154.47, 164.46])
        self.bet_detect_pos  = posj([ 3.66886177e+01, -7.82176971e+00,  8.74181137e+01, -7.45938271e-02,
        1.00406769e+02,  3.90628128e+01])
        self.win_pos         = posj([0.0, 0.0, 90.0, 0.0, -90.0, 0.0])
        self.lose_pos        = posj([0.0, 0.0, 90.0, 0.0, 90.0, 0.0])
        self.draw_pos        = posj([0.0, 0.0, 90.0, 0.0, 0.0, 0.0])

        self.named_pose_map = {
            "home":         ('j', self.home),
            "card_grip":    ('x', self.card_grip_pos),
            "card_place":   ('j', self.card_place_pos),
            "card_detect":  ('j', self.card_detect_pos),
            "card_show":    ('j', self.card_detect_pos),  
            "gesture_view": ('j', self.bluff_detect_pos),
            "bet":          ('x', self.bet_pos),
            "bet_detect":   ('j', self.bet_detect_pos),
            "pose_win":     ('j', self.win_pos),
            "pose_draw":    ('j', self.draw_pos),
            "pose_lose":    ('j', self.lose_pos)
        }

        self.post_hooks = {
            "card_place": self._post_card_place,
            "pose_win":   self._post_pose_win,
            "pose_lose":  self._post_pose_lose,
            "pose_draw":  self._post_pose_draw,
        }
        

        # ====================================== 초기 설정 ======================================
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

        # ============================== Action 서버 등록 ===========================================
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


        self.as_follow = ActionServer(
        self.node, FollowGaze, 'robot/follow_gaze',
        execute_callback=self._exec_follow_gaze,
        goal_callback=self._goal_accept('follow_gaze'),
        cancel_callback=self._cancel_cb('follow_gaze'),
        callback_group=self.cb_follow
        )
        self.logger.info("[Action] /robot/follow_gaze OK")

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

            # === post-action hooks by name ===
            try:
                hook = self.post_hooks.get(name)
                if hook:
                    hook(kind, pose)   # 바운드 메서드라 self 안 넘겨도 됨
            except Exception as pe:
                self.logger.warn(f"[MoveNamedPose] post-action failed: {pe}")

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
            amt_total = int(goal_handle.request.amount)
            pts = list(getattr(goal_handle.request, "points", [])) if hasattr(goal_handle.request, "points") else []
            self.logger.info(f"[Bet] start amount={amt_total} points_len={len(pts)}")

            if amt_total <= 0:
                res.ok=False; res.message=f"amount must be >=1, got {amt_total}"
                goal_handle.abort()
                self.logger.error(f"[Bet] ABORT: {res.message}")
                return res

            # --- 픽 시작 좌표 결정 (요청 > 캐시) ---
            if len(pts) >= 3:
                self._chip_cached_xyz = [float(pts[0]), float(pts[1]), float(pts[2])]
            xyz = self._chip_cached_xyz
            if xyz is None:
                res.ok=False; res.message="no chip target: provide points in goal or set cache earlier"
                goal_handle.abort()
                self.logger.error(f"[Bet] ABORT: {res.message}")
                return res

            x0, y0, z_top = xyz
            self.logger.info(f"[Bet] using xyz=({x0:.2f},{y0:.2f},{z_top:.2f})")

            remain, removed = amt_total, 0



            if self._pick_rpy is None:
                curr = get_current_posx()[0]
                self._pick_rpy = (curr[3], curr[4], curr[5])
            # === 배치 루프 (한 번에 최대 3개) ===
            while remain > 0:
                if goal_handle.is_cancel_requested:
                    res.ok=False; res.message="canceled"
                    goal_handle.canceled()
                    self.logger.warn("[Bet] canceled by client")
                    return res

                take = min(3, remain)

                # move/pick: 베이스 xyz만 변경, EE 회전 유지
                fb.stage="move"; fb.progress = float(removed)/max(1.0, amt_total); goal_handle.publish_feedback(fb)
                self._pick_batch_base_xyz(x0, y0, z_top, take, removed)

                # place: 미리 정한 bet_pos(posx)로 이동 → release
                fb.stage="place"; fb.progress = float(removed + 0.6)/max(1.0, amt_total); goal_handle.publish_feedback(fb)
                self.move_linear(self.bet_pos, vel=self.v_lin, acc=self.a_lin)
                self.release()

                removed += take
                remain  -= take
                fb.stage="place"; fb.progress = float(removed)/max(1.0, amt_total); goal_handle.publish_feedback(fb)

            # 완료
            res.ok=True; res.message=f"bet {amt_total} chips (batched)"
            goal_handle.succeed()
            self.logger.info(f"[Bet] OK total={amt_total} in {int((time.time()-t0)*1000)} ms")
            return res

        except Exception as e:
            res.ok=False; res.message=str(e)
            goal_handle.abort()
            self.logger.error(f"[Bet] EXC: {e}\n{traceback.format_exc()}")
            return res
    # ------------------ GazeTracker methods ----------------------------
    def _ensure_gaze_tracker(self):
        """Create GazeTracker once (holds its own control timer)."""
        if self.gaze_tracker is None:
            # DSR API ensure: the module import is already done in main(); just use globals
            self.gaze_tracker = GazeTracker(
                node=self.node,
                amovej_func=amovej,
                posj_func=posj,
                get_current_posj_func=get_current_posj
            )
            self.logger.info("[FollowGaze] GazeTracker initialized.")

    def _on_face_loc(self, msg: Float32MultiArray):
        """Subscribe callback to feed face center into GazeTracker."""
        if self.gaze_tracker is None:
            return
        try:
            if len(msg.data) >= 6:
                cx = float(msg.data[4])
                cy = float(msg.data[5])
                self.gaze_tracker.face_data_callback((cx, cy))
            else:
                self.logger.warn("[FollowGaze] face_loc msg length < 6")
        except Exception as e:
            self.logger.warn(f"[FollowGaze] face_loc parse error: {e}")

    def _subscribe_face(self):
        """Enable gaze following by subscribing to face topic."""
        if self._gaze_sub is None:
            self._gaze_sub = self.node.create_subscription(
                Float32MultiArray, f"/{ROBOT_ID}/face_loc",
                self._on_face_loc, RT_NO_BACKLOG
            )
            self.logger.info(f"[FollowGaze] Subscribed /{ROBOT_ID}/face_loc")

    def _unsubscribe_face(self):
        """Disable gaze following by dropping the subscription."""
        if self._gaze_sub is not None:
            try:
                self.node.destroy_subscription(self._gaze_sub)
            except Exception:
                pass
            self._gaze_sub = None
            # ⬇️ 추가: 바로 '타겟 없음' 상태로 전환 (드리프트 방지)
            if self.gaze_tracker:
                self.gaze_tracker.face_data_callback(None)
            self.logger.info("[FollowGaze] Unsubscribed /face_loc & tracker cleared")

    def _exec_follow_gaze(self, goal_handle):
        """Action execution: follow gaze for duration_s seconds (or until canceled)."""
        res = FollowGaze.Result()
        fb  = FollowGaze.Feedback()
        try:
            duration = float(goal_handle.request.duration_s) if goal_handle.request.duration_s > 0.0 else 10.0
            self.logger.info(f"[FollowGaze] start duration_s={duration:.2f}")

            # Prepare tracker & subscription
            self._ensure_gaze_tracker()
            self._subscribe_face()

            t0 = time.time()
            # simple time-based loop; GazeTracker has its own timer for control
            while rclpy.ok():
                if goal_handle.is_cancel_requested:
                    self._unsubscribe_face()
                    res.ok = False
                    res.message = "canceled"
                    goal_handle.canceled()
                    self.logger.warn("[FollowGaze] canceled by client")
                    return res

                elapsed = time.time() - t0
                fb.progress = min(1.0, float(elapsed / max(1e-3, duration)))
                goal_handle.publish_feedback(fb)

                if elapsed >= duration:
                    break
                time.sleep(0.05)

            # done
            self._unsubscribe_face()
            res.ok = True
            res.message = f"followed for {duration:.2f}s"
            goal_handle.succeed()
            self.logger.info("[FollowGaze] OK")
            return res

        except Exception as e:
            self._unsubscribe_face()
            res.ok = False
            res.message = str(e)
            self.logger.error(f"[FollowGaze] EXC: {e}\n{traceback.format_exc()}")
            goal_handle.abort()
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
    
    def _pick_batch_base_xyz(self, x: float, y: float, z_top: float, take: int, removed: int):
        """
        x,y,z_top: 칩 스택 최상단 좌표(베이스, mm)
        take     : 이번에 집을 칩 수 (1|2|3)
        removed  : 지금까지 옮긴 칩 수
        """
        offset_down = self.pick_offsets_mm.get(take, 15.0)
        z_pick = z_top - removed*self.chip_height_mm - offset_down # 잡으러 가는 Z 위치

        # 현재 EE 회전 유지
        rx, ry, rz = self._pick_rpy if self._pick_rpy else get_current_posx()[0][3:6]

        p_approach = posx([x, y, z_top + self.pick_approach_mm, rx, ry, rz])
        p_pick     = posx([x, y, z_pick,                          rx, ry, rz])
        p_lift     = posx([x, y, z_top + self.pick_safe_up_mm,    rx, ry, rz])

        # approach → pick
        self.move_linear(p_approach, vel=self.v_lin, acc=self.a_lin)
        self.move_linear(p_pick,     vel=self.v_lin, acc=self.a_lin)

        # grip (현장 보정)
        self.grip_w(300)
        time.sleep(3.0)

        # lift
        self.move_linear(p_lift, vel=self.v_lin, acc=self.a_lin)

    def _post_card_place(self, kind, pose):
        self.logger.info("[MoveNamedPose] post: release @ card_place")
        self.release()
        time.sleep(0.5)
        if kind == 'x':  # Cartesian
            safe_up = self.pose_trans(pose, [0, 0, 100.0, 0, 0, 0])
            self.move_linear(safe_up, vel=self.v_lin, acc=self.a_lin)

    def _post_pose_win(self, kind, pose):
        # 작은 세리머니: 위로 톡 올렸다가 원위치
        try:
            if kind == 'j':
                self.move_linear([0.0, 0.0, 50.0, 0.0, 0.0, 0.0], vel=self.v_lin, acc=self.a_lin, ref=DR_TOOL)
                self.move_linear([0.0, 0.0, -50.0, 0.0, 0.0, 0.0], vel=self.v_lin, acc=self.a_lin, ref=DR_TOOL)
                self.move_linear([0.0, 0.0, 50.0, 0.0, 0.0, 0.0], vel=self.v_lin, acc=self.a_lin, ref=DR_TOOL)
                self.move_linear([0.0, 0.0, -50.0, 0.0, 0.0, 0.0], vel=self.v_lin, acc=self.a_lin, ref=DR_TOOL)
                self.move_joint(self.home, vel=self.v_lin, acc=self.a_lin)
        except Exception as e:
            self.logger.warn(f"[post_pose_win] failed: {e}")

    def _post_pose_lose(self, kind, pose):
        # 살짝 아래로 (풀이 죽은 제스처)
        try:
            if kind == 'j':
                self.move_linear([0.0, 0.0, 50.0, 0.0, 0.0, 0.0], vel=self.v_lin, acc=self.a_lin, ref=DR_TOOL)
                self.move_linear([0.0, 0.0, -50.0, 0.0, 0.0, 0.0], vel=self.v_lin, acc=self.a_lin, ref=DR_TOOL)
                self.move_linear([0.0, 0.0, 50.0, 0.0, 0.0, 0.0], vel=self.v_lin, acc=self.a_lin, ref=DR_TOOL)
                self.move_linear([0.0, 0.0, -50.0, 0.0, 0.0, 0.0], vel=self.v_lin, acc=self.a_lin, ref=DR_TOOL)
                self.move_joint(self.home, vel=self.v_lin, acc=self.a_lin)
        except Exception as e:
            self.logger.warn(f"[post_pose_lose] failed: {e}")

    def _post_pose_draw(self, kind, pose):
        # 고개 끄덕이듯 위/아래 한 번
        try:
            if kind == 'j':
                self.move_linear([0.0, 50.0, 0.0, 0.0, 0.0, 0.0], vel=self.v_lin, acc=self.a_lin, ref=DR_TOOL)
                self.move_linear([0.0, -50.0, 0.0, 0.0, 0.0, 0.0], vel=self.v_lin, acc=self.a_lin, ref=DR_TOOL)
                self.move_linear([0.0, 50.0, 0.0, 0.0, 0.0, 0.0], vel=self.v_lin, acc=self.a_lin, ref=DR_TOOL)
                self.move_linear([0.0, -50.0, 0.0, 0.0, 0.0, 0.0], vel=self.v_lin, acc=self.a_lin, ref=DR_TOOL)
                self.move_joint(self.home, vel=self.v_lin, acc=self.a_lin)
        except Exception as e:
            self.logger.warn(f"[post_pose_draw] failed: {e}")

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
