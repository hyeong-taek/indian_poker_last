#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
쉬운 통합을 위해 단일 클래스로 캡슐화된 시선 추적 기능.
"""

import time
import statistics
from collections import deque

# ========= 제어 파라미터 기본값 =========
DEFAULT_CONFIG = {
    'W': 640,                  # 입력 영상 가로 해상도
    'H': 480,                  # 입력 영상 세로 해상도
    'SAFE_START_SEC': 1.0,     # 시작 시 안전 대기 시간 (초) → 로봇 튀는 거 방지
    'SEND_HZ': 6.0,           # 제어 명령 송신 주기 (Hz) → 초당 몇 번 명령 보낼지
    'MOVE_TIME': 0.40,          # amovej 명령에 들어가는 이동 시간 (s) (1/SEND_HZ보다 커야함)
    'LEAD_SEC': 0.50,           # 예측 리드 시간 (s) → 현재 오차 + 미래 오차 보정
    'TARGET_X_RATIO': 0.50,    # 화면 가로 목표 위치 (0~1 비율, 0.5면 정중앙)
    'TARGET_Y_RATIO': 0.40,    # 화면 세로 목표 위치 (0~1 비율, 0.5면 정중앙)
    'DEAD_BAND_PX': 20,        # 오차 허용 데드존 (픽셀 단위) → 이 이내는 무시
    'ERR_LP': 0.60,            # 오차 Low-pass filter 계수 (0~1, 높을수록 반응 느림)
    'GAIN_X_DPS_PER_PX': 0.10, # 가로 오차 대비 J1 속도 게인 (deg/sec per pixel)
    'GAIN_Y_DPS_PER_PX': -0.10,# 세로 오차 대비 J5 속도 게인 (deg/sec per pixel)
    'MAX_DPS_J1': 5.5,         # J1 축 최대 속도 제한 (deg/sec)
    'MAX_DPS_J5': 5.0,         # J5 축 최대 속도 제한 (deg/sec)
    'EPS_CMD_DEG': 0.03,       # 명령 무시 임계값 (deg) → 이 이하 변화는 무시
    'ZERO_HOLD_TICKS': 2,      # 오차 0 근처일 때 명령 유지 틱 수
    'LOST_HOLD_SEC': 0.2,      # 타겟 놓쳤을 때 유지 시간 (초)
    'MOVE_VEL': 40,             # 로봇 움직임 속도 스케일 (Doosan API vel)
    'MOVE_ACC': 40,             # 로봇 움직임 가속도 스케일 (Doosan API acc)
}


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

class GazeTracker:
    """기존 ROS2 노드에 시선 추적 기능을 추가하기 위한 독립형 클래스입니다."""

    def __init__(self, node, amovej_func, posj_func, get_current_posj_func, config=None):
        """
        GazeTracker 기능을 초기화합니다.
        :param node: 부모 rclpy.node.Node 객체 (로깅 및 타이머용).
        :param amovej_func: DSR 라이브러리에서 임포트된 amovej 함수.
        :param posj_func: DSR 라이브러리에서 임포트된 posj 함수.
        :param get_current_posj_func: DSR 라이브러리에서 임포트된 get_current_posj 함수.
        :param config: 기본값을 덮어쓸 제어 파라미터 딕셔너리.
        """
        self._node = node
        self._amovej = amovej_func
        self._posj = posj_func
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)

        self.t_start = time.time()

        # --- 내부 상태 ---
        self.errx_lp = 0.0
        self.erry_lp = 0.0
        self.have_face = False
        self.last_seen = 0.0
        self.cx_buf = deque(maxlen=5)
        self.cy_buf = deque(maxlen=5)
        self._prev_ex = 0.0
        self._prev_ey = 0.0
        self._zero_hold = 0
        
        # --- 로봇 상태 ---
        initial_posj = get_current_posj_func()
        if not isinstance(initial_posj, (list, tuple)) or len(initial_posj) < 6:
            raise RuntimeError(f"유효한 초기 관절 위치를 가져오는 데 실패했습니다: {initial_posj}")
        self.j_last_cmd = list(map(float, initial_posj[:6]))
        self.lead_target = self.j_last_cmd[:]

        # --- 화면상 목표 지점 ---
        self.tx = int(self.config['W'] * self.config['TARGET_X_RATIO'])
        self.ty = int(self.config['H'] * self.config['TARGET_Y_RATIO'])

        # --- 제어 루프 타이머 (원본 방식) ---
        self._send_period = 1.0 / self.config['SEND_HZ']
        self._node.create_timer(0.005, self._tick_control)  # 고속 틱
        self._next_send = time.time()
        self._node.get_logger().info("GazeTracker 기능 초기화 완료 (고속 틱 방식).")

    def face_data_callback(self, face_center):
        """부모 노드에서 새로운 얼굴 데이터가 있을 때 호출되는 공개 메서드입니다."""
        self.have_face = face_center is not None
        if self.have_face:
            self.last_seen = time.time()
            cx, cy = face_center

            self.cx_buf.append(cx)
            self.cy_buf.append(cy)
            if len(self.cx_buf) >= 3:
                cx = int(statistics.median(self.cx_buf))
                cy = int(statistics.median(self.cy_buf))

            err_x = self.tx - cx
            err_y = self.ty - cy

            lp = self.config['ERR_LP']
            self.errx_lp = (1 - lp) * self.errx_lp + lp * err_x
            self.erry_lp = (1 - lp) * self.erry_lp + lp * err_y
        else:
            self.errx_lp *= 0.9
            self.erry_lp *= 0.9

    def _tick_control(self):
        """제어 사이클을 실행하기 위해 타이머에 의해 호출되는 내부 메서드입니다."""
        now = time.time()
        if now < self._next_send:
            return
        self._next_send += self._send_period

        if now - self.t_start < self.config['SAFE_START_SEC']:
            return

        # --- 목표 계산 ---
        dead_band = self.config['DEAD_BAND_PX']
        ex = 0.0 if abs(self.errx_lp) < dead_band else self.errx_lp
        ey = 0.0 if abs(self.erry_lp) < dead_band else self.erry_lp

        if not self.have_face or (time.time() - self.last_seen) > self.config['LOST_HOLD_SEC']:
            ex = 0.0; ey = 0.0

        sign_flip = (ex * self._prev_ex < 0) or (ey * self._prev_ey < 0)
        if sign_flip and (abs(ex) < dead_band * 2 and abs(ey) < dead_band * 2):
            self._zero_hold = self.config['ZERO_HOLD_TICKS']
        if self._zero_hold > 0:
            self._zero_hold -= 1; ex = 0.0; ey = 0.0
        self._prev_ex = ex; self._prev_ey = ey

        vj1_dps = _clamp(ex * self.config['GAIN_X_DPS_PER_PX'], -self.config['MAX_DPS_J1'], self.config['MAX_DPS_J1'])
        vj5_dps = _clamp(ey * self.config['GAIN_Y_DPS_PER_PX'], -self.config['MAX_DPS_J5'], self.config['MAX_DPS_J5'])

        self.lead_target[0] = self.j_last_cmd[0] + (vj1_dps * self.config['LEAD_SEC'])
        self.lead_target[4] = self.j_last_cmd[4] + (vj5_dps * self.config['LEAD_SEC'])

        if abs(self.lead_target[0] - self.j_last_cmd[0]) < self.config['EPS_CMD_DEG'] and \
           abs(self.lead_target[4] - self.j_last_cmd[4]) < self.config['EPS_CMD_DEG']:
            return # 명령 스킵

        final_target_list = self.lead_target[:]
        final_target_list[1:4] = self.j_last_cmd[1:4]
        final_target_list[5] = self.j_last_cmd[5]
        
        # --- 이동 실행 ---
        tgt = self._posj(final_target_list)
        try:
            self._amovej(tgt, vel=self.config['MOVE_VEL'], acc=self.config['MOVE_ACC'], time=self.config['MOVE_TIME'])
            self.j_last_cmd = [tgt.j1, tgt.j2, tgt.j3, tgt.j4, tgt.j5, tgt.j6] if hasattr(tgt, 'j1') else list(tgt[:6])
        except Exception as e:
            self._node.get_logger().error(f"GazeTracker: amovej 실패: {e}")
