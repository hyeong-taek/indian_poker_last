#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import random

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer

from poker_interfaces.action import DetectCardOnce, ObserveGestures


class CameraMock(Node):
    """
    카메라 없는 환경에서 테스트용 액션서버 2개를 제공:
      - /camera/detect_card_once (DetectCardOnce)
      - /camera/observe_gestures (ObserveGestures)

    • DetectCardOnce: rank=1..10 랜덤, conf≈0.75~0.98 랜덤
    • ObserveGestures: nose/arms/ear를 무작위 True/False
    • action result의 필드가 패키지 정의와 다르면(예: conf/ok/message가 없음) 그냥 무시(try/except)
    """

    def __init__(self):
        super().__init__("camera_mock")

        # 액션 서버 2개 등록
        self._srv_detect = ActionServer(
            self,
            DetectCardOnce,
            'camera/detect_card_once',
            execute_callback=self._exec_detect,
        )
        self._srv_tells = ActionServer(
            self,
            ObserveGestures,
            'camera/observe_gestures',
            execute_callback=self._exec_tells,
        )

        self.get_logger().info("CameraMock started with 2 action servers:\n"
                               " - /camera/detect_card_once (DetectCardOnce)\n"
                               " - /camera/observe_gestures (ObserveGestures)")

    # ---- DetectCardOnce ----
    def _exec_detect(self, goal_handle):
        req = goal_handle.request
        timeout_s = float(getattr(req, 'timeout_s', 2.0))
        self.get_logger().info(f"[Detect] goal received: timeout_s={timeout_s:.2f}")

        # 간단히 절반 정도만 대기 (응답 빠르게)
        time.sleep(min(max(timeout_s, 0.0) * 0.5, 2.0))

        if goal_handle.is_cancel_requested:
            self.get_logger().warn("[Detect] goal canceled")
            goal_handle.canceled()
            return DetectCardOnce.Result()

        # 결과 생성
        result = DetectCardOnce.Result()
        # rank 1~10 랜덤
        try:
            result.rank = random.randint(1, 10)
        except Exception:
            pass
        # conf가 있다면 적당히 채움
        try:
            result.conf = random.uniform(0.75, 0.98)
        except Exception:
            pass
        # ok/message 필드가 있다면 채움(없으면 무시)
        try:
            result.ok = True
        except Exception:
            pass
        try:
            result.message = "mock detect ok"
        except Exception:
            pass

        goal_handle.succeed()
        self.get_logger().info(f"[Detect] done: rank={getattr(result, 'rank', '?')}, "
                               f"conf={getattr(result, 'conf', '?')}")
        return result

    # ---- ObserveGestures ----
    def _exec_tells(self, goal_handle):
        req = goal_handle.request
        duration_s = float(getattr(req, 'duration_s', 5.0))
        self.get_logger().info(f"[Tells] goal received: duration_s={duration_s:.2f}")

        # 너무 오래 기다리지 않게 상한 5초
        wait_s = min(max(duration_s, 0.0), 5.0)
        start = time.time()

        # (옵션) 간단한 피드백 루프 — 피드백 타입이 없으면 그냥 무시됨
        while time.time() - start < wait_s:
            if goal_handle.is_cancel_requested:
                self.get_logger().warn("[Tells] goal canceled")
                goal_handle.canceled()
                return ObserveGestures.Result()
            time.sleep(0.2)

        # 결과 생성: nose/arms/ear 랜덤 True/False
        result = ObserveGestures.Result()
        try:
            result.nose = True if random.random() < 0.5 else False
        except Exception:
            pass
        try:
            result.arms = True if random.random() < 0.5 else False
        except Exception:
            pass
        try:
            result.ear = True if random.random() < 0.5 else False
        except Exception:
            pass
        try:
            result.ok = True
        except Exception:
            pass
        try:
            result.message = "mock tells ok"
        except Exception:
            pass

        goal_handle.succeed()
        self.get_logger().info(f"[Tells] done: nose={getattr(result,'nose','?')}, "
                               f"arms={getattr(result,'arms','?')}, "
                               f"ear={getattr(result,'ear','?')}")
        return result


def main():
    rclpy.init()
    node = CameraMock()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
