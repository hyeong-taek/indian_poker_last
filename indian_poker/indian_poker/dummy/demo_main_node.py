#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GazeTracker 기능을 통합하여 사용하는 데모용 메인 제어 노드.

실행 방법:
1. 터미널 1: ros2 run <패키지명> tracking_eye_part.py (팀원 노드)
2. 터미널 2: ros2 run <패키지명> demo_main_node.py
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
import signal

# geometry_msgs.msg.PointStamped 대신 Float32MultiArray 사용
from std_msgs.msg import Float32MultiArray

# DSR 관련 임포트
import DR_init

# 재사용 가능한 기능 클래스 임포트
from test_test.gaze_tracking_control import GazeTracker
from rclpy.qos import qos_profile_sensor_data

# 로봇 설정
ROBOT_ID    = "dsr01"
ROBOT_MODEL = "m0609"

class DemoMainNode(Node):
    """GazeTracker와 같은 기능 클래스를 통합하여 사용하는 메인 노드 예시"""
    def __init__(self, amovej_func, posj_func, get_current_posj_func):
        super().__init__('demo_main_node')
        self.get_logger().info("데모 메인 노드를 시작합니다.")

        self.gaze_tracker = None
        try:
            # GazeTracker 기능 클래스를 초기화하고 인스턴스화합니다.
            self.gaze_tracker = GazeTracker(
                node=self, # 로거, 타이머 등을 위해 자신의 노드 인스턴스를 전달
                amovej_func=amovej_func,
                posj_func=posj_func,
                get_current_posj_func=get_current_posj_func
            )
            self.get_logger().info("GazeTracker 기능이 성공적으로 초기화되었습니다.")

        except Exception as e:
            self.get_logger().fatal(f"GazeTracker 초기화에 실패했습니다: {e}")
            return

        # 얼굴 검출 노드(/dsr01/face_loc)로부터 데이터를 수신하기 위한 구독자
        # 토픽 이름과 메시지 타입 변경
        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/dsr01/face_loc',
            self.face_data_receiver,
            qos_profile_sensor_data
        )
        self.get_logger().info("'/dsr01/face_loc' 토픽을 구독합니다.")

    def face_data_receiver(self, msg: Float32MultiArray):
        """/dsr01/face_loc 토픽에서 메시지를 수신할 때마다 호출되는 콜백 함수"""
        if self.gaze_tracker:
            # Float32MultiArray에서 cx, cy 추출 (인덱스 4, 5)
            if len(msg.data) >= 6:
                cx = msg.data[4]
                cy = msg.data[5]
                # 수신한 얼굴 중심점 좌표를 GazeTracker에 전달합니다.
                self.gaze_tracker.face_data_callback((cx, cy))
            else:
                self.get_logger().warn("수신된 Float32MultiArray 메시지의 데이터 길이가 부족합니다.")

def main(args=None):
    # ROS 클라이언트 라이브러리 초기화
    rclpy.init(args=args)

    # DSR 로봇 브릿지 노드 초기화
    dsr_bridge_node = rclpy.create_node("dsr_bridge_node", namespace=ROBOT_ID)
    DR_init.__dsr__node  = dsr_bridge_node
    DR_init.__dsr__id    = ROBOT_ID
    DR_init.__dsr__model = ROBOT_MODEL

    # DSR 로봇 제어 함수 임포트
    try:
        from DSR_ROBOT2 import get_current_posj, amovej
        from DR_common2 import posj
    except ImportError as e:
        dsr_bridge_node.get_logger().fatal(f"DSR 라이브러리 임포트에 실패했습니다: {e}")
        rclpy.shutdown()
        return

    # 데모 메인 노드 인스턴스 생성 및 DSR 함수 전달
    demo_node = DemoMainNode(amovej, posj, get_current_posj)

    # 여러 노드를 함께 실행하기 위한 Multi-threaded executor 생성
    executor = MultiThreadedExecutor()
    executor.add_node(dsr_bridge_node)
    executor.add_node(demo_node)

    # 인터럽트(Ctrl+C) 처리를 위한 핸들러
    def signal_handler(sig, frame):
        print('\nCtrl+C 감지! 노드를 종료합니다.')
        demo_node.destroy_node()
        dsr_bridge_node.destroy_node()
        rclpy.shutdown()

    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Executor 실행 (노드들의 콜백 처리 시작)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        # 확실한 종료 처리
        if rclpy.ok():
            demo_node.destroy_node()
            dsr_bridge_node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()
