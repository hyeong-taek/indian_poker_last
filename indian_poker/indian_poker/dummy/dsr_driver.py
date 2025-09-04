#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String
import json
import traceback
import sys
import threading
import time

import DR_init
from indian_poker.onrobot import RG

ROBOT_ID = "dsr01"; ROBOT_MODEL = "m0609"
DR_init.__dsr__id = ROBOT_ID; DR_init.__dsr__model = ROBOT_MODEL
GRIPPER_NAME = "rg2"; TOOLCHARGER_IP = "192.168.1.1"; TOOLCHARGER_PORT = "502"
VELOCITY, ACC = 70, 50

class DsrDriver(Node):
    def __init__(self):
        super().__init__('dsr_driver', namespace=ROBOT_ID)
        self.pub_result = self.create_publisher(String, 'dsr_result', 10)
        self.sub_command = self.create_subscription(String, 'dsr_command', self.on_command, 10)
        self.command_lock = threading.Lock()
        
        # DSR 라이브러리 및 로봇 초기화는 main 함수에서 진행
        self.get_logger().info("DSR Driver node created.")

    def on_command(self, msg):
        # 명령 수신 시, 작업자 스레드에 업무 위임
        if not self.command_lock.acquire(blocking=False):
            self.get_logger().warn(f"Already processing. Dropping command: {msg.data}")
            return
        
        worker_thread = threading.Thread(target=self.command_worker, args=(msg.data,))
        worker_thread.daemon = True
        worker_thread.start()

    def command_worker(self, msg_data):
        # DSR 함수들은 main에서 import 되어 전역처럼 사용됩니다.
        from DSR_ROBOT2 import movej, movel, wait
        from DR_common2 import posx
        
        cmd = None
        try:
            data = json.loads(msg_data)
            cmd = data.get("command")
            params = data.get("params", {})
            self.get_logger().info(f"Executing command: {cmd}")

            if cmd == "move_named_pose":
                name = params['name']; kind, pose = self.named_pose_map[name]
                vel = params.get('velocity', VELOCITY); acc = params.get('acc', ACC)
                if kind == 'x': movel(pose, vel, acc)
                else: movej(pose, vel, acc)
            
            elif cmd == "pick_card":
                dz = params.get('approach_z', 20.0); width = params.get('grip_width', 600.0)
                movel(posx(self.card_grip_pos[0], self.card_grip_pos[1], self.card_grip_pos[2] - dz, self.card_grip_pos[3], self.card_grip_pos[4], self.card_grip_pos[5]), 50, 30)
                self.gripper.move_gripper(int(width))
                wait(1.0)
                movel(posx(self.card_grip_pos[0], self.card_grip_pos[1], self.card_grip_pos[2] + 100, self.card_grip_pos[3], self.card_grip_pos[4], self.card_grip_pos[5]), 50, 30)
            
            elif cmd == "show_card":
                wait(params.get('timeout_s', 3.0))

            elif cmd == "bet_chips":
                amount = params.get('amount', 1)
                movej(self.bet_pose, VELOCITY, ACC)
                self.gripper.move_gripper(400 + 50 * amount)
                wait(1.0)
                self.gripper.open_gripper()
                wait(1.0)
                movej(self.home, VELOCITY, ACC)

            result = {"status": "success", "command": cmd}
            self.pub_result.publish(String(data=json.dumps(result)))
            self.get_logger().info(f"Command '{cmd}' succeeded.")
        except Exception as e:
            self.get_logger().error(f"Command '{cmd}' failed: {e}\n{traceback.format_exc()}")
            result = {"status": "error", "command": cmd, "message": str(e)}
            self.pub_result.publish(String(data=json.dumps(result)))
        finally:
            self.command_lock.release()

def main():
    rclpy.init()
    node = DsrDriver()
    DR_init.__dsr__node = node
    
    try:
        from DSR_ROBOT2 import movej, movel, wait, set_robot_mode, set_tool, set_tcp, ROBOT_MODE_MANUAL, ROBOT_MODE_AUTONOMOUS
        from DR_common2 import posx, posj
        
        node.get_logger().info("Configuring robot...")
        node.home = posj([0.0, 0.0, 90.0, 0.0, 90.0, 0.0])
        node.card_grip_pos = posx([290.23, -221.55, -0.32 + 20.0, 101.91, 179.98, -167.75])
        node.card_detect_pos = posj([-3.53180909, -22.98944855, 112.55172729, 7.8614459, 0.45518515, 81.9596405])
        node.bluff_detect_pos= posj([-7.51381588, -74.87239075, 123.9002533, 11.15486908, 47.28850937, 80.92360687])
        node.bet_pose = node.home
        node.named_pose_map = {
            "home": ('j', node.home), "card_grip": ('x', node.card_grip_pos),
            "card_detect": ('j', node.card_detect_pos), "card_show": ('j', node.card_detect_pos),
            "gesture_view": ('j', node.bluff_detect_pos), "bet": ('j', node.bet_pose),
        }
        node.gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)
        
        set_robot_mode(ROBOT_MODE_MANUAL); set_tool("Tool Weight"); set_tcp("rg2"); set_robot_mode(ROBOT_MODE_AUTONOMOUS)
        movej(node.home, vel=70, acc=50)
        node.gripper.open_gripper()
        node.get_logger().info("Robot and Gripper initialized successfully.")
    except Exception as e:
        node.get_logger().fatal(f"CRITICAL ERROR during initialization: {e}\n{traceback.format_exc()}"); node.destroy_node(); rclpy.shutdown(); sys.exit(1)

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        node.get_logger().info("DSR Driver is spinning...")
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()