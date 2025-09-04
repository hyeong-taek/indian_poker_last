#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from std_msgs.msg import String
import json
import threading
import time
from poker_interfaces.action import MoveNamedPose, PickCard, ShowCard, BetChips

ROBOT_ID = "dsr01"

class PokerRobotInterface(Node):
    def __init__(self):
        super().__init__('poker_robot_interface', namespace=ROBOT_ID)
        
        self.pub_command = self.create_publisher(String, 'dsr_command', 10)
        self.sub_result = self.create_subscription(String, 'dsr_result', self.on_result, 10)

        self.result_event = threading.Event()
        self.last_result = {}
        self.action_lock = threading.Lock() # 한 번에 하나의 액션만 처리하도록 보장

        # 4개의 모든 액션 서버를 생성합니다.
        cb_group = ReentrantCallbackGroup()
        self.as_move = ActionServer(self, MoveNamedPose, 'robot/move_named_pose', self._exec_move_named_pose, callback_group=cb_group)
        self.as_pick = ActionServer(self, PickCard, 'robot/pick_card', self._exec_pick_card, callback_group=cb_group)
        self.as_show = ActionServer(self, ShowCard, 'robot/show_card', self._exec_show_card, callback_group=cb_group)
        self.as_bet = ActionServer(self, BetChips, 'robot/bet_chips', self._exec_bet_chips, callback_group=cb_group)
        
        self.get_logger().info("Poker Robot Interface is ready with all 4 actions.")

    def on_result(self, msg):
        self.last_result = json.loads(msg.data)
        self.result_event.set()

    def send_command_and_wait(self, command, params, timeout=15.0):
        if not self.action_lock.acquire(timeout=2.0):
             return {"status": "error", "message": "Could not acquire action lock, another action is in progress."}

        try:
            self.result_event.clear()
            cmd_msg = {"command": command, "params": params}
            self.pub_command.publish(String(data=json.dumps(cmd_msg)))
            
            event_triggered = self.result_event.wait(timeout=timeout)
            if not event_triggered:
                return {"status": "error", "message": f"Timeout waiting for result of command '{command}'"}
            
            return self.last_result
        finally:
            self.action_lock.release()

    def _exec_move_named_pose(self, goal_handle):
        req = goal_handle.request
        self.get_logger().info(f"Executing MoveNamedPose: {req.name}")
        params = {"name": req.name, "velocity": req.velocity, "acc": req.acc}
        result_data = self.send_command_and_wait("move_named_pose", params)
        
        res = MoveNamedPose.Result()
        if result_data.get("status") == "success":
            res.ok = True; res.message = f"moved to {req.name}"
            goal_handle.succeed()
        else:
            res.ok = False; res.message = result_data.get("message", "Unknown error in driver")
            goal_handle.abort()
        return res

    def _exec_pick_card(self, goal_handle):
        req = goal_handle.request
        self.get_logger().info(f"Executing PickCard")
        params = {"approach_z": req.approach_z, "grip_width": req.grip_width}
        result_data = self.send_command_and_wait("pick_card", params)

        res = PickCard.Result()
        if result_data.get("status") == "success":
            res.ok = True; res.message = "card picked"
            goal_handle.succeed()
        else:
            res.ok = False; res.message = result_data.get("message", "Unknown error in driver")
            goal_handle.abort()
        return res

    # --- 추가된 콜백 함수들 ---
    def _exec_show_card(self, goal_handle):
        req = goal_handle.request
        self.get_logger().info(f"Executing ShowCard for {req.timeout_s} seconds")
        params = {"timeout_s": req.timeout_s}
        result_data = self.send_command_and_wait("show_card", params)

        res = ShowCard.Result()
        if result_data.get("status") == "success":
            res.ok = True; res.message = "card shown"
            goal_handle.succeed()
        else:
            res.ok = False; res.message = result_data.get("message", "Unknown error in driver")
            goal_handle.abort()
        return res

    def _exec_bet_chips(self, goal_handle):
        req = goal_handle.request
        self.get_logger().info(f"Executing BetChips with amount: {req.amount}")
        params = {"amount": req.amount}
        result_data = self.send_command_and_wait("bet_chips", params)

        res = BetChips.Result()
        if result_data.get("status") == "success":
            res.ok = True; res.message = f"bet {req.amount} chips"
            goal_handle.succeed()
        else:
            res.ok = False; res.message = result_data.get("message", "Unknown error in driver")
            goal_handle.abort()
        return res

def main():
    rclpy.init()
    node = PokerRobotInterface()
    executor = MultiThreadedExecutor(num_threads=4) # 여러 액션을 동시에 수락하기 위해 MultiThreadedExecutor 사용
    executor.add_node(node)
    executor.spin()
    rclpy.shutdown()

if __name__ == '__main__':
    main()