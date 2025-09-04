#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Doosan M0609 — YuNet Face Tracking (amovej 연속 호출 최적화: 위상고정 + 리드타겟)
- amovej만 계속 호출 (radius=0 한계 수용)
- SEND_HZ로 송신주기 고정, MOVE_TIME은 더 길게 → stop&go 제거
- 현재 각속도 추정값을 LEAD_SEC 만큼 앞으로 적분해 타겟을 항상 '앞'으로 던짐
- 중앙값 필터 + LPF + 데드밴드 + 히스테리시스 + 명령 스킵

ros2 run doosan_gaze_tracking gaze_track_amovej_ultra
"""

import os, time, signal, threading
from collections import deque
import statistics

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from std_msgs.msg import Float32, Float32MultiArray
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy, DurabilityPolicy




# ========= 카메라/표시 =========
W, H    = 640, 480
SHOW    = True
WIN     = "YuNet Face Tracking (amovej phase-lead)"

# ========= 제어 파라미터 (덜컥 최소화) =========
SAFE_START_SEC   = 1.0

# 송신/모션 위상 고정
SEND_HZ       = 15.0                 # amovej 송신 주파수(6~10 권장)

MOVE_TIME     = 0.2              # 세그먼트 계획 시간 (SEND_PERIOD보다 약간 길게)
LEAD_SEC      = 0.4                # 앞으로 던질 시간 (>= MOVE_TIME)

# 추적 타겟(영상 좌표)
TARGET_X_RATIO = 0.50
TARGET_Y_RATIO = 0.40

# 입력 필터/데드밴드
DEAD_BAND_PX   = 17                 # 데드밴드
ERR_LP         = 0.50               # EMA계수(0~1), 높을수록 느리게









# ========= 감지/필터 =========
MIN_FACE_W, MIN_FACE_H = 70, 70
SCORE_TH = 0.85
NMS_TH   = 0.30
TOP_K    = 5000
LOST_HOLD_SEC = 0.5

# ========= 유틸 =========
def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

class YuNetDetector:
    def __init__(self, model_path: str):
        if not model_path or not os.path.exists(model_path):
            raise RuntimeError("YuNet .onnx model not found")
        self.det = cv2.FaceDetectorYN_create(
            model_path, "", (W, H),
            score_threshold=SCORE_TH, nms_threshold=NMS_TH, top_k=TOP_K
        )
    def detect(self, frame_bgr):
        self.det.setInputSize((W, H))
        _, faces = self.det.detect(frame_bgr)
        if faces is None or len(faces)==0:
            return None
        best=None; best_key=-1.0
        for f in faces:
            x,y,w,h = map(int, f[:4])
            score = float(f[14])
            if w<MIN_FACE_W or h<MIN_FACE_H:
                continue
            key = score*(w*h)
            if key>best_key:
                best_key=key; best=(x,y,w,h,score)
        return best

class FaceTrackerPhaseLead(Node):
    def __init__(self, detector: YuNetDetector):
        super().__init__("face_tracker_amovej_phase_lead")

        # 콜백 그룹
        self.cb_image = ReentrantCallbackGroup()
        self.cb_ctrl  = MutuallyExclusiveCallbackGroup()

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')
        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value

        self.detector = detector
        self.bridge   = CvBridge()
        self.tx = int(W*TARGET_X_RATIO)
        self.ty = int(H*TARGET_Y_RATIO)
        self.t_start = time.time()

        # 검출 상태
        self.errx_lp = 0.0
        self.erry_lp = 0.0
        self.last_box = None
        self.last_seen = 0.0
        self.have_face = False

        # 중앙값 필터 버퍼
        self.cx_buf = deque(maxlen=5)
        self.cy_buf = deque(maxlen=5)
        self.agree_pub = False

        # # [FIX - ORDER] bbox publish window 상태값을 타이머 생성 이전에 정의
        # self.pub_hz    = 15.0         # 퍼블리시 주기(Hz)
        # self.pub_until = 0.0          # 이 시각까지 퍼블리시
        # self._last_pub = 0.0          # (필요 시 위상 고정에 사용 가능)
###########
        # 퍼블리시 윈도우 (agree_to_pub로 제어)
        self.pub_until = 0.0          # 이 시각까지 퍼블리시 허용
###########


        # 방향 전환 히스테리시스 상태
        self._prev_ex = 0.0
        self._prev_ey = 0.0
        self._zero_hold = 0

        # UI 버퍼
        self._frame=None; self._hud="No face"; self._bbox=None
        self._lock = threading.Lock()

        # 구독/타이머/UI
        self.subscription = self.create_subscription(
            Image, image_topic, self.image_callback, 10,
            callback_group=self.cb_image
        )
        self.agree_subscription = self.create_subscription(
            Float32, '/dsr01/agree_to_pub', self.get_agree, 10,
        )
        self.loc_pub = self.create_publisher(
            Float32MultiArray, '/dsr01/face_loc', qos,
        )


        if SHOW:
            cv2.setNumThreads(1)
            threading.Thread(target=self.ui_loop, daemon=True).start()

        self._next_send = time.time()
        self.get_logger().info(f"Subscribing: {image_topic} | send={SEND_HZ:.1f}Hz, movetime={MOVE_TIME:.2f}s, lead={LEAD_SEC:.2f}s")
        self.get_logger().info("Press 'q' to quit window.")

    def get_agree(self, msg: Float32):
        # [ADD] msg.data(초) 동안 face_loc 퍼블리시 윈도우 설정
        try:
            sec = float(msg.data)
        except Exception:
            self.get_logger().warn(f"/dsr01/agree_to_pub invalid data: {msg.data}")
            return
        now = time.time()
        if sec <= 0.0:
            self.pub_until = now  # 0 이하 → 즉시 중단
            self.get_logger().info("face_loc publish window cleared (sec<=0).")
        else:
            self.pub_until = now + sec
            self.get_logger().info(f"face_loc will be published for {sec:.3f}s.")

    # def tick_publish(self):
    #     # [ADD] 퍼블리시 윈도우 동안 최신 bbox를 /dsr01/face_loc로 전송
    #     if time.time() >= self.pub_until:
    #         return
    #     # 최신 bbox는 UI 버퍼에 있으므로 락 후 읽기
    #     with self._lock:
    #         bbox = self._bbox
    #     if bbox is None:
    #         return
    #     # bbox=(x,y,w,h,cx,cy,label) → label 제외 6개만 전송
    #     x, y, w, h, cx, cy, _label = bbox
    #     out = Float32MultiArray()
    #     out.data = [float(x), float(y), float(w), float(h), float(cx), float(cy)]
    #     self.loc_pub.publish(out)



    # ---- 이미지 콜백: 중앙값 + LPF ----
    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'img convert fail: {e}')
            return

        if time.time() - self.t_start < SAFE_START_SEC:
            with self._lock:
                self._frame = frame; self._hud="Warmup..."; self._bbox=None
            return

        det = self.detector.detect(frame)
        label="no"
        if det is None:
            if self.last_box and (time.time()-self.last_seen)<=LOST_HOLD_SEC:
                det=self.last_box; label="hold"
            else:
                self.have_face=False; label="no"
        else:
            self.last_box=det; self.last_seen=time.time(); label="face"; self.have_face=True

        hud="No face"; bbox=None
        if det is not None:
            x,y,w,h,score = det
            cx, cy = x + w//2, y + h//2

            # 중앙값 필터
            self.cx_buf.append(cx); self.cy_buf.append(cy)
            if len(self.cx_buf) >= 3:
                cx = int(statistics.median(self.cx_buf))
                cy = int(statistics.median(self.cy_buf))

            err_x = self.tx - cx
            err_y = self.ty - cy

            # LPF
            self.errx_lp = (1-ERR_LP)*self.errx_lp + ERR_LP*err_x
            self.erry_lp = (1-ERR_LP)*self.erry_lp + ERR_LP*err_y

            hud = f"tracking (score {score:.2f})"
            bbox=(x,y,w,h,cx,cy,label)

            # === 이미지 1장당 1회 퍼블리시 (퍼블리시 윈도우 내에서만) ===
            if time.time() < self.pub_until:
                out = Float32MultiArray()
                # label 제외 6개만 전송
                out.data = [float(x), float(y), float(w), float(h), float(cx), float(cy)]
                self.loc_pub.publish(out)
                # time.sleep(0.0333)
                # (선택) 디버그 로그
                # self.get_logger().debug(f"face_loc pub: {out.data}")

        else:
            # 유실 시 서서히 0으로
            self.errx_lp *= 0.9
            self.erry_lp *= 0.9

        if SHOW:
            with self._lock:
                self._frame = frame
                self._hud   = hud
                self._bbox  = bbox

   
    # ---- UI ----
    def ui_loop(self):
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, W, H)
        while rclpy.ok():
            frame=None; hud=""; bbox=None
            with self._lock:
                if self._frame is not None:
                    frame = self._frame.copy()
                hud=self._hud; bbox=self._bbox
            if frame is not None:
                cv2.drawMarker(frame, (self.tx, self.ty), (0,255,0),
                               markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
                cv2.rectangle(frame,
                              (self.tx-DEAD_BAND_PX, self.ty-DEAD_BAND_PX),
                              (self.tx+DEAD_BAND_PX, self.ty+DEAD_BAND_PX),
                              (0,120,0), 1)
                if bbox is not None:
                    x,y,w,h,cx,cy,label=bbox
                    color=(0,255,0) if label=="face" else (0,200,200)
                    cv2.rectangle(frame, (x,y),(x+w,y+h), color, 2)
                    cv2.circle(frame, (cx,cy), 5, color, -1)
                cv2.putText(frame, hud, (10,25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255,255,255), 2, cv2.LINE_AA)
                cv2.putText(frame, hud, (10,25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0,0,0), 1, cv2.LINE_AA)
                cv2.imshow(WIN, frame)
                if (cv2.waitKey(1)&0xFF)==ord('q'):
                    rclpy.shutdown(); break
            time.sleep(0.01)

# ========= 메인 =========
def main():
    def _sigint_handler(_s,_f): raise KeyboardInterrupt()
    signal.signal(signal.SIGINT, _sigint_handler)

    # 모델 경로
    try:
        import ament_index_python.packages
        pkg_share = ament_index_python.packages.get_package_share_directory('doosan_gaze_tracking')
        model_path = os.path.join(pkg_share, "resource", "models", "face_detection_yunet_2023mar.onnx")
    except Exception:
        model_path = os.environ.get("YUNET_ONNX","")

    if not model_path or not os.path.exists(model_path):
        print(f"[FATAL] YuNet model not found at: {model_path}")
        return

    detector = YuNetDetector(model_path)

    # ROS/DSR 초기화 (순서 중요)
    rclpy.init()





    app = FaceTrackerPhaseLead(detector)
    exe = MultiThreadedExecutor(num_threads=1)
    exe.add_node(app)

    try:
        exe.spin()
    except KeyboardInterrupt:
        app.get_logger().info("Shutting down...")
    finally:
        if SHOW: cv2.destroyAllWindows()
        exe.shutdown()
        app.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
