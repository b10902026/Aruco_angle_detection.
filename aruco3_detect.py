"""用 ArUco Marker 追蹤轉盤角度。

ID 0、1、2：Marker 平貼在轉盤頂面，使用影像平面方向。
ID 3：Marker 貼在轉盤側面，使用 PnP 估計 3D 姿態與 Marker 法向量偏航角。
"""

import math

import cv2
import numpy as np

CAMERA_INDEX = 0
WINDOW_NAME = "ArUco Multi-Mode Rotation Tracker"
TOP_IDS = (0, 1, 2)
SIDE_IDS = (3,)
TARGET_IDS = TOP_IDS + SIDE_IDS

# 側面 Marker 的黑色方形邊長；目前印製規格是 50 mm。
MARKER_SIZE_M = 0.05

# 未校正相機的初始內參估算。精確角度建議先做相機校正，再填入下方矩陣。
CAMERA_HORIZONTAL_FOV_DEG = 70.0
CAMERA_MATRIX = None
DISTORTION_COEFFICIENTS = None


def create_detector():
    """支援新版與舊版 OpenCV ArUco API。"""
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    if hasattr(aruco, "DetectorParameters"):
        parameters = aruco.DetectorParameters()
    else:
        parameters = aruco.DetectorParameters_create()

    parameters.minMarkerPerimeterRate = 0.01
    if hasattr(parameters, "cornerRefinementMethod") and hasattr(aruco, "CORNER_REFINE_SUBPIX"):
        parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(dictionary, parameters)
        return detector.detectMarkers
    return lambda gray: aruco.detectMarkers(gray, dictionary, parameters=parameters)


def wrap_degrees(angle):
    """將角度差折回 [-180, 180)。"""
    return (angle + 180.0) % 360.0 - 180.0


def top_marker_angle(corners):
    """頂面模式：以角點 0 到 1 的方向計算平面角度。"""
    p0, p1 = corners[0], corners[1]
    return math.degrees(math.atan2(-float(p1[1] - p0[1]), float(p1[0] - p0[0])))


def get_camera_parameters(frame_shape):
    """回傳相機內參；未提供校正值時，以水平視角估算。"""
    height, width = frame_shape[:2]
    if CAMERA_MATRIX is not None:
        camera_matrix = np.asarray(CAMERA_MATRIX, dtype=np.float64)
    else:
        focal_length = width / (2.0 * math.tan(math.radians(CAMERA_HORIZONTAL_FOV_DEG) / 2.0))
        camera_matrix = np.array([
            [focal_length, 0.0, width / 2.0],
            [0.0, focal_length, height / 2.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

    if DISTORTION_COEFFICIENTS is not None:
        distortion = np.asarray(DISTORTION_COEFFICIENTS, dtype=np.float64)
    else:
        distortion = np.zeros((5, 1), dtype=np.float64)
    return camera_matrix, distortion


def side_marker_yaw(corners, frame_shape):
    """以方形 Marker 的 3D 姿態，計算其法向量在相機水平面的偏航角。"""
    half_size = MARKER_SIZE_M / 2.0
    # 順序對應 OpenCV ArUco 角點：左上、右上、右下、左下。
    object_points = np.array([
        [-half_size, half_size, 0.0],
        [half_size, half_size, 0.0],
        [half_size, -half_size, 0.0],
        [-half_size, -half_size, 0.0],
    ], dtype=np.float32)
    image_points = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    camera_matrix, distortion = get_camera_parameters(frame_shape)

    flag = getattr(cv2, "SOLVEPNP_IPPE_SQUARE", cv2.SOLVEPNP_ITERATIVE)
    ok, rotation_vector, _ = cv2.solvePnP(
        object_points, image_points, camera_matrix, distortion, flags=flag
    )
    if not ok:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    # Marker 的 +Z 軸是面朝方向；繞直立軸旋轉時，法向量在相機 X-Z 平面轉動。
    normal = rotation_matrix[:, 2]
    return math.degrees(math.atan2(float(normal[0]), float(normal[2])))


def draw_label(frame, text, position, color, scale=0.65, thickness=1):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("無法開啟攝影機。請確認 macOS「系統設定 → 隱私權與安全性 → 相機」已允許目前的 Python／IDE。")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    try:
        detect = create_detector()
    except AttributeError:
        cap.release()
        print("請在此專案的 Python 環境安裝 opencv-contrib-python。")
        return

    tracking_id = None
    mode = None
    previous_angle = None
    initial_angle = None
    current_angle = None
    total_rotation = 0.0
    was_lost = False
    last_direction = 0  # +1 正轉、-1 反轉；供 ID 3 遮擋後估算盲區方向

    print("追蹤已啟動：頂面模式 ID 0、1、2；側面 3D 姿態模式 ID 3。按 R 歸零，Q 或 ESC 結束。")
    if CAMERA_MATRIX is None:
        print(f"提示：側面角度以 {CAMERA_HORIZONTAL_FOV_DEG:.0f}° 視角估算內參；相機校正後可提高準確度。")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("攝影機已開啟，但讀取影格失敗。")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detect(gray)
            detected = {}
            if ids is not None:
                for marker_id, marker_corners in zip(ids.flatten(), corners):
                    detected[int(marker_id)] = marker_corners.reshape(4, 2)
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            if tracking_id is None:
                available = sorted(set(detected).intersection(TARGET_IDS))
                if available:
                    tracking_id = available[0]
                    mode = "SIDE" if tracking_id in SIDE_IDS else "TOP"

            marker_visible = tracking_id is not None and tracking_id in detected
            angle_now = None
            pose_ok = False
            if marker_visible:
                points = detected[tracking_id]
                if mode == "TOP":
                    angle_now = top_marker_angle(points)
                    pose_ok = True
                else:
                    angle_now = side_marker_yaw(points, frame.shape)
                    pose_ok = angle_now is not None

            if marker_visible and pose_ok:
                points = detected[tracking_id]
                center = tuple(points.mean(axis=0).round().astype(int))
                cv2.circle(frame, center, 5, (0, 255, 0), -1)

                if initial_angle is None:
                    initial_angle = angle_now
                    total_rotation = 0.0
                gap_delta = None
                gap_direction_was_known = last_direction != 0
                if previous_angle is not None:
                    gap_delta = wrap_degrees(angle_now - previous_angle)
                    if was_lost and mode == "SIDE" and last_direction != 0:
                        # 重新出現時，選擇與消失前轉向一致的角度分支。
                        # 例如 +80° 消失、-80° 重現：若原本正轉，盲區估算為 +200°，
                        # 而不是最短差 -160°。這假設盲區內沒有反轉，且未多轉完整圈。
                        if last_direction > 0 and gap_delta < 0:
                            gap_delta += 360.0
                        elif last_direction < 0 and gap_delta > 0:
                            gap_delta -= 360.0
                    total_rotation += gap_delta
                    if abs(gap_delta) >= 0.2:
                        last_direction = 1 if gap_delta > 0 else -1

                previous_angle = angle_now
                current_angle = wrap_degrees(angle_now - initial_angle)

                if was_lost:
                    if mode == "SIDE" and gap_delta is not None and gap_direction_was_known:
                        status = f"BLIND GAP ESTIMATE: {gap_delta:+.1f} DEG (SAME DIRECTION)"
                    else:
                        status = f"REACQUIRED ID {tracking_id}; GAP DIRECTION UNKNOWN"
                    status_color = (0, 210, 255)
                    was_lost = False
                else:
                    status = f"TRACKING ID {tracking_id} ({mode} MODE)"
                    status_color = (0, 255, 0)
            elif marker_visible:
                status = "ID 3 FOUND, BUT POSE ESTIMATION FAILED"
                status_color = (0, 165, 255)
            else:
                if tracking_id is not None:
                    was_lost = True
                    status = f"ID {tracking_id} LOST - ROTATION DURING GAP MAY BE AMBIGUOUS"
                else:
                    other_ids = sorted(detected)
                    status = "WRONG ID: " + ", ".join(map(str, other_ids)) if other_ids else "NO MARKER DETECTED"
                status_color = (0, 165, 255)

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 192), (18, 22, 28), -1)
            cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

            label_id = tracking_id if tracking_id is not None else "None"
            draw_label(frame, f"Mode: {mode or 'Searching'}  |  Target ID: {label_id}",
                       (18, 28), (235, 235, 235), 0.56)
            draw_label(frame, status, (18, 59), status_color, 0.57, 2)
            draw_label(frame,
                       f"Angle from start: {current_angle:+.1f} deg" if current_angle is not None else "Angle from start: --",
                       (18, 96), (0, 255, 0), 0.72, 2)
            draw_label(frame, f"Accumulated rotation: {total_rotation:+.1f} deg",
                       (18, 132), (0, 190, 255), 0.68, 2)
            if mode == "SIDE":
                draw_label(frame, f"Marker size: {MARKER_SIZE_M * 1000:.0f} mm | HFOV estimate: {CAMERA_HORIZONTAL_FOV_DEG:.0f} deg",
                           (18, 161), (210, 210, 210), 0.48)
            draw_label(frame, "R: reset angle    Q / ESC: quit",
                       (18, 184), (220, 220, 220), 0.45)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                total_rotation = 0.0
                if marker_visible and pose_ok:
                    initial_angle = angle_now
                    previous_angle = angle_now
                    current_angle = 0.0
                else:
                    initial_angle = None
                    previous_angle = None
                    current_angle = None
                was_lost = False
                last_direction = 0
                print("角度已歸零。")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
