"""多個側面 ArUco Marker 的轉盤角度追蹤器。

頂面：ID 0、1、2，依單一 Marker 的平面方向追蹤。
側面：預設 ID 3–8，沿轉盤圓周每 60 度貼一個不同 ID；由各 Marker 的 3D 法向量
      扣除其固定圓周位置，再融合可見 Marker 的角度，讓 Marker 交接時保持連續。
"""

import math

import cv2
import numpy as np

CAMERA_INDEX = 0
WINDOW_NAME = "ArUco Multi-Marker Rotation Tracker"
TOP_IDS = (0, 1, 2)
SIDE_FIRST_ID = 3
SIDE_MARKER_COUNT = 6
SIDE_MARKER_SPACING_DEG = 360.0 / SIDE_MARKER_COUNT
# 貼 Marker 時，ID 3、4、5... 必須沿轉盤圓周同一方向依序排列。
# 如果 ID 增加方向與程式的正角度方向相反，改成 -1.0。
SIDE_ID_ORDER_SIGN = 1.0
SIDE_IDS = tuple(range(SIDE_FIRST_ID, SIDE_FIRST_ID + SIDE_MARKER_COUNT))
TARGET_IDS = TOP_IDS + SIDE_IDS

MARKER_SIZE_M = 0.05
CAMERA_HORIZONTAL_FOV_DEG = 70.0
CAMERA_MATRIX = None
DISTORTION_COEFFICIENTS = None


def create_detector():
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
    return (angle + 180.0) % 360.0 - 180.0


def top_marker_angle(corners):
    p0, p1 = corners[0], corners[1]
    return math.degrees(math.atan2(-float(p1[1] - p0[1]), float(p1[0] - p0[0])))


def camera_parameters(frame_shape):
    height, width = frame_shape[:2]
    if CAMERA_MATRIX is not None:
        matrix = np.asarray(CAMERA_MATRIX, dtype=np.float64)
    else:
        focal = width / (2.0 * math.tan(math.radians(CAMERA_HORIZONTAL_FOV_DEG) / 2.0))
        matrix = np.array([[focal, 0.0, width / 2.0],
                           [0.0, focal, height / 2.0],
                           [0.0, 0.0, 1.0]], dtype=np.float64)
    distortion = (np.asarray(DISTORTION_COEFFICIENTS, dtype=np.float64)
                  if DISTORTION_COEFFICIENTS is not None
                  else np.zeros((5, 1), dtype=np.float64))
    return matrix, distortion


def marker_yaw(corners, frame_shape):
    """估計 Marker 面法向量在相機水平面內的方向角。"""
    half = MARKER_SIZE_M / 2.0
    object_points = np.array([
        [-half, half, 0.0], [half, half, 0.0],
        [half, -half, 0.0], [-half, -half, 0.0],
    ], dtype=np.float32)
    image_points = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    matrix, distortion = camera_parameters(frame_shape)
    flag = getattr(cv2, "SOLVEPNP_IPPE_SQUARE", cv2.SOLVEPNP_ITERATIVE)
    ok, rvec, _ = cv2.solvePnP(object_points, image_points, matrix, distortion, flags=flag)
    if not ok:
        return None
    rotation, _ = cv2.Rodrigues(rvec)
    normal = rotation[:, 2]
    return math.degrees(math.atan2(float(normal[0]), float(normal[2])))


def circular_mean_degrees(angles):
    """用圓形平均融合角度，避免 +179° 和 -179° 平均成 0°。"""
    radians = np.radians(np.asarray(angles, dtype=np.float64))
    sin_mean = float(np.mean(np.sin(radians)))
    cos_mean = float(np.mean(np.cos(radians)))
    if math.hypot(sin_mean, cos_mean) < 1e-8:
        return float(angles[0])
    return math.degrees(math.atan2(sin_mean, cos_mean))


def draw_label(frame, text, position, color, scale=0.62, thickness=1):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("無法開啟攝影機。請確認 macOS 相機權限。")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    try:
        detect = create_detector()
    except AttributeError:
        cap.release()
        print("請在此專案的 Python 環境安裝 opencv-contrib-python。")
        return

    mode = None
    tracking_top_id = None
    previous_angle = None
    initial_angle = None
    current_angle = None
    total_rotation = 0.0
    last_direction = 0
    was_lost = False

    print(f"多 Marker 追蹤已啟動：頂面 ID 0–2；側面 ID {SIDE_FIRST_ID}–{SIDE_IDS[-1]}，每 {SIDE_MARKER_SPACING_DEG:.0f}° 一個。R 歸零，Q/ESC 結束。")
    if CAMERA_MATRIX is None:
        print(f"側面姿態使用 {CAMERA_HORIZONTAL_FOV_DEG:.0f}° 水平視角估算內參；相機校正可提高準確度。")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("攝影機已開啟，但讀取影格失敗。")
                break

            corners, ids, _ = detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            detections = []
            if ids is not None:
                for marker_id, marker_corners in zip(ids.flatten(), corners):
                    detections.append((int(marker_id), marker_corners.reshape(4, 2)))
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            visible_ids = sorted({mid for mid, _ in detections})
            if mode is None:
                visible_top = sorted(set(visible_ids).intersection(TOP_IDS))
                visible_side = sorted(set(visible_ids).intersection(SIDE_IDS))
                if visible_top:
                    mode = "TOP"
                    tracking_top_id = visible_top[0]
                elif visible_side:
                    mode = "SIDE"

            angle_now = None
            marker_count = 0
            if mode == "TOP":
                top_points = next((pts for mid, pts in detections if mid == tracking_top_id), None)
                if top_points is not None:
                    angle_now = top_marker_angle(top_points)
                    marker_count = 1
            elif mode == "SIDE":
                corrected_angles = []
                for marker_id, points in detections:
                    if marker_id not in SIDE_IDS:
                        continue
                    yaw = marker_yaw(points, frame.shape)
                    if yaw is None:
                        continue
                    index = marker_id - SIDE_FIRST_ID
                    fixed_offset = SIDE_ID_ORDER_SIGN * index * SIDE_MARKER_SPACING_DEG
                    # 將每個 Marker 的固定圓周位置扣除，換算為同一個轉盤角度。
                    corrected_angles.append(wrap_degrees(yaw - fixed_offset))
                    marker_count += 1
                if corrected_angles:
                    angle_now = circular_mean_degrees(corrected_angles)

            marker_visible = angle_now is not None
            status_color = (0, 255, 0)
            gap_delta = None
            direction_was_known = last_direction != 0

            if marker_visible:
                if initial_angle is None:
                    initial_angle = angle_now
                    total_rotation = 0.0
                if previous_angle is not None:
                    gap_delta = wrap_degrees(angle_now - previous_angle)
                    if was_lost and mode == "SIDE" and last_direction != 0:
                        # 遮擋後沿用遮擋前的轉向，選 +/−360° 中相符的分支。
                        # 多 Marker 理應縮短或消除盲區；若仍全數消失，這仍是估算值。
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
                    if mode == "SIDE" and gap_delta is not None and direction_was_known:
                        status = f"SIDE MARKERS REACQUIRED; GAP ESTIMATE {gap_delta:+.1f} DEG"
                        status_color = (0, 210, 255)
                    else:
                        status = "MARKERS REACQUIRED; GAP ROTATION MAY BE AMBIGUOUS"
                        status_color = (0, 210, 255)
                    was_lost = False
                elif mode == "SIDE":
                    status = f"SIDE RING TRACKING: {marker_count}/{SIDE_MARKER_COUNT} MARKERS VISIBLE"
                else:
                    status = f"TRACKING TOP ID {tracking_top_id}"
            else:
                if mode is not None:
                    was_lost = True
                    status = "ALL TARGET MARKERS LOST - ROTATION GAP MAY BE AMBIGUOUS"
                else:
                    other_ids = ", ".join(map(str, visible_ids)) if visible_ids else "none"
                    status = f"SEARCHING FOR TARGET IDS; SEEN: {other_ids}"
                status_color = (0, 165, 255)

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 205), (18, 22, 28), -1)
            cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
            mode_name = mode or "Searching"
            target = f"ID {tracking_top_id}" if mode == "TOP" else (f"IDs {SIDE_FIRST_ID}-{SIDE_IDS[-1]}" if mode == "SIDE" else "--")
            draw_label(frame, f"Mode: {mode_name} | Target: {target} | Dict: 4X4_50",
                       (18, 27), (235, 235, 235), 0.55)
            draw_label(frame, status, (18, 58), status_color, 0.55, 2)
            draw_label(frame,
                       f"Angle from start: {current_angle:+.1f} deg" if current_angle is not None else "Angle from start: --",
                       (18, 95), (0, 255, 0), 0.70, 2)
            draw_label(frame, f"Accumulated rotation: {total_rotation:+.1f} deg",
                       (18, 130), (0, 190, 255), 0.68, 2)
            if mode == "SIDE":
                shown = ",".join(str(mid) for mid in visible_ids if mid in SIDE_IDS) or "none"
                draw_label(frame, f"Visible side IDs: {shown}", (18, 159), (220, 220, 220), 0.48)
            draw_label(frame, "R: reset angle    Q / ESC: quit",
                       (18, 190), (220, 220, 220), 0.45)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                total_rotation = 0.0
                if marker_visible:
                    initial_angle = angle_now
                    previous_angle = angle_now
                    current_angle = 0.0
                else:
                    initial_angle = None
                    previous_angle = None
                    current_angle = None
                last_direction = 0
                was_lost = False
                print("角度已歸零。")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
