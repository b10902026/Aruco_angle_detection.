"""用單一 ArUco Marker 即時追蹤平面旋轉角度。

字典：DICT_4X4_50；接受 ID 0、1、2。角度以程式啟動後首次看到 Marker 的方向為 0°。
"""

import math

import cv2

CAMERA_INDEX = 0
WINDOW_NAME = "ArUco Single Marker Rotation Tracker"
TARGET_IDS = (0, 1, 2)


def create_detector():
    """支援新版與舊版 OpenCV ArUco API。"""
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

    if hasattr(aruco, "DetectorParameters"):
        parameters = aruco.DetectorParameters()
    else:
        parameters = aruco.DetectorParameters_create()

    # 讓較小、較遠的 Marker 也有機會被偵測。
    parameters.minMarkerPerimeterRate = 0.01
    if hasattr(parameters, "cornerRefinementMethod") and hasattr(aruco, "CORNER_REFINE_SUBPIX"):
        parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(dictionary, parameters)
        return detector.detectMarkers

    return lambda gray: aruco.detectMarkers(gray, dictionary, parameters=parameters)


def wrap_degrees(angle):
    """將角度差折回 [-180, 180)，避免跨越 180° 時跳變。"""
    return (angle + 180.0) % 360.0 - 180.0


def marker_angle(corners):
    """用 Marker 角點 0 到 1 的上邊方向計算影像平面角度。"""
    p0, p1 = corners[0], corners[1]
    dx = float(p1[0] - p0[0])
    dy = float(p1[1] - p0[1])
    return math.degrees(math.atan2(-dy, dx))


def draw_label(frame, text, position, color, scale=0.65, thickness=1):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def main():
    # macOS 優先使用 AVFoundation；失敗時再使用 OpenCV 預設後端。
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
        print("目前的 OpenCV 沒有 ArUco 模組。請在此專案的 Python 環境安裝 opencv-contrib-python。")
        return

    # 第一個辨識到的目標 ID 會被鎖定，避免畫面中途出現其他 Marker 時換標追蹤。
    tracking_id = None
    previous_angle = None
    initial_angle = None
    current_angle = None
    total_rotation = 0.0
    print("單一 Marker 旋轉追蹤已啟動：DICT_4X4_50，接受 ID 0、1、2。按 R 歸零；按 Q 或 ESC 結束。")

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

            # 初次出現時選一個目標 ID，之後持續追蹤相同 ID。
            if tracking_id is None:
                available = sorted(set(detected).intersection(TARGET_IDS))
                if available:
                    tracking_id = available[0]

            marker_visible = tracking_id is not None and tracking_id in detected
            if marker_visible:
                points = detected[tracking_id]
                angle_now = marker_angle(points)
                p0 = tuple(points[0].round().astype(int))
                p1 = tuple(points[1].round().astype(int))
                center = tuple(points.mean(axis=0).round().astype(int))
                cv2.line(frame, p0, p1, (0, 255, 0), 3, cv2.LINE_AA)
                draw_label(frame, f"ID {tracking_id}", (center[0] - 28, center[1]),
                           (0, 255, 0), 0.7, 2)

                if initial_angle is None:
                    initial_angle = angle_now
                    total_rotation = 0.0
                if previous_angle is not None:
                    total_rotation += wrap_degrees(angle_now - previous_angle)
                previous_angle = angle_now
                current_angle = wrap_degrees(angle_now - initial_angle)
                status = f"TRACKING ID {tracking_id}"
                status_color = (0, 255, 0)
            else:
                # Marker 暫時離開畫面時不累加角度；回到畫面後從新影格接續，避免假跳動。
                previous_angle = None
                if tracking_id is None:
                    other_ids = sorted(detected)
                    if other_ids:
                        status = "WRONG ID: " + ", ".join(map(str, other_ids))
                    else:
                        status = "NO MARKER DETECTED"
                else:
                    status = f"ID {tracking_id} LOST - SHOW IT AGAIN"
                status_color = (0, 165, 255)

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 180), (18, 22, 28), -1)
            cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

            draw_label(frame, "Dictionary: DICT_4X4_50  |  Target IDs: 0, 1, 2",
                       (18, 30), (235, 235, 235), 0.58)
            draw_label(frame, status, (18, 63), status_color, 0.68, 2)
            draw_label(frame,
                       f"Angle from start: {current_angle:+.1f} deg" if current_angle is not None else "Angle from start: --",
                       (18, 101), (0, 255, 0), 0.75, 2)
            draw_label(frame, f"Accumulated rotation: {total_rotation:+.1f} deg",
                       (18, 139), (0, 190, 255), 0.72, 2)
            draw_label(frame, "R: reset angle    Q / ESC: quit",
                       (18, 168), (220, 220, 220), 0.48)

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
                print("旋轉角度已歸零。")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
