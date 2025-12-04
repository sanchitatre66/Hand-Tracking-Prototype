import cv2
import numpy as np
import time

# Face detector (classic Haar)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Calibration globals
calibrated_hsv = None
hsv_margin = np.array([12, 60, 60], dtype=np.int32)  # tuneable
show_masks = False

# mouse callback param will hold the current frame copy
def on_mouse(event, x, y, flags, param):
    global calibrated_hsv
    frame = param.get('frame', None)
    if event == cv2.EVENT_LBUTTONDOWN and frame is not None:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        calibrated_hsv = hsv[y, x].astype(np.int32)
        print("Calibrated HSV:", calibrated_hsv)

def get_skin_mask(hsv_frame):
    global calibrated_hsv, hsv_margin
    if calibrated_hsv is not None:
        lower = np.clip(calibrated_hsv - hsv_margin, 0, 255).astype(np.uint8)
        upper = np.clip(calibrated_hsv + hsv_margin, 0, 255).astype(np.uint8)
        mask = cv2.inRange(hsv_frame, lower, upper)
    else:
        # default rough skin range
        lower = np.array([0, 20, 70], dtype=np.uint8)
        upper = np.array([25, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv_frame, lower, upper)

    # cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (5,5), 0)
    return mask

def main():
    global show_masks, hsv_margin, calibrated_hsv
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Camera not found")
        return

    # virtual object (rectangle)
    obj_x1, obj_y1 = 250, 150
    obj_x2, obj_y2 = 390, 290

    far_thresh = 150
    near_thresh = 60

    back_sub = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=25, detectShadows=False)

    window_name = "Hand Prototype (convex hull)"
    cv2.namedWindow(window_name)
    mouse_param = {'frame': None}
    cv2.setMouseCallback(window_name, on_mouse, mouse_param)

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        mouse_param['frame'] = frame.copy()  # used by mouse callback

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        skin_mask = get_skin_mask(hsv)

        # motion mask
        motion_mask = back_sub.apply(frame)
        kernel_motion = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel_motion, iterations=1)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel_motion, iterations=1)

        # combined mask prefers moving skin-like regions
        combined_mask = cv2.bitwise_and(skin_mask, motion_mask)

        # remove face area from combined and skin masks
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            pad = 10
            rx1, ry1 = max(0, x-pad), max(0, y-pad)
            rx2, ry2 = min(frame.shape[1], x+w+pad), min(frame.shape[0], y+h+pad)
            combined_mask[ry1:ry2, rx1:rx2] = 0
            skin_mask[ry1:ry2, rx1:rx2] = 0
            # debug: draw face box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (200, 100, 10), 1)

        # find contours
        contours, _ = cv2.findContours(combined_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            contours, _ = cv2.findContours(skin_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # find candidate hand contour
        hand_contour = None
        hand_center = None
        min_area = 800
        max_area = 120000

        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area or area > max_area:
                continue

            x,y,w,h = cv2.boundingRect(c)
            aspect = w / float(h + 1e-5)

            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull) if len(hull) >= 3 else 0
            solidity = float(area) / hull_area if hull_area > 0 else 0

            if solidity < 0.25:
                continue
            if aspect < 0.2 or aspect > 4.0:
                continue

            if hand_contour is None or area > cv2.contourArea(hand_contour):
                hand_contour = c

        if hand_contour is not None:
            hull = cv2.convexHull(hand_contour)
            cv2.drawContours(frame, [hull], -1, (0, 255, 0), 3)

            M = cv2.moments(hull)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                hand_center = (cx, cy)
                cv2.circle(frame, hand_center, 7, (0,255,0), -1)

        # draw virtual object
        cv2.rectangle(frame, (obj_x1, obj_y1), (obj_x2, obj_y2),
                      (255, 255, 255), 2)

        state = "NO HAND"
        if hand_center is not None:
            cx, cy = hand_center

            if obj_x1 <= cx <= obj_x2 and obj_y1 <= cy <= obj_y2:
                distance = 0.0
            else:
                clamped_x = min(max(cx, obj_x1), obj_x2)
                clamped_y = min(max(cy, obj_y1), obj_y2)
                dx = cx - clamped_x
                dy = cy - clamped_y
                distance = (dx*dx + dy*dy)**0.5

            if distance <= near_thresh:
                state = "DANGER"
            elif distance <= far_thresh:
                state = "WARNING"
            else:
                state = "SAFE"

        cv2.putText(frame, f"Current state: {state}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 255), 2)

        if state == "DANGER":
            cv2.putText(frame, "DANGER DANGER",
                        (obj_x1 - 40, obj_y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.1f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 0), 2)

        if show_masks:
            skin_bgr = cv2.cvtColor(skin_mask, cv2.COLOR_GRAY2BGR)
            motion_bgr = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
            combined_bgr = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
            top = np.hstack((frame, skin_bgr))
            bottom = np.hstack((motion_bgr, combined_bgr))
            vis = np.vstack((top, bottom))
            cv2.imshow(window_name, vis)
        else:
            cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        elif key == ord('m'):
            show_masks = not show_masks
            print("Show masks:", show_masks)
        elif key == ord('c'):
            print("Click on your palm to calibrate.")
        elif key == ord('+'):
            hsv_margin = np.clip(hsv_margin + 2, 0, 127)
            print("HSV margin:", hsv_margin)
        elif key == ord('-'):
            hsv_margin = np.clip(hsv_margin - 2, 0, 127)
            print("HSV margin:", hsv_margin)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    prev_time = time.time()
    main()
