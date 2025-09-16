# posture_auto_snap_debug.py
# Posture multi-view detector (front/side/back) with debug auto-capture visible and audible.
# Requirements: mediapipe, opencv-python, numpy
# pip install mediapipe opencv-python numpy

import cv2
import math
import numpy as np
import mediapipe as mp
from collections import deque
import time
import os
from datetime import datetime

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

os.makedirs("snapshots", exist_ok=True)

# ------------------- AUTO-CAPTURE TUNABLES -------------------
AUTO_ENABLED = True
LEVEL_THRESH = 0.085      # lower = stricter. typical: 0.06..0.13
STABLE_FRAMES = 12        # how many consecutive frames under threshold to trigger
COUNTDOWN_FRAMES = 12     # visible countdown frames (set 0 to save immediately)
MIN_SAVE_INTERVAL = 2.0   # seconds between auto saves
# ------------------------------------------------------------

def normalized_landmark(lm, w, h):
    return (int(lm.x * w), int(lm.y * h))

def angle_between(p1, p2, p3):
    a = np.array(p1) - np.array(p2)
    b = np.array(p3) - np.array(p2)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    cosang = np.clip(np.dot(a, b) / denom, -1.0, 1.0)
    return math.degrees(math.acos(cosang))

class RunningAvg:
    def __init__(self, window=8):
        self.q = deque(maxlen=window)
    def add(self, v):
        self.q.append(v)
    def avg(self):
        return sum(self.q)/len(self.q) if self.q else 0.0

# ---------------- front metrics ----------------
def compute_front_metrics(landmarks, w, h):
    L_SHOULDER = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    R_SHOULDER = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    L_HIP = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    R_HIP = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    NOSE = landmarks[mp_pose.PoseLandmark.NOSE.value]
    L_ANKLE = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    R_ANKLE = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

    ls = normalized_landmark(L_SHOULDER, w, h)
    rs = normalized_landmark(R_SHOULDER, w, h)
    lh = normalized_landmark(L_HIP, w, h)
    rh = normalized_landmark(R_HIP, w, h)
    nose = normalized_landmark(NOSE, w, h)
    la = normalized_landmark(L_ANKLE, w, h)
    ra = normalized_landmark(R_ANKLE, w, h)

    shoulder_dx = rs[0] - ls[0]; shoulder_dy = rs[1] - ls[1]
    shoulder_tilt_deg = math.degrees(math.atan2(shoulder_dy, shoulder_dx))
    hip_dx = rh[0] - lh[0]; hip_dy = rh[1] - lh[1]
    hip_tilt_deg = math.degrees(math.atan2(hip_dy, hip_dx))
    mid_sh = ((ls[0]+rs[0])/2.0, (ls[1]+rs[1])/2.0)
    head_tilt_deg = math.degrees(math.atan2(nose[1]-mid_sh[1], nose[0]-mid_sh[0]))
    mid_ankles_x = (la[0] + ra[0]) / 2.0
    mid_shoulders_x = mid_sh[0]
    lateral_shift_px = mid_ankles_x - mid_shoulders_x
    lateral_shift_norm = lateral_shift_px / w
    shoulder_vert_diff = ls[1] - rs[1]
    hip_vert_diff = lh[1] - rh[1]

    rotation_x = (nose[0] - mid_sh[0]) / w  # normalized yaw-ish

    return {
        "shoulder_tilt_deg": shoulder_tilt_deg,
        "hip_tilt_deg": hip_tilt_deg,
        "head_tilt_deg": head_tilt_deg,
        "lateral_shift_norm": lateral_shift_norm,
        "shoulder_vert_diff_px": shoulder_vert_diff,
        "hip_vert_diff_px": hip_vert_diff,
        "rotation_x": rotation_x,
        "points": {"ls": ls, "rs": rs, "lh": lh, "rh": rh, "nose": nose, "la": la, "ra": ra}
    }

# ---------------- side metrics ----------------
def compute_side_metrics(landmarks, w, h):
    def vis_of(idx):
        return landmarks[idx].visibility if hasattr(landmarks[idx], 'visibility') else 0.0
    left_indices = [mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                    mp_pose.PoseLandmark.LEFT_HIP.value,
                    mp_pose.PoseLandmark.LEFT_ANKLE.value]
    right_indices = [mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                    mp_pose.PoseLandmark.RIGHT_HIP.value,
                    mp_pose.PoseLandmark.RIGHT_ANKLE.value]
    left_vis = sum([vis_of(i) for i in left_indices]) / len(left_indices)
    right_vis = sum([vis_of(i) for i in right_indices]) / len(right_indices)

    use_side = "left" if left_vis > right_vis else "right"
    if use_side == "left":
        SHOULDER = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        HIP = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        KNEE = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        ANKLE = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    else:
        SHOULDER = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        HIP = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        KNEE = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        ANKLE = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

    sh = normalized_landmark(SHOULDER, w, h)
    hp = normalized_landmark(HIP, w, h)
    kn = normalized_landmark(KNEE, w, h)
    an = normalized_landmark(ANKLE, w, h)

    v = (hp[0]-sh[0], hp[1]-sh[1])
    trunk_angle_deg = math.degrees(math.atan2(v[0], v[1]))
    lumbar_angle = angle_between(sh, hp, kn)
    thoracic_proxy = angle_between(sh, hp, an)

    return {
        "side": use_side,
        "trunk_angle_deg": trunk_angle_deg,
        "lumbar_angle_deg": lumbar_angle,
        "thoracic_proxy_deg": thoracic_proxy,
        "points": {"sh": sh, "hp": hp, "kn": kn, "an": an}
    }

# ------------- Levelness scoring -------------
def levelness_score(front_metrics):
    st = abs(front_metrics["shoulder_tilt_deg"]) / 20.0
    ht = abs(front_metrics["hip_tilt_deg"]) / 20.0
    hd = abs(front_metrics["head_tilt_deg"]) / 20.0
    ls = abs(front_metrics["lateral_shift_norm"]) / 0.25
    rx = abs(front_metrics.get("rotation_x", 0.0)) * 1.5
    score = 0.32*st + 0.28*ht + 0.18*hd + 0.18*ls + 0.04*rx
    return max(0.0, min(1.0, score))

def save_snapshot(frame, mode, tag="auto"):
    t = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"snapshots/snap_{tag}_{mode}_{t}.png"
    cv2.imwrite(fname, frame)
    print(f"\a[AUTO-SNAP] Saved snapshot: {fname}")  # \a = bell

def open_camera_robust(preferred_index=None, tries=3, delay=0.5):
    # simple candidate list from /dev/video*
    import glob
    candidates = []
    for dev in sorted(glob.glob("/dev/video*")):
        try:
            n = int(dev.replace("/dev/video", ""))
            if n not in candidates:
                candidates.append(n)
        except:
            pass
    if preferred_index is not None and preferred_index not in candidates:
        candidates.insert(0, preferred_index)
    for i in range(4):
        if i not in candidates:
            candidates.append(i)
    backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
    for idx in candidates:
        for backend in backends:
            for attempt in range(tries):
                cap = cv2.VideoCapture(idx, backend)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size>0:
                        print(f"[camera] Opened index {idx} (backend={backend})")
                        return cap
                    cap.release()
                time.sleep(delay)
    return None

# ------------------- Main -------------------
def main():
    cap = open_camera_robust()
    if cap is None:
        print("ERROR: Could not open camera. Try camera_probe_all.py")
        return

    view_mode = "front"
    s_sh = RunningAvg(window=6)
    s_hp = RunningAvg(window=6)
    s_hd = RunningAvg(window=6)
    s_ls = RunningAvg(window=6)

    stable_counter = 0
    countdown_counter = 0
    last_saved_time = 0

    with mp_pose.Pose(static_image_mode=False, model_complexity=1,
                      enable_segmentation=False,
                      min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        prev_time = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame read failed — stopping")
                break
            h, w = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                if view_mode in ("front", "back"):
                    front_m = compute_front_metrics(results.pose_landmarks.landmark, w, h)
                    s_sh.add(front_m["shoulder_tilt_deg"])
                    s_hp.add(front_m["hip_tilt_deg"])
                    s_hd.add(front_m["head_tilt_deg"])
                    s_ls.add(front_m["lateral_shift_norm"])
                    smooth_front = {
                        "shoulder_tilt_deg": s_sh.avg(),
                        "hip_tilt_deg": s_hp.avg(),
                        "head_tilt_deg": s_hd.avg(),
                        "lateral_shift_norm": s_ls.avg(),
                        "shoulder_vert_diff_px": front_m["shoulder_vert_diff_px"],
                        "hip_vert_diff_px": front_m["hip_vert_diff_px"],
                        "rotation_x": front_m["rotation_x"],
                        "points": front_m["points"]
                    }
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    pts = smooth_front["points"]
                    cv2.line(frame, pts["ls"], pts["rs"], (0,255,0), 2)
                    cv2.line(frame, pts["lh"], pts["rh"], (255,0,0), 2)
                    mid_sh = (int((pts["ls"][0]+pts["rs"][0])/2), int((pts["ls"][1]+pts["rs"][1])/2))
                    mid_an = (int((pts["la"][0]+pts["ra"][0])/2), int((pts["la"][1]+pts["ra"][1])/2))
                    cv2.circle(frame, mid_sh, 5, (0,255,255), -1)
                    cv2.circle(frame, mid_an, 5, (0,255,255), -1)
                    cv2.line(frame, mid_sh, mid_an, (0,255,255), 1)

                    score = levelness_score(smooth_front)
                    # draw level bar
                    sb_x = 10; sb_y = 110
                    cv2.putText(frame, f"Level score: {score:.3f}", (sb_x, sb_y-12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240,240,240), 1)
                    bar_w = 300
                    fill_w = int((1.0 - score) * bar_w)
                    cv2.rectangle(frame, (sb_x, sb_y), (sb_x+bar_w, sb_y+18), (40,40,40), -1)
                    cv2.rectangle(frame, (sb_x, sb_y), (sb_x+fill_w, sb_y+18), (50,200,50), -1)
                    cv2.rectangle(frame, (sb_x, sb_y), (sb_x+bar_w, sb_y+18), (200,200,200), 1)

                    # show stable counter and countdown large on-screen
                    cv2.putText(frame, f"StableCounter: {stable_counter}", (sb_x, sb_y+40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220,220,0), 2)
                    if countdown_counter>0:
                        cv2.putText(frame, f"Auto in: {countdown_counter}", (w//2 - 100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,200,200), 3)

                    # debug small overlay
                    cv2.putText(frame, f"Sdeg:{smooth_front['shoulder_tilt_deg']:+.1f} Hdeg:{smooth_front['hip_tilt_deg']:+.1f} Shift:{smooth_front['lateral_shift_norm']:+.3f} RotX:{smooth_front['rotation_x']:+.3f}",
                                (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230,230,230), 1)

                    # AUTO logic
                    if AUTO_ENABLED:
                        now = time.time()
                        if (now - last_saved_time) > MIN_SAVE_INTERVAL:
                            if score <= LEVEL_THRESH:
                                stable_counter += 1
                                print(f"[DEBUG] score {score:.3f} <= {LEVEL_THRESH}, stable_counter -> {stable_counter}")
                                if stable_counter >= STABLE_FRAMES:
                                    # begin countdown if set
                                    if COUNTDOWN_FRAMES > 0:
                                        if countdown_counter == 0:
                                            countdown_counter = COUNTDOWN_FRAMES
                                        countdown_counter -= 1
                                        # display countdown (already drawn above)
                                        if countdown_counter <= 0:
                                            save_snapshot(frame, view_mode, tag="auto")
                                            last_saved_time = now
                                            stable_counter = 0
                                            countdown_counter = 0
                                    else:
                                        save_snapshot(frame, view_mode, tag="auto")
                                        last_saved_time = now
                                        stable_counter = 0
                            else:
                                if stable_counter != 0:
                                    print(f"[DEBUG] score {score:.3f} > {LEVEL_THRESH}, resetting stable_counter")
                                stable_counter = 0
                                countdown_counter = 0
                        else:
                            # cooldown text
                            cd = int(max(0, MIN_SAVE_INTERVAL - (now - last_saved_time)))
                            cv2.putText(frame, f"Auto cooldown: {cd}s", (w-240, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,80), 2)

                elif view_mode == "side":
                    side_m = compute_side_metrics(results.pose_landmarks.landmark, w, h)
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    pts = side_m["points"]
                    cv2.circle(frame, pts["sh"], 6, (0,255,0), -1)
                    cv2.circle(frame, pts["hp"], 6, (255,0,0), -1)
                    cv2.circle(frame, pts["an"], 6, (0,255,255), -1)
                    cv2.line(frame, pts["sh"], pts["hp"], (0,255,0), 2)
                    cv2.line(frame, pts["hp"], pts["an"], (255,0,0), 2)
                    trunk = side_m["trunk_angle_deg"]
                    lum = side_m["lumbar_angle_deg"]
                    thor = side_m["thoracic_proxy_deg"]
                    cv2.putText(frame, f"Side: {side_m['side']}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
                    cv2.putText(frame, f"Trunk lean (deg): {trunk:.1f}", (10,44), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                    cv2.putText(frame, f"Lumbar proxy (deg): {lum:.1f}", (10,68), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                    cv2.putText(frame, f"Thoracic proxy (deg): {thor:.1f}", (10,92), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            else:
                cv2.putText(frame, "No pose detected - move into frame", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,200), 2)

            cv2.putText(frame, f"Mode: {view_mode.upper()}  (f:front  l:side  b:back  s:snap  g:force-autosnap  q:quit)", (10, h-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1)

            cur = time.time()
            fps = 1.0 / (cur - prev_time) if cur != prev_time else 0.0
            prev_time = cur
            cv2.putText(frame, f"FPS: {fps:.1f}", (w-110, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

            cv2.imshow("Posture Multi-View Detector (AutoSnap DEBUG)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('f'):
                view_mode = "front"; stable_counter = 0; countdown_counter = 0
            elif key == ord('l'):
                view_mode = "side"; stable_counter = 0; countdown_counter = 0
            elif key == ord('b'):
                view_mode = "back"; stable_counter = 0; countdown_counter = 0
            elif key == ord('s'):
                t = datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = f"snapshots/snap_manual_{view_mode}_{t}.png"
                cv2.imwrite(fname, frame)
                print(f"[MANUAL] Saved snapshot: {fname}")
            elif key == ord('g'):
                # force an auto-snap (test)
                save_snapshot(frame, view_mode, tag="forced")
                last_saved_time = time.time()
                stable_counter = 0
                countdown_counter = 0

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
