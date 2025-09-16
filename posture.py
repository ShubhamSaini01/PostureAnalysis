# posture_report_autocapture.py
# Auto-capture on movement stop + compute pelvis/hip/knee/ankle angles + produce imbalance report (JSON + CSV) + snapshot
# Requirements: mediapipe, opencv-python, numpy, optional: simpleaudio
# pip install mediapipe opencv-python numpy simpleaudio

import cv2, math, os, time, json, csv
import numpy as np
import mediapipe as mp
from collections import deque
from datetime import datetime

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ensure folders
os.makedirs("snapshots", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# ---------- Tunables ----------
STABLE_SECONDS = 2.0          # how long movement must be low to auto-capture
DISP_WINDOW = 10              # number of frames to compute average displacement (sliding window)
DISP_THRESHOLD_RATIO = 0.008  # displacement threshold relative to image diagonal (smaller = stricter)
MIN_FRAMES_FOR_POSE = 6       # require some frames with pose to start stability checks
FPS_SMOOTH = 6
AUTO_ENABLED = True
# --------------------------------

# optional audio (simpleaudio recommended, fallback to terminal bell)
_use_simpleaudio = False
try:
    import simpleaudio as sa
    _use_simpleaudio = True
except Exception:
    _use_simpleaudio = False

def make_beep_wave(frequency=880.0, duration_ms=100, volume=0.25, sr=44100):
    t = np.linspace(0, duration_ms/1000.0, int(sr * duration_ms/1000.0), False)
    note = np.sin(frequency * t * 2 * np.pi)
    audio = note * (2**15 - 1) * volume
    audio = audio.astype(np.int16)
    return audio.tobytes()

if _use_simpleaudio:
    BEEP = make_beep_wave(880.0, 120)

def play_beep():
    if _use_simpleaudio:
        try:
            sa.play_buffer(BEEP, 1, 2, 44100)
            return
        except Exception:
            pass
    print("\a", end="", flush=True)

# helper functions
def normalized_landmark(lm, w, h):
    return (lm.x * w, lm.y * h)

def angle_at(p1, p2, p3):
    a = np.array(p1) - np.array(p2)
    b = np.array(p3) - np.array(p2)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    cosang = np.clip(np.dot(a, b) / denom, -1.0, 1.0)
    return math.degrees(math.acos(cosang))

def line_angle_deg(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.degrees(math.atan2(dy, dx))

# compute metrics (angles & points)
def compute_all_metrics(landmarks, w, h):
    L = mp_pose.PoseLandmark
    def pt(idx):
        lm = landmarks[idx.value]
        return normalized_landmark(lm, w, h)

    pts = {}
    idxs = [
        L.LEFT_SHOULDER, L.RIGHT_SHOULDER, L.LEFT_HIP, L.RIGHT_HIP,
        L.NOSE, L.LEFT_KNEE, L.RIGHT_KNEE, L.LEFT_ANKLE, L.RIGHT_ANKLE,
        L.LEFT_FOOT_INDEX, L.RIGHT_FOOT_INDEX, L.LEFT_HEEL, L.RIGHT_HEEL
    ]
    for idn in idxs:
        pts[idn.name.lower()] = pt(idn)

    ls = pts['left_shoulder']; rs = pts['right_shoulder']
    lh = pts['left_hip']; rh = pts['right_hip']
    nose = pts['nose']
    la = pts['left_ankle']; ra = pts['right_ankle']
    lfoot = pts['left_foot_index']; rfoot = pts['right_foot_index']

    shoulder_tilt = line_angle_deg(ls, rs)
    hip_tilt = line_angle_deg(lh, rh)
    mid_shoulders = ((ls[0]+rs[0])/2.0, (ls[1]+rs[1])/2.0)
    head_tilt = math.degrees(math.atan2(nose[1]-mid_shoulders[1], nose[0]-mid_shoulders[0]))
    mid_ankles_x = (la[0] + ra[0]) / 2.0
    lateral_shift_norm = (mid_ankles_x - mid_shoulders[0]) / w
    pelvic_tilt = hip_tilt

    left_knee_angle = angle_at(pts['left_hip'], pts['left_knee'], pts['left_ankle'])
    right_knee_angle = angle_at(pts['right_hip'], pts['right_knee'], pts['right_ankle'])
    left_ankle_angle = angle_at(pts['left_knee'], pts['left_ankle'], pts['left_foot_index'])
    right_ankle_angle = angle_at(pts['right_knee'], pts['right_ankle'], pts['right_foot_index'])
    if np.allclose(pts['left_foot_index'], (0.0, 0.0)):
        left_ankle_angle = angle_at(pts['left_knee'], pts['left_ankle'], pts['left_heel'])
    if np.allclose(pts['right_foot_index'], (0.0, 0.0)):
        right_ankle_angle = angle_at(pts['right_knee'], pts['right_ankle'], pts['right_heel'])

    out = {
        "shoulder_tilt_deg": shoulder_tilt,
        "hip_tilt_deg": hip_tilt,
        "pelvic_tilt_deg": pelvic_tilt,
        "head_tilt_deg": head_tilt,
        "lateral_shift_norm": lateral_shift_norm,
        "left_knee_deg": left_knee_angle,
        "right_knee_deg": right_knee_angle,
        "left_ankle_deg": left_ankle_angle,
        "right_ankle_deg": right_ankle_angle,
        "points": pts
    }
    return out

def interpret_report(metrics):
    msgs = []
    s = abs(metrics['shoulder_tilt_deg'])
    if s < 5:
        msgs.append(("Shoulders", "Level"))
    elif s < 10:
        msgs.append(("Shoulders", f"Mild tilt ({s:.1f}°)"))
    else:
        msgs.append(("Shoulders", f"Marked tilt ({s:.1f}°)"))

    p = abs(metrics['pelvic_tilt_deg'])
    if p < 5:
        msgs.append(("Pelvis", "Level"))
    elif p < 10:
        msgs.append(("Pelvis", f"Mild pelvic obliquity ({p:.1f}°)"))
    else:
        msgs.append(("Pelvis", f"Marked pelvic obliquity ({p:.1f}°)"))

    ls = abs(metrics['lateral_shift_norm'])
    if ls < 0.03:
        msgs.append(("Weight", "Centered"))
    elif ls < 0.07:
        msgs.append(("Weight", "Slight lateral shift"))
    else:
        msgs.append(("Weight", "Notable lateral shift"))

    lk = metrics['left_knee_deg']; rk = metrics['right_knee_deg']
    if lk < 160:
        msgs.append(("Left knee", f"Flexed/forward angle {lk:.1f}°"))
    if rk < 160:
        msgs.append(("Right knee", f"Flexed/forward angle {rk:.1f}°"))

    la = metrics['left_ankle_deg']; ra = metrics['right_ankle_deg']
    msgs.append(("Left ankle", f"{la:.1f}°"))
    msgs.append(("Right ankle", f"{ra:.1f}°"))

    return msgs

# CSV setup
CSV_PATH = os.path.join("reports", "metrics.csv")
CSV_HEADER = [
    "timestamp","mode","image","shoulder_tilt_deg","hip_tilt_deg","pelvic_tilt_deg","head_tilt_deg",
    "lateral_shift_norm","left_knee_deg","right_knee_deg","left_ankle_deg","right_ankle_deg","notes"
]
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)

def save_report_and_snapshot(frame, mode, metrics, notes=None):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_name = f"snapshots/snap_{mode}_{ts}.png"
    cv2.imwrite(img_name, frame)
    report = {
        "timestamp": ts,
        "mode": mode,
        "image": img_name,
        "metrics": {k:v for k,v in metrics.items() if k!="points"},
        "notes": notes or [],
    }
    report_name = os.path.join("reports", f"report_{ts}.json")
    with open(report_name, "w") as rf:
        json.dump(report, rf, indent=2)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.writer(f)
        row = [
            ts, mode, img_name,
            metrics['shoulder_tilt_deg'], metrics['hip_tilt_deg'], metrics['pelvic_tilt_deg'], metrics['head_tilt_deg'],
            metrics['lateral_shift_norm'], metrics['left_knee_deg'], metrics['right_knee_deg'], metrics['left_ankle_deg'], metrics['right_ankle_deg'],
            "; ".join(notes) if notes else ""
        ]
        w.writerow(row)
    play_beep()
    print(f"[SNAPSHOT] {img_name}  report -> {report_name}")

# movement detection helpers
def avg_disp_between_frames(ptsA, ptsB):
    tot = 0.0; n=0
    for k in ptsA.keys():
        pa = np.array(ptsA[k]); pb = np.array(ptsB.get(k, pa))
        d = np.linalg.norm(pa-pb)
        tot += d; n+=1
    return (tot / n) if n>0 else 0.0

def open_camera_robust():
    import glob
    candidates = []
    for dev in sorted(glob.glob("/dev/video*")):
        try:
            n = int(dev.replace("/dev/video",""))
            candidates.append(n)
        except:
            pass
    for i in range(4):
        if i not in candidates:
            candidates.append(i)
    backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
    for idx in candidates:
        for b in backends:
            cap = cv2.VideoCapture(idx, b)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    print(f"Opened camera idx {idx} backend {b}")
                    return cap
                cap.release()
    return None

# ----------------- Main -----------------
def main():
    cap = open_camera_robust()
    if cap is None:
        print("ERROR: no camera opened. Run camera_probe_all.py")
        return

    view_mode = "front"
    mp_pose_instance = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    prev_pts = None
    disp_window = deque(maxlen=DISP_WINDOW)
    pose_frames = 0
    fps_est = deque(maxlen=FPS_SMOOTH)
    stable_start_time = None

    print("Controls: f=front l=side b=back s=manual snap g=force autosnap c=save baseline as baseline.json q=quit")

    baseline = None
    if os.path.exists("baseline.json"):
        try:
            baseline = json.load(open("baseline.json","r"))
            print("Loaded baseline.json")
        except:
            baseline = None

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Frame read failed — stopping.")
            break
        h,w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_pose_instance.process(frame_rgb)

        # defaults for display (avoid UnboundLocalError)
        avg_disp = 0.0
        norm_disp = 1.0

        if results.pose_landmarks:
            pose_frames += 1
            lm = results.pose_landmarks.landmark
            keys = [
                mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
                mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP,
                mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE,
                mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE,
                mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_FOOT_INDEX, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX
            ]
            pts = {}
            for k in keys:
                pts[k.name] = normalized_landmark(lm[k.value], w, h)

            if prev_pts is not None:
                disp = avg_disp_between_frames(pts, prev_pts)
                disp_window.append(disp)
            prev_pts = pts

            if len(disp_window) >= 1:
                avg_disp = float(np.mean(disp_window))
                diag = math.hypot(w,h)
                norm_disp = avg_disp / diag

            metrics = compute_all_metrics(lm, w, h)

            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            ls_pt = (int(metrics['points']['left_shoulder'][0]), int(metrics['points']['left_shoulder'][1]))
            rs_pt = (int(metrics['points']['right_shoulder'][0]), int(metrics['points']['right_shoulder'][1]))
            lh_pt = (int(metrics['points']['left_hip'][0]), int(metrics['points']['left_hip'][1]))
            rh_pt = (int(metrics['points']['right_hip'][0]), int(metrics['points']['right_hip'][1]))
            cv2.line(frame, ls_pt, rs_pt, (0,255,0), 2)
            cv2.line(frame, lh_pt, rh_pt, (255,0,0), 2)

            cv2.putText(frame, f"Sdeg:{metrics['shoulder_tilt_deg']:+.1f} Hdeg:{metrics['hip_tilt_deg']:+.1f} Lshift:{metrics['lateral_shift_norm']:+.3f}",
                        (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240,240,240), 2)

            cv2.putText(frame, f"Mov avg(pixels): {avg_disp:.1f}  norm: {norm_disp:.4f}", (10,45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

            if pose_frames >= MIN_FRAMES_FOR_POSE and norm_disp <= DISP_THRESHOLD_RATIO:
                if stable_start_time is None:
                    stable_start_time = time.time()
                elapsed = time.time() - stable_start_time
                cv2.putText(frame, f"STABLE: {elapsed:.2f}s (need {STABLE_SECONDS}s)", (10,75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,0), 2)
                if AUTO_ENABLED and elapsed >= STABLE_SECONDS:
                    notes = [m[0] + ": " + m[1] for m in interpret_report(metrics)]
                    save_report_and_snapshot(frame.copy(), view_mode, metrics, notes=notes)
                    stable_start_time = None
                    disp_window.clear()
                    pose_frames = 0
                    time.sleep(0.6)
            else:
                if stable_start_time is not None:
                    stable_start_time = None
                cv2.putText(frame, f"STABLE: 0.00s (move to stop)", (10,75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80,80,80), 2)

        else:
            prev_pts = None
            disp_window.clear()
            pose_frames = 0
            stable_start_time = None
            cv2.putText(frame, "No pose detected - move into frame", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,200), 2)

        cv2.putText(frame, f"Mode:{view_mode.upper()}  (f:front l:side b:back s:manual g:force c:set baseline q:quit)", (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1)

        cv2.imshow("Posture AutoCapture + Report", frame)
        key = cv2.waitKey(1) & 0xFF
        fps_est.append(1.0 / max(1e-6, time.time()-t0))

        if key == ord('q') or key == 27:
            break
        elif key == ord('f'):
            view_mode = "front"
        elif key == ord('l'):
            view_mode = "side"
        elif key == ord('b'):
            view_mode = "back"
        elif key == ord('s'):
            if results and results.pose_landmarks:
                metrics = compute_all_metrics(results.pose_landmarks.landmark, w, h)
                notes = [m[0] + ": " + m[1] for m in interpret_report(metrics)]
                save_report_and_snapshot(frame.copy(), view_mode, metrics, notes=notes)
        elif key == ord('g'):
            if results and results.pose_landmarks:
                metrics = compute_all_metrics(results.pose_landmarks.landmark, w, h)
                notes = [m[0] + ": " + m[1] for m in interpret_report(metrics)]
                save_report_and_snapshot(frame.copy(), view_mode, metrics, notes=notes)
        elif key == ord('c'):
            if results and results.pose_landmarks:
                metrics = compute_all_metrics(results.pose_landmarks.landmark, w, h)
                baseline_data = {k:metrics[k] for k in metrics if k!="points"}
                baseline_data['timestamp'] = datetime.now().isoformat()
                with open("baseline.json","w") as bf:
                    json.dump(baseline_data, bf, indent=2)
                print("Saved baseline.json")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
