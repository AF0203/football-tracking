# tracker/track.py
"""
Optimierter Football Tracker
→ YOLOv9e Detection
→ Frame-by-Frame ohne Tracking (stabile Erkennung)
→ K-Means Team-Zuweisung mit Vote-System
→ Kalman-Filter für Ball-Smoothing
→ Ball-Interpolation bis 20 Frames

Aufruf:
    python3 tracker/track.py \
        --clip clips/test.mp4 \
        --calibration output/calibration_test.json
"""

import sys, os, argparse, json
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from sklearn.cluster import KMeans
from collections import deque
import supervision as sv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import MODELS_DIR, CLIPS_DIR, OUTPUT_DIR

FIELD_WIDTH  = 68.0
FIELD_LENGTH = 105.0
CLASS_NAMES  = {0:"ball", 1:"goalkeeper", 2:"player", 3:"referee"}
TEAM_A_COL   = (255, 80,  80)
TEAM_B_COL   = ( 80, 120, 255)
REF_COL      = (200, 200,  80)
BALL_COL     = (255, 255, 255)


class CalibrationData:
    def __init__(self, path):
        with open(path) as f: data=json.load(f)
        self.H    = np.array(data["H"],     dtype=np.float32)
        self.H_inv= np.array(data["H_inv"], dtype=np.float32)
        self.field_w = data["field_w"]
        self.field_h = data["field_h"]
        self.team_a_color = data.get("team_a_color")
        self.team_b_color = data.get("team_b_color")
        self.ignored_positions = data.get("ignored_positions",[])
        if self.team_a_color:
            self.ca = np.array(self.team_a_color, dtype=np.float32)
            self.cb = np.array(self.team_b_color, dtype=np.float32)
        else:
            self.ca = self.cb = None
        print(f"  Kalibrierung geladen")

    def pixel_to_field(self, px, py):
        pt = np.array([[[float(px),float(py)]]],dtype=np.float32)
        r  = cv2.perspectiveTransform(pt,self.H)
        fx,fy = r[0][0]
        return round(float((fx/self.field_w)*FIELD_LENGTH),2), \
               round(float((fy/self.field_h)*FIELD_WIDTH),2)

    def is_in_field(self, fx, fy, margin=3.0):
        return -margin<=fx<=FIELD_LENGTH+margin and \
               -margin<=fy<=FIELD_WIDTH+margin

    def is_ignored(self, px, py, threshold=100):
        return any(((px-ip[0])**2+(py-ip[1])**2)**0.5<threshold
                   for ip in self.ignored_positions)

    def assign_team_ref(self, frame, x1, y1, x2, y2):
        """Referenzfarbe-basierte Team-Zuweisung"""
        if self.ca is None: return "A"
        h=y2-y1; w=x2-x1
        top  = max(0,              y1+int(h*0.10))
        bot  = min(frame.shape[0]-1, y1+int(h*0.65))
        left = max(0,              x1+int(w*0.10))
        right= min(frame.shape[1]-1, x2-int(w*0.10))
        roi  = frame[top:bot,left:right]
        if roi.size==0: return "A"
        avg = np.median(roi.reshape(-1,3),axis=0).astype(np.float32)
        return "A" if np.linalg.norm(avg-self.ca)<np.linalg.norm(avg-self.cb) else "B"


class StableTeamTracker:
    """
    Stabile Team-Zuordnung über ByteTrack IDs:
    → ByteTrack gibt stabile IDs
    → Team wird über erste LOCK_FRAMES Frames
      per Vote bestimmt dann eingefroren
    → danach ändert sich Team nie mehr
    """
    LOCK_FRAMES = 15  # Nach N Frames Team einfrieren

    def __init__(self):
        self.tracker   = sv.ByteTrack(
            track_activation_threshold=0.3,
            lost_track_buffer=60,
            minimum_matching_threshold=0.8,
            frame_rate=25,
        )
        self.team_votes  = {}  # tid → {"A":0,"B":0}
        self.team_locked = {}  # tid → "A"/"B" (eingefroren)
        self.frame_count = {}  # tid → anzahl frames gesehen

    def assign(self, sv_dets, raw_teams, frame):
        """
        sv_dets: supervision Detections
        raw_teams: list of "A"/"B" pro Detection
        → gibt dict {tracker_id → team} zurück
        """
        tracked = self.tracker.update_with_detections(sv_dets)
        result  = {}

        if tracked.tracker_id is None:
            return result

        for i, tid in enumerate(tracked.tracker_id):
            tid = int(tid)

            # Schon eingefroren?
            if tid in self.team_locked:
                result[tid] = self.team_locked[tid]
                continue

            # Vote sammeln
            if i < len(raw_teams):
                raw = raw_teams[i]
                if tid not in self.team_votes:
                    self.team_votes[tid]  = {"A":0,"B":0}
                    self.frame_count[tid] = 0
                self.team_votes[tid][raw] += 1
                self.frame_count[tid]     += 1

                # Nach LOCK_FRAMES einfrieren
                if self.frame_count[tid] >= self.LOCK_FRAMES:
                    v = self.team_votes[tid]
                    self.team_locked[tid] = "A" if v["A"]>=v["B"] else "B"
                    result[tid] = self.team_locked[tid]
                else:
                    v = self.team_votes[tid]
                    result[tid] = "A" if v["A"]>=v["B"] else "B"

        return result, tracked


class BallKalman:
    """Kalman-Filter für Ball-Position"""
    def __init__(self):
        self.kf = cv2.KalmanFilter(4,2)
        self.kf.transitionMatrix   = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],dtype=np.float32)
        self.kf.measurementMatrix  = np.array([[1,0,0,0],[0,1,0,0]],dtype=np.float32)
        self.kf.processNoiseCov    = np.eye(4,dtype=np.float32)*0.1
        self.kf.measurementNoiseCov= np.eye(2,dtype=np.float32)*1.0
        self.kf.errorCovPost       = np.eye(4,dtype=np.float32)*10
        self.initialized = False

    def update(self, x, y):
        if not self.initialized:
            self.kf.statePost = np.array([[x],[y],[0.],[0.]],dtype=np.float32)
            self.initialized  = True
        self.kf.correct(np.array([[x],[y]],dtype=np.float32))
        return x, y

    def predict(self):
        pred = self.kf.predict()
        return float(pred[0][0]), float(pred[1][0])


class EMAPosition:
    """
    Robustes Positions-Smoothing:
    → EMA für Grundglättung
    → Median-Buffer gegen Ausreißer
    → Mindestbewegung verhindert Zittern
    """
    def __init__(self, alpha=0.20, buffer_size=5, min_move=0.4):
        self.alpha      = alpha
        self.buffer_size= buffer_size
        self.min_move   = min_move  # Meter
        self.pos        = {}   # key → (ema_x, ema_y)
        self.buffer     = {}   # key → deque of (x,y)

    def smooth(self, key, x, y):
        from collections import deque

        # Buffer initialisieren
        if key not in self.buffer:
            self.buffer[key] = deque(maxlen=self.buffer_size)
            self.pos[key]    = (x, y)

        self.buffer[key].append((x, y))

        # Median aus Buffer — eliminiert Ausreißer
        buf = np.array(list(self.buffer[key]))
        mx  = float(np.median(buf[:,0]))
        my  = float(np.median(buf[:,1]))

        # EMA auf Median
        ox, oy = self.pos[key]
        nx = self.alpha * mx + (1-self.alpha) * ox
        ny = self.alpha * my + (1-self.alpha) * oy

        # Mindestbewegung — kleines Zittern ignorieren
        dist = ((nx-ox)**2 + (ny-oy)**2)**0.5
        if dist < self.min_move and len(self.buffer[key]) >= 2:
            return ox, oy  # Position nicht updaten

        self.pos[key] = (nx, ny)
        return nx, ny


def get_trikot_color(frame, x1, y1, x2, y2):
    h=y2-y1; w=x2-x1
    top  = max(0,              y1+int(h*0.10))
    bot  = min(frame.shape[0]-1, y1+int(h*0.65))
    left = max(0,              x1+int(w*0.10))
    right= min(frame.shape[1]-1, x2-int(w*0.10))
    roi  = frame[top:bot,left:right]
    if roi.size==0: return [128,128,128]
    return np.median(roi.reshape(-1,3),axis=0).astype(int).tolist()


def process_frame(frame, model, calib, team_tracker, ball_kalman,
                  ema, conf=0.3, last_ball=None, ball_missing=0):

    # Höhere Auflösung für bessere Erkennung kleiner Spieler
    scale  = min(1536/frame.shape[1],1536/frame.shape[0])
    small  = cv2.resize(frame,(int(frame.shape[1]*scale),int(frame.shape[0]*scale))) if scale<1.0 else frame
    results= model(small,conf=conf,iou=0.5,verbose=False)[0]

    player_boxes = []
    player_meta  = []
    ball         = None

    for i,box in enumerate(results.boxes):
        cls_id = int(box.cls[0])
        x1,y1,x2,y2 = map(int,box.xyxy[0])
        if scale<1.0:
            x1=int(x1/scale);y1=int(y1/scale);x2=int(x2/scale);y2=int(y2/scale)
        cx=(x1+x2)//2; cy=(y1+y2)//2

        if cls_id==0:
            fm_x,fm_y = calib.pixel_to_field(cx,cy)
            if calib.is_in_field(fm_x,fm_y,margin=20):
                sx,sy = ball_kalman.update(fm_x,fm_y)
                sx,sy = ema.smooth("ball",sx,sy)
                ball  = {"center":[cx,cy],"field_pos":[round(sx,2),round(sy,2)],
                         "conf":round(float(box.conf[0]),3),"interpolated":False}
            continue

        if (x2-x1)*(y2-y1)<400: continue
        if calib.is_ignored(cx,cy): continue

        fm_x,fm_y = calib.pixel_to_field((x1+x2)//2,y2)
        if not calib.is_in_field(fm_x,fm_y): continue

        cls_name = CLASS_NAMES.get(cls_id,"player")
        raw_team = "ref" if cls_name=="referee" else                    calib.assign_team_ref(frame,x1,y1,x2,y2)

        player_boxes.append([x1,y1,x2,y2])
        player_meta.append({
            "cls_name": cls_name,
            "raw_team": raw_team,
            "bbox":     [x1,y1,x2,y2],
            "center":   [cx,cy],
            "foot":     [(x1+x2)//2,y2],
            "fm_x":     fm_x,
            "fm_y":     fm_y,
            "conf":     round(float(box.conf[0]),3),
        })

    # ByteTrack für stabile IDs + Team-Zuordnung
    players = []
    if player_boxes:
        sv_dets = sv.Detections(
            xyxy       = np.array(player_boxes, dtype=np.float32),
            class_id   = np.array([2]*len(player_boxes)),
            confidence = np.array([m["conf"] for m in player_meta], dtype=np.float32),
        )
        raw_teams = [m["raw_team"] for m in player_meta]
        team_map, tracked = team_tracker.assign(sv_dets, raw_teams, frame)

        if tracked.tracker_id is not None:
            for i, tid in enumerate(tracked.tracker_id):
                if i >= len(player_meta): continue
                meta  = player_meta[i]
                tid   = int(tid)
                team  = team_map.get(tid, meta["raw_team"])
                fm_x  = meta["fm_x"]; fm_y = meta["fm_y"]

                # EMA Smoothing per stabiler ID
                sx,sy = ema.smooth(f"p_{tid}", fm_x, fm_y)

                players.append({
                    "id":        tid,
                    "class":     meta["cls_name"],
                    "team":      team,
                    "bbox":      meta["bbox"],
                    "center":    meta["center"],
                    "foot":      meta["foot"],
                    "field_pos": [round(sx,2),round(sy,2)],
                    "conf":      meta["conf"],
                })

    # Ball-Interpolation
    if ball is None:
        ball_missing += 1
        if last_ball and ball_missing<=20:
            px_f,py_f = ball_kalman.predict()
            sx,sy     = ema.smooth("ball",px_f,py_f)
            ball = {"center":last_ball.get("center"),
                    "field_pos":[round(sx,2),round(sy,2)],
                    "conf":0.0,"interpolated":True}
    else:
        ball_missing = 0
        last_ball    = ball.copy()

    return players, ball, last_ball, ball_missing


def annotate(frame, players, ball, frame_idx):
    out = frame.copy()
    for p in players:
        x1,y1,x2,y2=p["bbox"]
        if p["team"]=="ref": col=REF_COL
        elif p["team"]=="A": col=TEAM_A_COL
        else:                col=TEAM_B_COL
        ov=out.copy(); cv2.rectangle(ov,(x1,y1),(x2,y2),col,-1)
        cv2.addWeighted(ov,0.18,out,0.82,0,out)
        cv2.rectangle(out,(x1,y1),(x2,y2),col,2)
        cv2.putText(out,p["team"],(x1+3,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.45,col,1)

    if ball and ball.get("center"):
        cx,cy=ball["center"]
        c=(150,150,150) if ball.get("interpolated") else BALL_COL
        cv2.circle(out,(cx,cy),8,c,-1); cv2.circle(out,(cx,cy),8,(0,0,0),1)

    n_a=sum(1 for p in players if p["team"]=="A")
    n_b=sum(1 for p in players if p["team"]=="B")
    cv2.putText(out,f"Frame {frame_idx} | A:{n_a} B:{n_b}",(10,40),
        cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,255,255),2)
    return out


def run(clip_path, model_path, calib_path, conf=0.3, max_frames=None):
    calib       = CalibrationData(calib_path)
    model       = YOLO(model_path)
    team_tracker = StableTeamTracker()
    ball_kalman = BallKalman()
    ema         = EMAPosition(alpha=0.20, buffer_size=5, min_move=0.4)

    cap    = cv2.VideoCapture(str(clip_path))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n{'='*50}")
    print(f"  FOOTBALL TRACKER — YOLOv9e")
    print(f"  {width}x{height} @ {fps:.1f}fps | {total} Frames")
    print(f"{'='*50}")

    out_path = OUTPUT_DIR/f"track_{Path(clip_path).stem}.mp4"
    fourcc   = cv2.VideoWriter_fourcc(*"avc1")
    writer   = cv2.VideoWriter(str(out_path),fourcc,fps,(width,height))
    if not writer.isOpened():
        fourcc=cv2.VideoWriter_fourcc(*"mp4v")
        writer=cv2.VideoWriter(str(out_path),fourcc,fps,(width,height))

    all_frames   = []
    frame_idx    = 0
    last_ball    = None
    ball_missing = 0

    print("  Detection läuft...")

    while True:
        ret,frame=cap.read()
        if not ret: break
        if max_frames and frame_idx>=max_frames: break

        players,ball,last_ball,ball_missing = process_frame(
            frame,model,calib,team_tracker,ball_kalman,ema,
            conf,last_ball,ball_missing
        )

        fd = {"frame":frame_idx,"players":players,"ball":ball}
        all_frames.append(fd)
        writer.write(annotate(frame,players,ball,frame_idx))
        frame_idx+=1

        if frame_idx%30==0:
            n_a=sum(1 for p in players if p["team"]=="A")
            n_b=sum(1 for p in players if p["team"]=="B")
            print(f"    Frame {frame_idx} | A:{n_a} B:{n_b} | "
                  f"Ball:{'ja' if ball and not ball.get('interpolated') else 'interp' if ball else 'nein'}")

    cap.release(); writer.release()

    json_path=OUTPUT_DIR/f"track_{Path(clip_path).stem}.json"
    with open(json_path,"w") as f: json.dump(all_frames,f)

    ball_det=sum(1 for fd in all_frames if fd["ball"] and not fd["ball"].get("interpolated"))
    avg_pl  =np.mean([len(fd["players"]) for fd in all_frames])

    print(f"\n{'='*50}")
    print(f"  FERTIG")
    print(f"  Frames:        {frame_idx}")
    print(f"  Spieler Ø:     {avg_pl:.1f}/Frame")
    print(f"  Ball erkannt:  {ball_det} ({ball_det/max(frame_idx,1)*100:.0f}%)")
    print(f"  Output:        {out_path.name}")
    print(f"{'='*50}\n")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--clip",        type=str, required=True)
    parser.add_argument("--calibration", type=str, required=True)
    parser.add_argument("--model",       type=str, default=None)
    parser.add_argument("--conf",        type=float, default=0.3)
    parser.add_argument("--frames",      type=int,   default=None)
    args=parser.parse_args()

    clip_path=Path(args.clip)
    if not clip_path.exists(): clip_path=CLIPS_DIR/args.clip
    if not clip_path.exists(): print("❌ Clip nicht gefunden"); return

    calib_path=Path(args.calibration)
    if not calib_path.exists(): calib_path=OUTPUT_DIR/args.calibration
    if not calib_path.exists(): print("❌ Kalibrierung nicht gefunden"); return

    if args.model: model_path=Path(args.model)
    else:
        models=list(MODELS_DIR.glob("*.pt"))
        if not models: print("❌ Kein Modell!"); return
        model_path=sorted(models)[-1]; print(f"  Modell: {model_path.name}")

    run(clip_path,model_path,calib_path,args.conf,args.frames)


if __name__=="__main__":
    main()