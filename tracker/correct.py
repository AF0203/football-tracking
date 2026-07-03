# tracker/correct.py
"""
Semi-automatisches Tracking — vereinfachter Ansatz
→ Frame 0: alle Spieler manuell markieren + Teams
→ max 22 Spieler werden getrackt
→ kein Kalman/Predict — nur ByteTrack
→ verlorene Spieler verschwinden sauber

Steuerung Frame 0:
  A = Team A    B = Team B
  R = Schiri    L = Ball
  H = Feldlinie neu
  ENTER = Tracking starten
  ESC = abbrechen

Aufruf:
    python3 tracker/correct.py \
        --clip clips/test.mp4 \
        --calibration output/calibration_test.json \
        --interval 100
"""

import sys, os, argparse, json
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import supervision as sv
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import MODELS_DIR, CLIPS_DIR, OUTPUT_DIR

FIELD_WIDTH  = 68.0
FIELD_LENGTH = 105.0

BG         = (13,13,13)
FIELD_GREEN= (22,75,22)
FIELD_LINE = (180,180,180)
TEAM_A_COL = (255,80,80)
TEAM_B_COL = (80,120,255)
BALL_COL   = (255,255,255)
REF_COL    = (200,200,80)
IGNORE_COL = (80,80,80)
TEXT_COLOR = (220,220,220)
NEW_COL    = (0,255,150)

CLASS_NAMES= {0:"ball",1:"goalkeeper",2:"player",3:"referee"}

MAX_PLAYERS = 22


def draw_field(w=680, h=460):
    img=np.zeros((h,w,3),dtype=np.uint8); img[:]=FIELD_GREEN
    px_=int(w*0.06); py_=int(h*0.06); fw=w-2*px_; fh=h-2*py_
    def px(x): return int(px_+(x/FIELD_LENGTH)*fw)
    def py(y): return int(py_+(y/FIELD_WIDTH)*fh)
    lw=2
    cv2.rectangle(img,(px(0),py(0)),(px(FIELD_LENGTH),py(FIELD_WIDTH)),FIELD_LINE,lw)
    cv2.line(img,(px(FIELD_LENGTH/2),py(0)),(px(FIELD_LENGTH/2),py(FIELD_WIDTH)),FIELD_LINE,lw)
    r=int((9.15/FIELD_LENGTH)*fw)
    cv2.circle(img,(px(FIELD_LENGTH/2),py(FIELD_WIDTH/2)),r,FIELD_LINE,lw)
    cv2.circle(img,(px(FIELD_LENGTH/2),py(FIELD_WIDTH/2)),4,FIELD_LINE,-1)
    for sx,bx in [(0,16.5),(FIELD_LENGTH,FIELD_LENGTH-16.5)]:
        cv2.rectangle(img,(px(min(sx,bx)),py(13.85)),(px(max(sx,bx)),py(FIELD_WIDTH-13.85)),FIELD_LINE,lw)
        xs=px(min(sx,sx+(5.5 if sx==0 else -5.5))); xe=px(max(sx,sx+(5.5 if sx==0 else -5.5)))
        cv2.rectangle(img,(xs,py(24.85)),(xe,py(FIELD_WIDTH-24.85)),FIELD_LINE,lw)
        ep=11 if sx==0 else FIELD_LENGTH-11
        cv2.circle(img,(px(ep),py(FIELD_WIDTH/2)),3,FIELD_LINE,-1)
    return img, px, py


class CalibrationData:
    def __init__(self, path):
        with open(path) as f: data=json.load(f)
        self.H   =np.array(data["H"],    dtype=np.float32)
        self.H_inv=np.array(data["H_inv"],dtype=np.float32)
        self.field_w=data["field_w"]; self.field_h=data["field_h"]
        self.team_a_color=data["team_a_color"]; self.team_b_color=data["team_b_color"]
        self.ignored_positions=data.get("ignored_positions",[])
        self.ca=np.array(self.team_a_color,dtype=np.float32)
        self.cb=np.array(self.team_b_color,dtype=np.float32)

    def pixel_to_field(self, px, py):
        pt=np.array([[[float(px),float(py)]]],dtype=np.float32)
        r=cv2.perspectiveTransform(pt,self.H); fx,fy=r[0][0]
        return round(float((fx/self.field_w)*FIELD_LENGTH),2), \
               round(float((fy/self.field_h)*FIELD_WIDTH),2)

    def is_in_field(self, fx, fy, margin=20.0):
        return -margin<=fx<=FIELD_LENGTH+margin and -margin<=fy<=FIELD_WIDTH+margin

    def is_ignored(self, px, py, threshold=100):
        return any(((px-ip[0])**2+(py-ip[1])**2)**0.5<threshold for ip in self.ignored_positions)

    def update_homography(self, vpts, fpts):
        src=np.array(vpts,dtype=np.float32); dst=np.array(fpts,dtype=np.float32)
        H,mask=cv2.findHomography(src,dst,cv2.RANSAC,5.0)
        if H is not None:
            self.H=H; self.H_inv=np.linalg.inv(H)
            print(f"  ✅ Homographie aktualisiert ({mask.sum()} Inlier)")


# ══════════════════════════════════════════════════════════════════
#  FRAME 0 SETUP INTERFACE
# ══════════════════════════════════════════════════════════════════

class SetupUI:
    """
    Frame 0 Interface:
    → zeigt automatisch erkannte Spieler
    → User weist Teams zu / korrigiert
    → User kann Spieler hinzufügen (N)
    → User kann Feldlinie neu setzen (H)
    → ENTER → gibt team_map zurück
    """
    def __init__(self, frame, detections, calib,
                 display_w=1500, display_h=520):
        self.frame=frame; self.dets=detections; self.calib=calib
        self.display_w=display_w; self.display_h=display_h
        self.half_w=display_w//2

        scale=min(self.half_w/frame.shape[1], display_h/frame.shape[0])
        self.vscale=scale
        self.vw=int(frame.shape[1]*scale); self.vh=int(frame.shape[0]*scale)
        self.vy_off=(display_h-self.vh)//2

        self.field_img,self.fpx,self.fpy=draw_field(display_w-self.half_w, display_h)

        # Team-Zuweisungen
        self.team_map={}  # det_id → "A"/"B"/"ref"/"ignore"
        for det in detections:
            if det.get("team"): self.team_map[det["id"]]=det["team"]

        # Manuelle neue Spieler
        self.manual_players=[]  # [{id, team, center, field_pos}]
        self.next_id=500

        # Feldlinie
        self.field_vpts=[]; self.field_fpts=[]; self.field_step="video"

        self.mode="A"  # A, B, ref, ignore, new, field
        self.new_team="A"
        self.window="Setup Frame 0 | A/B/R/I/N/H/Z | ENTER=starten"

    def find_nearest_det(self, ox, oy, max_d=80):
        best=None; best_d=999999
        for det in self.dets:
            if det.get("class")=="ball": continue
            cx,cy=det["center"]
            d=((cx-ox)**2+(cy-oy)**2)**0.5
            if d<best_d: best_d=d; best=det
        for mp in self.manual_players:
            cx,cy=mp["center"]
            d=((cx-ox)**2+(cy-oy)**2)**0.5
            if d<best_d: best_d=d; best=mp
        return best if best and best_d<max_d/self.vscale else None

    def find_nearest_on_field(self, fx, fy, max_d=40):
        """Nächsten Spieler auf 2D Feld finden"""
        best=None; best_d=999999
        all_players=list(self.dets)+self.manual_players
        for p in all_players:
            if p.get("class")=="ball": continue
            if "field_pos" not in p: continue
            pfx,pfy=p["field_pos"]
            dx=self.fpx(pfx); dy=self.fpy(pfy)
            d=((dx-fx)**2+(dy-fy)**2)**0.5
            if d<best_d: best_d=d; best=p
        return best if best and best_d<max_d else None

    def mouse_callback(self, event, x, y, flags, param):
        if event!=cv2.EVENT_LBUTTONDOWN: return
        vy=y-self.vy_off

        if x<self.half_w:
            # Video-Seite
            ox=x/self.vscale; oy=vy/self.vscale

            if self.mode=="field":
                self.field_vpts.append((ox,oy)); self.field_step="2d"
                print(f"  Video-Punkt → jetzt 2D klicken")

            elif self.mode=="new":
                if len(self.team_map)+len(self.manual_players) >= MAX_PLAYERS:
                    print(f"  ⚠️  Max {MAX_PLAYERS} Spieler erreicht")
                    return
                fm_x,fm_y=self.calib.pixel_to_field(ox,oy)
                mp={"id":self.next_id,"class":"player","team":self.new_team,
                    "center":[int(ox),int(oy)],"field_pos":[fm_x,fm_y]}
                self.manual_players.append(mp)
                self.team_map[self.next_id]=self.new_team
                self.next_id+=1
                print(f"  Neuer Spieler #{mp['id']} Team {self.new_team}: {fm_x:.1f},{fm_y:.1f}m")

            else:
                det=self.find_nearest_det(ox,oy)
                if det:
                    tid=det["id"]
                    if self.mode=="ignore":
                        self.team_map[tid]="ignore"
                        print(f"  #{tid} → ignoriert")
                    elif self.mode=="ref":
                        self.team_map[tid]="ref"
                        print(f"  #{tid} → Schiri")
                    else:
                        self.team_map[tid]=self.mode
                        print(f"  #{tid} → Team {self.mode}")

        else:
            # 2D Feld
            fx=x-self.half_w; fy=y

            if self.mode=="field" and self.field_step=="2d":
                self.field_fpts.append((fx,fy)); self.field_step="video"
                print(f"  Feld-Punkt ({len(self.field_vpts)} Paare)")
                if len(self.field_vpts)>=4 and len(self.field_vpts)==len(self.field_fpts):
                    self.calib.update_homography(self.field_vpts,self.field_fpts)
                    # field_pos neu berechnen
                    for det in self.dets:
                        if det.get("foot"):
                            det["field_pos"]=list(self.calib.pixel_to_field(*det["foot"]))

            elif self.mode=="ignore":
                # Spieler auf 2D Feld ignorieren
                p=self.find_nearest_on_field(fx,fy)
                if p:
                    self.team_map[p["id"]]="ignore"
                    print(f"  #{p['id']} → ignoriert (2D Klick)")

    def render(self):
        canvas=np.zeros((self.display_h,self.display_w,3),dtype=np.uint8)
        canvas[:]=BG

        # Video
        vf=cv2.resize(self.frame,(self.vw,self.vh))
        for det in self.dets:
            if det.get("class")=="ball": continue
            x1,y1,x2,y2=det["bbox"]
            sx1=int(x1*self.vscale); sy1=int(y1*self.vscale)+self.vy_off
            sx2=int(x2*self.vscale); sy2=int(y2*self.vscale)+self.vy_off
            scx=int(det["center"][0]*self.vscale); scy=int(det["center"][1]*self.vscale)+self.vy_off
            tid=det["id"]; team=self.team_map.get(tid,"?")
            if team=="ignore": col=IGNORE_COL
            elif team=="A":    col=TEAM_A_COL
            elif team=="B":    col=TEAM_B_COL
            elif team=="ref":  col=REF_COL
            else:              col=(150,150,150)
            ov=vf.copy(); cv2.rectangle(ov,(sx1,sy1),(sx2,sy2),col,-1)
            cv2.addWeighted(ov,0.2,vf,0.8,0,vf)
            cv2.rectangle(vf,(sx1,sy1),(sx2,sy2),col,2)
            cv2.putText(vf,f"#{tid}{team}",(sx1+2,sy1-5),cv2.FONT_HERSHEY_SIMPLEX,0.4,col,1)

        # Manuelle Spieler im Video
        for mp in self.manual_players:
            cx,cy=mp["center"]
            scx=int(cx*self.vscale); scy=int(cy*self.vscale)+self.vy_off
            col=TEAM_A_COL if mp["team"]=="A" else TEAM_B_COL
            cv2.circle(vf,(scx,scy),12,col,-1)
            cv2.circle(vf,(scx,scy),12,(255,255,255),2)
            cv2.putText(vf,f"#{mp['id']}N",(scx+5,scy-5),cv2.FONT_HERSHEY_SIMPLEX,0.4,col,1)

        canvas[self.vy_off:self.vy_off+self.vh,0:self.vw]=vf

        # 2D Feld
        field=self.field_img.copy()
        for det in self.dets:
            if det.get("class")=="ball": continue
            if "field_pos" not in det: continue
            fm_x,fm_y=det["field_pos"]; dx=self.fpx(fm_x); dy=self.fpy(fm_y)
            tid=det["id"]; team=self.team_map.get(tid,"?")
            if team=="ignore": col=IGNORE_COL
            elif team=="A":    col=TEAM_A_COL
            elif team=="B":    col=TEAM_B_COL
            elif team=="ref":  col=REF_COL
            else:              col=(150,150,150)
            if team!="ignore":
                cv2.circle(field,(dx,dy),9,col,-1)
                cv2.circle(field,(dx,dy),9,(0,0,0),1)
                cv2.putText(field,str(tid),(dx-4,dy+4),cv2.FONT_HERSHEY_SIMPLEX,0.35,(0,0,0),1)

        for mp in self.manual_players:
            if self.team_map.get(mp["id"])=="ignore": continue
            fm_x,fm_y=mp["field_pos"]; dx=self.fpx(fm_x); dy=self.fpy(fm_y)
            col=TEAM_A_COL if mp["team"]=="A" else TEAM_B_COL
            cv2.circle(field,(dx,dy),9,col,-1)
            cv2.circle(field,(dx,dy),9,(255,255,255),2)

        canvas[:,self.half_w:]=field
        cv2.line(canvas,(self.half_w,0),(self.half_w,self.display_h),(60,60,60),2)

        # Header
        n_a=sum(1 for t in self.team_map.values() if t=="A")
        n_b=sum(1 for t in self.team_map.values() if t=="B")
        n_ign=sum(1 for t in self.team_map.values() if t=="ignore")
        col_m=TEAM_A_COL if self.mode=="A" else TEAM_B_COL if self.mode=="B" else TEXT_COLOR
        cv2.putText(canvas,f"SETUP | Modus: {self.mode} | A:{n_a} B:{n_b} Ign:{n_ign} | Max:{MAX_PLAYERS}",
            (10,28),cv2.FONT_HERSHEY_SIMPLEX,0.65,col_m,2)

        if self.mode=="new":
            cv2.putText(canvas,f"Neuer Spieler Team {self.new_team} (a/b wechseln) | Klick im Video",
                (10,55),cv2.FONT_HERSHEY_SIMPLEX,0.55,NEW_COL,1)
        if self.mode=="field":
            cv2.putText(canvas,f"Feldlinie: {'→ Video' if self.field_step=='video' else '→ 2D Feld'} | {len(self.field_vpts)} Paare",
                (10,55),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,220,0),1)

        keys=[("A","A",TEAM_A_COL),("B","B",TEAM_B_COL),("R","Schiri",REF_COL),
              ("N","Neu",NEW_COL),("I","Ign.",IGNORE_COL),("H","Feld",(0,220,0)),("Z","Undo",(200,200,200))]
        for i,(k,lbl,c) in enumerate(keys):
            active=(self.mode==k.lower()) or (self.mode=="A" and k=="A") or (self.mode=="B" and k=="B")
            cv2.putText(canvas,f"[{k}]{lbl}",(10+i*185,self.display_h-12),
                cv2.FONT_HERSHEY_SIMPLEX,0.45,c if active else (80,80,80),1)
        cv2.putText(canvas,"ENTER=Tracking starten | ESC=abbrechen",
            (self.display_w-380,self.display_h-12),cv2.FONT_HERSHEY_SIMPLEX,0.45,TEXT_COLOR,1)

        return canvas

    def run(self):
        print(f"\n  SETUP Frame 0:")
        print(f"  A/B = Team zuweisen | N = Neuer Spieler")
        print(f"  I = Ignorieren | H = Feldlinie | Z = Undo")
        print(f"  ENTER wenn fertig")

        cv2.namedWindow(self.window,cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window,self.display_w,self.display_h)
        cv2.setMouseCallback(self.window,self.mouse_callback)

        while True:
            cv2.imshow(self.window,self.render())
            key=cv2.waitKey(1)&0xFF

            if key==ord('a') or key==ord('A'):
                if self.mode=="new": self.new_team="A"
                else: self.mode="A"
            elif key==ord('b') or key==ord('B'):
                if self.mode=="new": self.new_team="B"
                else: self.mode="B"
            elif key==ord('r') or key==ord('R'): self.mode="ref"
            elif key==ord('n') or key==ord('N'): self.mode="new"; self.new_team="A"
            elif key==ord('i') or key==ord('I'): self.mode="ignore"
            elif key==ord('h') or key==ord('H'):
                self.mode="field"; self.field_vpts=[]; self.field_fpts=[]
            elif key==ord('z') or key==ord('Z'):
                if self.manual_players:
                    mp=self.manual_players.pop()
                    self.team_map.pop(mp["id"],None)
                    print(f"  Spieler #{mp['id']} entfernt")
            elif key==13: break
            elif key==27: cv2.destroyWindow(self.window); return None,None

        cv2.destroyWindow(self.window)
        return self.team_map, self.manual_players


# ══════════════════════════════════════════════════════════════════
#  TRACKER
# ══════════════════════════════════════════════════════════════════

class SimpleTracker:
    """
    Einfacher Tracker ohne Kalman/Predict
    → nur ByteTrack
    → max MAX_PLAYERS IDs
    → verlorene Spieler verschwinden sauber
    """
    def __init__(self, model_path, calib, conf=0.3):
        self.model=YOLO(model_path); self.calib=calib; self.conf=conf

        self.player_tracker=sv.ByteTrack(
            track_activation_threshold=conf,
            lost_track_buffer=30,  # kürzer — sauberer verlieren
            minimum_matching_threshold=0.8,
            frame_rate=25,
        )
        self.ball_tracker=sv.ByteTrack(
            track_activation_threshold=conf,
            lost_track_buffer=20,
            minimum_matching_threshold=0.5,
            frame_rate=25,
        )

        # Aus Frame 0 Setup
        self.team_map={}      # tracker_id → "A"/"B"/"ref"/"ignore"
        self.allowed_ids=set()  # nur diese IDs werden gezeigt
        self.perm_ignored=set()

        # EMA nur für Smoothing
        self.ema={}; self.EMA=0.45

        self.last_ball=None; self.ball_missing=0; self.MAX_BALL=20

    def setup_from_frame0(self, team_map, allowed_tracker_ids, perm_ignored):
        self.team_map=dict(team_map)
        self.allowed_ids=set(allowed_tracker_ids)
        self.perm_ignored=set(perm_ignored)

    def _smooth(self, tid, x, y):
        if tid not in self.ema: self.ema[tid]=(x,y); return x,y
        ox,oy=self.ema[tid]; nx=self.EMA*x+(1-self.EMA)*ox; ny=self.EMA*y+(1-self.EMA)*oy
        self.ema[tid]=(nx,ny); return nx,ny

    def process(self, frame, frame_idx):
        scale=min(1280/frame.shape[1],1280/frame.shape[0])
        small=cv2.resize(frame,(int(frame.shape[1]*scale),int(frame.shape[0]*scale))) if scale<1.0 else frame
        results=self.model(small,conf=self.conf,iou=0.6,verbose=False)[0]
        dets=sv.Detections.from_ultralytics(results)
        if scale<1.0: dets.xyxy=dets.xyxy/scale

        pmask=np.isin(dets.class_id,[1,2]); p_dets=dets[pmask]
        p_dets=self.player_tracker.update_with_detections(p_dets)

        players=[]
        for i in range(len(p_dets)):
            x1,y1,x2,y2=map(int,p_dets.xyxy[i])
            tid=int(p_dets.tracker_id[i]) if p_dets.tracker_id is not None else i
            cls_id=int(p_dets.class_id[i]); cx=(x1+x2)//2; cy=(y1+y2)//2

            # Nur erlaubte IDs
            if tid in self.perm_ignored: continue
            if self.calib.is_ignored(cx,cy): continue

            # Team aus Setup
            team=self.team_map.get(tid)
            if team=="ignore": continue
            if team is None:
                # Neue ID nach Frame 0 — ignorieren
                continue

            fm_x,fm_y=self.calib.pixel_to_field((x1+x2)//2,y2)
            if not self.calib.is_in_field(fm_x,fm_y): continue

            sx,sy=self._smooth(tid,fm_x,fm_y)
            players.append({
                "id":tid,"class":CLASS_NAMES.get(cls_id,"player"),
                "team":team,"bbox":[x1,y1,x2,y2],
                "center":[cx,cy],"foot":[(x1+x2)//2,y2],
                "field_pos":[round(sx,2),round(sy,2)],
                "conf":round(float(p_dets.confidence[i]),3),
            })

        # Ball
        bmask=dets.class_id==0; b_dets=dets[bmask]
        b_dets=self.ball_tracker.update_with_detections(b_dets)
        ball=None
        if len(b_dets)>0:
            x1,y1,x2,y2=map(int,b_dets.xyxy[0]); cx=(x1+x2)//2; cy=(y1+y2)//2
            fm_x,fm_y=self.calib.pixel_to_field(cx,cy)
            ball={"center":[cx,cy],"field_pos":[fm_x,fm_y],
                  "conf":round(float(b_dets.confidence[0]),3),"interpolated":False}
            self.last_ball=ball.copy(); self.ball_missing=0
        else:
            self.ball_missing+=1
            if self.last_ball and self.ball_missing<=self.MAX_BALL:
                ball=self.last_ball.copy(); ball["interpolated"]=True; ball["conf"]=0.0

        return {"frame":frame_idx,"players":players,"ball":ball}


# ══════════════════════════════════════════════════════════════════
#  ANNOTATE + SAVE
# ══════════════════════════════════════════════════════════════════

def annotate(frame, fd):
    out=frame.copy()
    for p in fd["players"]:
        if p.get("bbox") is None: continue
        x1,y1,x2,y2=p["bbox"]
        col=TEAM_A_COL if p["team"]=="A" else TEAM_B_COL if p["team"]=="B" else REF_COL
        ov=out.copy(); cv2.rectangle(ov,(x1,y1),(x2,y2),col,-1)
        cv2.addWeighted(ov,0.2,out,0.8,0,out)
        cv2.rectangle(out,(x1,y1),(x2,y2),col,2)
        cv2.putText(out,f"#{p['id']}{p['team']}",(x1+2,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.4,col,1)
    if fd["ball"] and fd["ball"].get("center"):
        cx,cy=fd["ball"]["center"]
        c=(150,150,150) if fd["ball"].get("interpolated") else (255,255,255)
        cv2.circle(out,(cx,cy),8,c,-1)
    n_a=sum(1 for p in fd["players"] if p.get("team")=="A")
    n_b=sum(1 for p in fd["players"] if p.get("team")=="B")
    cv2.putText(out,f"Frame {fd['frame']} | A:{n_a} B:{n_b}",(10,40),cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,255,255),2)
    return out


def save_json(all_frames, clip_path):
    json_path=OUTPUT_DIR/f"track_{clip_path.stem}.json"
    with open(json_path,"w") as f: json.dump(all_frames,f)
    ball_det=sum(1 for fd in all_frames if fd["ball"] and not fd["ball"].get("interpolated"))
    print(f"\n{'='*50}")
    print(f"  FERTIG: {len(all_frames)} Frames")
    print(f"  Ball erkannt: {ball_det} ({ball_det/max(len(all_frames),1)*100:.0f}%)")
    print(f"  JSON: {json_path.name}")
    print(f"{'='*50}\n")


# ══════════════════════════════════════════════════════════════════
#  HAUPT-PIPELINE
# ══════════════════════════════════════════════════════════════════

def run(clip_path, model_path, calib_path, conf=0.3, max_frames=None):
    calib=CalibrationData(calib_path)
    model=YOLO(model_path)

    cap=cv2.VideoCapture(str(clip_path))
    fps=cap.get(cv2.CAP_PROP_FPS); width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n{'='*55}\n  FOOTBALL TRACKER\n  {width}x{height} @ {fps:.1f}fps | {total} Frames\n{'='*55}")

    # ── Frame 0 Detection + Setup ─────────────────────────────────
    ret,frame0=cap.read()
    if not ret: print("❌ Kein Frame"); return

    print(f"  Erkenne Spieler in Frame 0...")
    scale=min(1280/frame0.shape[1],1280/frame0.shape[0])
    small=cv2.resize(frame0,(int(frame0.shape[1]*scale),int(frame0.shape[0]*scale))) if scale<1.0 else frame0
    results=model(small,conf=conf,iou=0.6,verbose=False)[0]

    dets=[]
    for i,box in enumerate(results.boxes):
        cls_id=int(box.cls[0])
        if cls_id==0: continue  # Ball separat
        x1,y1,x2,y2=map(int,box.xyxy[0])
        if scale<1.0: x1=int(x1/scale);y1=int(y1/scale);x2=int(x2/scale);y2=int(y2/scale)
        if (x2-x1)*(y2-y1)<500: continue
        cx=(x1+x2)//2; cy=(y1+y2)//2
        fm_x,fm_y=calib.pixel_to_field((x1+x2)//2,y2)
        if not calib.is_in_field(fm_x,fm_y): continue
        if calib.is_ignored(cx,cy): continue

        # Auto Team
        px_=(x2-x1)//4; py_=(y2-y1)//4
        roi=frame0[max(0,y1+py_):min(height-1,y2-py_),max(0,x1+px_):min(width-1,x2-px_)]
        avg=np.array(roi.mean(axis=(0,1)),dtype=np.float32) if roi.size>0 else np.array([128,128,128],dtype=np.float32)
        team="A" if np.linalg.norm(avg-calib.ca)<np.linalg.norm(avg-calib.cb) else "B"

        dets.append({"id":i,"class":CLASS_NAMES.get(cls_id,"player"),"bbox":[x1,y1,x2,y2],
                     "center":[cx,cy],"foot":[(x1+x2)//2,y2],"field_pos":[fm_x,fm_y],"team":team})

    print(f"  {len(dets)} Spieler erkannt")

    # Setup UI
    setup_ui=SetupUI(frame0,dets,calib)
    team_map,manual_players=setup_ui.run()
    if team_map is None: cap.release(); return

    print(f"\n  Setup abgeschlossen:")
    n_a=sum(1 for t in team_map.values() if t=="A")
    n_b=sum(1 for t in team_map.values() if t=="B")
    n_ign=sum(1 for t in team_map.values() if t=="ignore")
    print(f"  Team A: {n_a} | Team B: {n_b} | Ignoriert: {n_ign}")

    # Tracker initialisieren
    tracker=SimpleTracker(model_path,calib,conf)

    # Erlaubte IDs: alle aus Setup die nicht ignoriert sind
    # ByteTrack muss erst ein Frame sehen um IDs zu vergeben
    # Wir lassen Frame 0 nochmal durch ByteTrack laufen
    # um die echten tracker_ids zu bekommen

    # Dummy-Frame durch tracker schicken um IDs zu initialisieren
    sv_dets=sv.Detections(
        xyxy=np.array([[d["bbox"][0],d["bbox"][1],d["bbox"][2],d["bbox"][3]] for d in dets if team_map.get(d["id"])!="ignore"],dtype=np.float32),
        class_id=np.array([2]*sum(1 for d in dets if team_map.get(d["id"])!="ignore")),
        confidence=np.array([0.9]*sum(1 for d in dets if team_map.get(d["id"])!="ignore"),dtype=np.float32),
    )
    if len(sv_dets)>0:
        tracked=tracker.player_tracker.update_with_detections(sv_dets)
        # Mappe Setup-IDs zu Tracker-IDs per nächstem Nachbar
        if tracked.tracker_id is not None:
            valid_dets=[d for d in dets if team_map.get(d["id"])!="ignore"]
            for i,tid in enumerate(tracked.tracker_id):
                if i<len(valid_dets):
                    setup_id=valid_dets[i]["id"]
                    team=team_map.get(setup_id,"A")
                    tracker.team_map[int(tid)]=team
                    tracker.allowed_ids.add(int(tid))

    # Ignorierte
    perm_ign=set(d["id"] for d in dets if team_map.get(d["id"])=="ignore")
    tracker.perm_ignored=perm_ign

    # Output
    out_video=OUTPUT_DIR/f"track_{clip_path.stem}.mp4"
    fourcc=cv2.VideoWriter_fourcc(*"avc1")
    writer=cv2.VideoWriter(str(out_video),fourcc,fps,(width,height))
    if not writer.isOpened():
        fourcc=cv2.VideoWriter_fourcc(*"mp4v")
        writer=cv2.VideoWriter(str(out_video),fourcc,fps,(width,height))

    # Frame 0 speichern
    fd0={"frame":0,"players":[],"ball":None}
    for d in dets:
        if team_map.get(d["id"]) in ["ignore",None]: continue
        fd0["players"].append({
            "id":d["id"],"class":d["class"],"team":team_map[d["id"]],
            "bbox":d["bbox"],"center":d["center"],"foot":d["foot"],
            "field_pos":d["field_pos"],"conf":0.9,
        })
    # Manuelle Spieler
    for mp in manual_players:
        if team_map.get(mp["id"])=="ignore": continue
        fd0["players"].append({**mp,"conf":1.0})
        tracker.team_map[mp["id"]]=mp["team"]
        tracker.allowed_ids.add(mp["id"])

    all_frames=[fd0]
    writer.write(annotate(frame0,fd0))

    print(f"  Tracking läuft...")
    frame_idx=1

    while True:
        ret,frame=cap.read()
        if not ret: break
        if max_frames and frame_idx>=max_frames: break

        fd=tracker.process(frame,frame_idx)

        # Manuelle Spieler aus Frame 0 weiterpropagieren per Kalman
        # (nur wenn sie vom Tracker verloren wurden)
        tracked_ids=set(p["id"] for p in fd["players"])
        for mp in manual_players:
            if mp["id"] in tracked_ids: continue
            if team_map.get(mp["id"])=="ignore": continue
            # Zeige letzten bekannten Standort
            if mp["id"] in tracker.ema:
                lx,ly=tracker.ema[mp["id"]]
                fd["players"].append({
                    "id":mp["id"],"class":"player","team":mp["team"],
                    "bbox":None,"center":None,"foot":None,
                    "field_pos":[round(lx,2),round(ly,2)],
                    "conf":0.0,"lost":True,
                })

        all_frames.append(fd)
        writer.write(annotate(frame,fd))
        frame_idx+=1

        if frame_idx%30==0:
            n_a=sum(1 for p in fd["players"] if p.get("team")=="A")
            n_b=sum(1 for p in fd["players"] if p.get("team")=="B")
            print(f"    Frame {frame_idx} | A:{n_a} B:{n_b} | Ball:{'ja' if fd['ball'] else 'nein'}")

    cap.release(); writer.release()
    save_json(all_frames,clip_path)


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