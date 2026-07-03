# tracker/calibrate.py
"""
Optimierte Kalibrierung mit automatischer Feldlinien-Erkennung
→ Hough Transform für Linienerkennung
→ Automatische Eckpunkt-Vorschläge
→ Manuelle Korrektur-Interface
→ 8+ Punkt Homographie

Aufruf:
    python3 tracker/calibrate.py --clip clips/test.mp4 --model models/football_v3.pt
"""

import sys, os, argparse, json
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import MODELS_DIR, CLIPS_DIR, OUTPUT_DIR

FIELD_WIDTH  = 68.0
FIELD_LENGTH = 105.0

BG          = (13, 13, 13)
FIELD_GREEN = (22, 75, 22)
FIELD_LINE  = (180, 180, 180)
TEAM_A_COL  = (255, 80, 80)
TEAM_B_COL  = (80, 120, 255)
IGNORE_COL  = (80, 80, 80)
TEXT_COLOR  = (220, 220, 220)
CYAN        = (255, 200, 0)
MAGENTA     = (180, 50, 200)
POINT_COLORS = [
    (255,80,80),(80,255,80),(80,80,255),(255,255,80),
    (255,80,255),(80,255,255),(255,160,80),(160,80,255),
    (200,255,80),(80,200,255),(255,80,200),(200,80,255),
]
CLASS_NAMES = {0:"ball", 1:"goalkeeper", 2:"player", 3:"referee"}


# ══════════════════════════════════════════════════════════════════
#  2D SPIELFELD
# ══════════════════════════════════════════════════════════════════

def draw_2d_field(w=650, h=450):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = FIELD_GREEN
    px_ = int(w*0.06); py_ = int(h*0.06)
    fw  = w - 2*px_;   fh  = h - 2*py_

    def px(x): return int(px_ + (x/FIELD_LENGTH)*fw)
    def py(y): return int(py_ + (y/FIELD_WIDTH)*fh)

    lw = 2
    cv2.rectangle(img,(px(0),py(0)),(px(FIELD_LENGTH),py(FIELD_WIDTH)),FIELD_LINE,lw)
    cv2.line(img,(px(FIELD_LENGTH/2),py(0)),(px(FIELD_LENGTH/2),py(FIELD_WIDTH)),FIELD_LINE,lw)
    r = int((9.15/FIELD_LENGTH)*fw)
    cv2.circle(img,(px(FIELD_LENGTH/2),py(FIELD_WIDTH/2)),r,FIELD_LINE,lw)
    cv2.circle(img,(px(FIELD_LENGTH/2),py(FIELD_WIDTH/2)),4,FIELD_LINE,-1)
    for sx,bx in [(0,16.5),(FIELD_LENGTH,FIELD_LENGTH-16.5)]:
        cv2.rectangle(img,(px(min(sx,bx)),py(13.85)),(px(max(sx,bx)),py(FIELD_WIDTH-13.85)),FIELD_LINE,lw)
        xs=px(min(sx,sx+(5.5 if sx==0 else -5.5))); xe=px(max(sx,sx+(5.5 if sx==0 else -5.5)))
        cv2.rectangle(img,(xs,py(24.85)),(xe,py(FIELD_WIDTH-24.85)),FIELD_LINE,lw)
        ep=11 if sx==0 else FIELD_LENGTH-11
        cv2.circle(img,(px(ep),py(FIELD_WIDTH/2)),3,FIELD_LINE,-1)
    cv2.circle(img,(px(FIELD_LENGTH/2),py(FIELD_WIDTH/2)),4,(255,255,255),-1)

    # Standard-Feldpunkte für Kalibrierung
    field_points = {
        "TL":  (px(0),           py(0)),
        "TR":  (px(FIELD_LENGTH), py(0)),
        "BL":  (px(0),           py(FIELD_WIDTH)),
        "BR":  (px(FIELD_LENGTH), py(FIELD_WIDTH)),
        "ML":  (px(0),           py(FIELD_WIDTH/2)),
        "MR":  (px(FIELD_LENGTH), py(FIELD_WIDTH/2)),
        "MT":  (px(FIELD_LENGTH/2), py(0)),
        "MB":  (px(FIELD_LENGTH/2), py(FIELD_WIDTH)),
        "MC":  (px(FIELD_LENGTH/2), py(FIELD_WIDTH/2)),
        "PL_TL": (px(0),    py(13.85)),
        "PL_BL": (px(0),    py(FIELD_WIDTH-13.85)),
        "PL_TR": (px(16.5), py(13.85)),
        "PL_BR": (px(16.5), py(FIELD_WIDTH-13.85)),
        "PR_TL": (px(FIELD_LENGTH-16.5), py(13.85)),
        "PR_BL": (px(FIELD_LENGTH-16.5), py(FIELD_WIDTH-13.85)),
        "PR_TR": (px(FIELD_LENGTH), py(13.85)),
        "PR_BR": (px(FIELD_LENGTH), py(FIELD_WIDTH-13.85)),
    }

    return img, px, py, field_points


# ══════════════════════════════════════════════════════════════════
#  AUTOMATISCHE FELDLINIEN-ERKENNUNG
# ══════════════════════════════════════════════════════════════════

def detect_field_lines(frame):
    """
    Erkennt Feldlinien mit Hough Transform
    und schlägt Eckpunkte vor
    """
    h, w = frame.shape[:2]

    # Grün-Maske für Rasen
    hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (35,40,40), (85,255,255))

    # Weiße Linien auf grünem Grund
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, white = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    on_green = cv2.bitwise_and(white, green)

    # Morphologie
    kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    on_green = cv2.morphologyEx(on_green, cv2.MORPH_CLOSE, kernel)

    # Hough Lines
    edges = cv2.Canny(on_green, 50, 150)
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi/180,
        threshold=80, minLineLength=60, maxLineGap=20
    )

    suggested_points = []
    line_img = np.zeros_like(frame)

    if lines is not None:
        # Linien zeichnen
        for line in lines:
            x1,y1,x2,y2 = line[0]
            cv2.line(line_img,(x1,y1),(x2,y2),(0,255,255),2)

        # Schnittpunkte berechnen
        intersections = []
        for i in range(len(lines)):
            for j in range(i+1, len(lines)):
                pt = line_intersection(lines[i][0], lines[j][0])
                if pt and 0<=pt[0]<w and 0<=pt[1]<h:
                    intersections.append(pt)

        # Cluster der Schnittpunkte
        if len(intersections) >= 4:
            pts = np.array(intersections)
            n_clusters = min(12, len(intersections))
            km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            km.fit(pts)
            suggested_points = [(int(c[0]),int(c[1])) for c in km.cluster_centers_]

    return suggested_points, line_img


def line_intersection(line1, line2):
    """Berechnet Schnittpunkt zweier Linien"""
    x1,y1,x2,y2 = line1
    x3,y3,x4,y4 = line2
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(denom) < 1e-10:
        return None
    t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
    x = x1 + t*(x2-x1)
    y = y1 + t*(y2-y1)
    return (int(x), int(y))


# ══════════════════════════════════════════════════════════════════
#  SPIELER-DETECTION MIT K-MEANS TEAM-ZUORDNUNG
# ══════════════════════════════════════════════════════════════════

def get_avg_color(frame, x1, y1, x2, y2):
    """Trikotfarbe aus oberem Bereich der Bounding Box"""
    h = y2 - y1; w = x2 - x1
    top    = max(0,              y1 + int(h*0.10))
    bottom = min(frame.shape[0]-1, y1 + int(h*0.65))
    left   = max(0,              x1 + int(w*0.10))
    right  = min(frame.shape[1]-1, x2 - int(w*0.10))
    roi = frame[top:bottom, left:right]
    if roi.size == 0: return [128,128,128]
    return np.median(roi.reshape(-1,3), axis=0).astype(int).tolist()


def auto_assign_teams(dets):
    """
    K-Means Clustering auf Trikotfarben
    → findet automatisch Team A + B
    → robuster als Referenzfarbe
    """
    players = [d for d in dets if d["class"] in ["player","goalkeeper"]
               and not d["ignored"]]
    if len(players) < 4:
        return dets

    colors = np.array([d["color"] for d in players], dtype=np.float32)
    km     = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = km.fit_predict(colors)

    # Team A = Cluster 0, Team B = Cluster 1
    for i, p in enumerate(players):
        p["team"] = "A" if labels[i] == 0 else "B"

    return dets


def find_best_frame(clip_path, model_path, conf=0.3, n_frames=60):
    """Bester Frame mit meisten Detektionen"""
    model = YOLO(model_path)
    cap   = cv2.VideoCapture(str(clip_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step  = max(1, total//n_frames)

    best_frame = None; best_dets = []; best_count = 0

    print(f"  Suche besten Frame ({n_frames} Kandidaten)...")
    for i in range(n_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i*step)
        ret, frame = cap.read()
        if not ret: break

        scale = min(1280/frame.shape[1], 1280/frame.shape[0])
        small = cv2.resize(frame,(int(frame.shape[1]*scale),int(frame.shape[0]*scale))) if scale<1.0 else frame
        results = model(small, conf=conf, iou=0.6, verbose=False)[0]
        dets_i  = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id == 0: continue
            x1,y1,x2,y2 = map(int,box.xyxy[0])
            if scale<1.0:
                x1=int(x1/scale);y1=int(y1/scale);x2=int(x2/scale);y2=int(y2/scale)
            if (x2-x1)*(y2-y1)<400: continue
            cx=(x1+x2)//2; cy=(y1+y2)//2
            dets_i.append({"cls_id":cls_id,"bbox":[x1,y1,x2,y2],
                           "center":[cx,cy],"color":get_avg_color(frame,x1,y1,x2,y2)})

        if len(dets_i)>best_count:
            best_count=len(dets_i); best_frame=frame.copy(); best_dets=dets_i
            print(f"    Frame {i*step}: {best_count} Objekte")

    cap.release()
    print(f"  Bester Frame: {best_count} Objekte")

    dets = []
    for d in best_dets:
        x1,y1,x2,y2=d["bbox"]; cx=(x1+x2)//2; cy=(y1+y2)//2
        dets.append({"id":len(dets),"class":CLASS_NAMES.get(d["cls_id"],"player"),
                     "bbox":[x1,y1,x2,y2],"center":[cx,cy],"foot":[cx,y2],
                     "color":d["color"],"team":None,"ignored":False})

    # NMS auf Detektionen — überlappende Kästchen entfernen
    dets = nms_dets(dets, iou_threshold=0.5)
    # Automatische Team-Zuweisung per K-Means
    dets = auto_assign_teams(dets)
    return best_frame, dets


def nms_dets(dets, iou_threshold=0.5):
    """Non-Maximum Suppression — entfernt überlappende Kästchen"""
    if len(dets) == 0:
        return dets

    boxes = np.array([d["bbox"] for d in dets], dtype=np.float32)
    scores= np.array([1.0]*len(dets), dtype=np.float32)

    x1 = boxes[:,0]; y1 = boxes[:,1]
    x2 = boxes[:,2]; y2 = boxes[:,3]
    areas = (x2-x1)*(y2-y1)

    order = np.argsort(areas)[::-1]  # größte zuerst
    keep  = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2-xx1)
        h = np.maximum(0, yy2-yy1)
        inter = w*h
        iou   = inter / (areas[i]+areas[order[1:]]-inter+1e-6)

        order = order[1:][iou < iou_threshold]

    return [dets[i] for i in keep]


# ══════════════════════════════════════════════════════════════════
#  KALIBRIERUNGS-INTERFACE
# ══════════════════════════════════════════════════════════════════

class Calibrator:
    def __init__(self, frame, detections,
                 suggested_pts=None, line_img=None,
                 display_w=1500, display_h=520):
        self.frame        = frame
        self.dets         = detections
        self.suggested    = suggested_pts or []
        self.line_img     = line_img
        self.display_w    = display_w
        self.display_h    = display_h
        self.half_w       = display_w // 2

        scale = min(self.half_w/frame.shape[1], display_h/frame.shape[0])
        self.vscale = scale
        self.vw     = int(frame.shape[1]*scale)
        self.vh     = int(frame.shape[0]*scale)
        self.vy_off = (display_h-self.vh)//2

        self.field_img, self.fpx, self.fpy, self.field_pts_map = \
            draw_2d_field(display_w-self.half_w, display_h)
        self.field_w = display_w - self.half_w
        self.field_h = display_h

        self.step       = 1  # 1=linien, 2=teams, 3=ignorieren
        self.video_pts  = []
        self.field_pts  = []
        self.homo_mode  = "video"
        self.n_pairs    = 0
        self.H          = None
        self.H_inv      = None
        self.show_lines = False  # Hough Linien anzeigen

        self.team_a_color = None
        self.team_b_color = None
        self.window = "Kalibrierung | ENTER=weiter | R=löschen | L=Linien"

    def compute_homography(self):
        src = np.array(self.video_pts, dtype=np.float32)
        dst = np.array(self.field_pts,  dtype=np.float32)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is not None:
            self.H     = H
            self.H_inv = np.linalg.inv(H)
            print(f"  Homographie: {mask.sum()}/{len(src)} Inlier")
            return True
        return False

    def assign_teams_kmeans(self):
        """K-Means Team-Zuweisung"""
        self.dets = auto_assign_teams(self.dets)
        # Team-Farben aus Clustern ableiten
        a_colors = [d["color"] for d in self.dets if d["team"]=="A" and not d["ignored"]]
        b_colors = [d["color"] for d in self.dets if d["team"]=="B" and not d["ignored"]]
        if a_colors: self.team_a_color = np.mean(a_colors, axis=0).astype(int).tolist()
        if b_colors: self.team_b_color = np.mean(b_colors, axis=0).astype(int).tolist()

    def find_nearest(self, ox, oy, max_d=80):
        best=None; best_d=999999
        for det in self.dets:
            if det["class"]=="ball": continue
            cx,cy=det["center"]
            d=((cx-ox)**2+(cy-oy)**2)**0.5
            if d<best_d: best_d=d; best=det
        return best if best_d<max_d/self.vscale else None

    def mouse_callback(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN: return
        vy = y - self.vy_off

        if self.step == 1:
            if x < self.half_w and self.homo_mode=="video" and 0<=vy<=self.vh:
                self.video_pts.append((x/self.vscale, vy/self.vscale))
                self.homo_mode = "field"
            elif x >= self.half_w and self.homo_mode=="field":
                self.field_pts.append((x-self.half_w, y))
                self.n_pairs  += 1
                self.homo_mode = "video"

        elif self.step == 2:
            if x < self.half_w:
                ox=x/self.vscale; oy=vy/self.vscale
                det = self.find_nearest(ox, oy)
                if det:
                    # Toggle zwischen A und B
                    det["team"] = "B" if det["team"]=="A" else "A"
                    print(f"  #{det['id']} → Team {det['team']}")

        elif self.step == 3:
            if x < self.half_w:
                ox=x/self.vscale; oy=vy/self.vscale
                best=None; best_d=999999
                for det in self.dets:
                    cx,cy=det["center"]
                    d=((cx-ox)**2+(cy-oy)**2)**0.5
                    if d<best_d: best_d=d; best=det
                if best and best_d<80/self.vscale:
                    best["ignored"]=not best["ignored"]
                    print(f"  #{best['id']} → {'ignoriert' if best['ignored'] else 'aktiv'}")

    def render(self):
        canvas = np.zeros((self.display_h,self.display_w,3),dtype=np.uint8)
        canvas[:] = BG

        # Video (mit oder ohne Hough-Linien)
        vf = cv2.resize(self.frame,(self.vw,self.vh))
        if self.show_lines and self.line_img is not None:
            line_small = cv2.resize(self.line_img,(self.vw,self.vh))
            vf = cv2.addWeighted(vf,0.7,line_small,0.3,0)

        # Vorgeschlagene Punkte anzeigen
        if self.step == 1 and self.suggested:
            for sp in self.suggested:
                sx=int(sp[0]*self.vscale); sy=int(sp[1]*self.vscale)+self.vy_off
                cv2.circle(canvas,(sx,sy),6,(0,255,200),1)

        canvas[self.vy_off:self.vy_off+self.vh,0:self.vw] = vf

        # 2D Feld
        if self.step == 1:
            canvas[:,self.half_w:] = self.field_img.copy()

        cv2.line(canvas,(self.half_w,0),(self.half_w,self.display_h),(60,60,60),2)

        if self.step == 1:
            col = CYAN if self.homo_mode=="video" else MAGENTA
            txt = "Klicke VIDEO" if self.homo_mode=="video" else "Klicke 2D FELD"
            cv2.putText(canvas,f"SCHRITT 1/3 — {txt} | Paare: {self.n_pairs} (mind. 8)",
                (10,30),cv2.FONT_HERSHEY_SIMPLEX,0.65,col,2)
            cv2.putText(canvas,"Türkis = Vorgeschlagene Punkte (Hough) | L = Linien ein/aus",
                (10,55),cv2.FONT_HERSHEY_SIMPLEX,0.45,TEXT_COLOR,1)
            cv2.putText(canvas,"ENTER=weiter | R=letztes löschen",
                (10,self.display_h-15),cv2.FONT_HERSHEY_SIMPLEX,0.45,TEXT_COLOR,1)

            for i,(vp,fp) in enumerate(zip(self.video_pts,self.field_pts)):
                c=POINT_COLORS[i%len(POINT_COLORS)]
                vx=int(vp[0]*self.vscale); vy_=int(vp[1]*self.vscale)+self.vy_off
                cv2.circle(canvas,(vx,vy_),8,c,-1)
                cv2.circle(canvas,(vx,vy_),8,(0,0,0),2)
                cv2.putText(canvas,str(i+1),(vx+8,vy_-5),cv2.FONT_HERSHEY_SIMPLEX,0.6,c,2)
                fpx_=int(fp[0])+self.half_w; fpy_=int(fp[1])
                cv2.circle(canvas,(fpx_,fpy_),8,c,-1)
                cv2.putText(canvas,str(i+1),(fpx_+8,fpy_-5),cv2.FONT_HERSHEY_SIMPLEX,0.6,c,2)

            if len(self.video_pts)>len(self.field_pts):
                vp=self.video_pts[-1]; c=POINT_COLORS[len(self.field_pts)%len(POINT_COLORS)]
                vx=int(vp[0]*self.vscale); vy_=int(vp[1]*self.vscale)+self.vy_off
                cv2.circle(canvas,(vx,vy_),8,c,-1)
                cv2.circle(canvas,(vx,vy_),10,(255,255,255),2)

        elif self.step == 2:
            n_a=sum(1 for d in self.dets if d["team"]=="A")
            n_b=sum(1 for d in self.dets if d["team"]=="B")
            cv2.putText(canvas,
                f"SCHRITT 2/3 — Teams korrigieren | A:{n_a} B:{n_b} | Klick=Toggle A↔B",
                (10,30),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,220,0),2)
            cv2.putText(canvas,"K-Means Auto-Zuordnung aktiv | ENTER=weiter wenn korrekt",
                (10,self.display_h-15),cv2.FONT_HERSHEY_SIMPLEX,0.45,TEXT_COLOR,1)
            self._draw_dets(canvas)

        elif self.step == 3:
            cv2.putText(canvas,"SCHRITT 3/3 — Trainer/Fotografen ignorieren",
                (10,30),cv2.FONT_HERSHEY_SIMPLEX,0.65,(200,200,0),2)
            cv2.putText(canvas,"Klick = ignorieren/aktivieren | ENTER=fertig",
                (10,self.display_h-15),cv2.FONT_HERSHEY_SIMPLEX,0.45,TEXT_COLOR,1)
            self._draw_dets(canvas)

        return canvas

    def _draw_dets(self, canvas):
        for det in self.dets:
            if det["class"]=="ball": continue
            x1,y1,x2,y2 = det["bbox"]
            sx1=int(x1*self.vscale); sy1=int(y1*self.vscale)+self.vy_off
            sx2=int(x2*self.vscale); sy2=int(y2*self.vscale)+self.vy_off
            scx=int(det["center"][0]*self.vscale); scy=int(det["center"][1]*self.vscale)+self.vy_off
            if det["ignored"]:   col=IGNORE_COL; label="X"
            elif det["team"]=="A": col=TEAM_A_COL; label="A"
            elif det["team"]=="B": col=TEAM_B_COL; label="B"
            else:                  col=(150,150,150); label="?"
            # Nur Rahmen — keine Füllung, kein Punkt
            cv2.rectangle(canvas,(sx1,sy1),(sx2,sy2),col,2)
            # Kleines Label oben links
            cv2.putText(canvas,label,(sx1+2,sy1-4),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,col,2)

    def run(self):
        cv2.namedWindow(self.window,cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window,self.display_w,self.display_h)
        cv2.setMouseCallback(self.window,self.mouse_callback)

        while True:
            cv2.imshow(self.window,self.render())
            key = cv2.waitKey(1)&0xFF

            if key==13:  # ENTER
                if self.step==1 and self.n_pairs>=4:
                    if self.compute_homography():
                        self.assign_teams_kmeans()
                        self.step=2
                elif self.step==2:
                    self.step=3
                elif self.step==3:
                    break
            elif key==ord('r') and self.step==1 and self.n_pairs>0:
                self.video_pts.pop(); self.field_pts.pop()
                self.n_pairs-=1; self.homo_mode="video"
            elif key==ord('l') or key==ord('L'):
                self.show_lines=not self.show_lines
            elif key==27:
                cv2.destroyAllWindows(); return None

        cv2.destroyAllWindows()

        ignored_pos = [[d["center"][0],d["center"][1]] for d in self.dets if d["ignored"]]
        team_assignments = {str(d["id"]):d["team"] for d in self.dets if d["team"]}

        return {
            "H":                 self.H.tolist(),
            "H_inv":             self.H_inv.tolist(),
            "field_w":           self.field_w,
            "field_h":           self.field_h,
            "n_pairs":           self.n_pairs,
            "video_pts":         [[float(x),float(y)] for x,y in self.video_pts],
            "field_pts":         [[float(x),float(y)] for x,y in self.field_pts],
            "team_a_color":      self.team_a_color,
            "team_b_color":      self.team_b_color,
            "ignored_positions": ignored_pos,
            "team_assignments":  team_assignments,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip",   type=str, required=True)
    parser.add_argument("--model",  type=str, default=None)
    parser.add_argument("--conf",   type=float, default=0.3)
    parser.add_argument("--frames", type=int,   default=60)
    args = parser.parse_args()

    clip_path = Path(args.clip)
    if not clip_path.exists(): clip_path=CLIPS_DIR/args.clip
    if not clip_path.exists(): print("❌ Clip nicht gefunden"); return

    if args.model: model_path=Path(args.model)
    else:
        models=list(MODELS_DIR.glob("*.pt"))
        if not models: print("❌ Kein Modell!"); return
        model_path=sorted(models)[-1]
        print(f"  Modell: {model_path.name}")

    print(f"\n{'='*55}\n  KALIBRIERUNG — {clip_path.name}\n{'='*55}")

    # Besten Frame finden
    frame, dets = find_best_frame(clip_path, model_path, args.conf, args.frames)
    print(f"  {len([d for d in dets if d['class']!='ball'])} Spieler erkannt")

    # Feldlinien automatisch erkennen
    print(f"  Erkenne Feldlinien (Hough Transform)...")
    suggested_pts, line_img = detect_field_lines(frame)
    print(f"  {len(suggested_pts)} Vorschläge gefunden")

    # Kalibrierungs-Interface
    cal = Calibrator(frame, dets, suggested_pts, line_img)
    result = cal.run()
    if result is None: print("❌ Abgebrochen"); return

    out = OUTPUT_DIR / f"calibration_{clip_path.stem}.json"
    with open(out,"w") as f: json.dump(result,f,indent=2)

    print(f"\n  ✅ Gespeichert: {out.name}")
    print(f"  Punkte:    {result['n_pairs']}")
    print(f"  Team A:    {result['team_a_color']}")
    print(f"  Team B:    {result['team_b_color']}")
    print(f"  Ignoriert: {len(result['ignored_positions'])}\n")


if __name__=="__main__":
    main()