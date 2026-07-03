# visualizer/pitch.py
"""
2D Spielfeld Visualisierung — optimiertes Design
→ 1920x1080 Side-by-Side
→ Glow-Effekte + moderne Optik
→ EMA Smoothing für stabile Punkte
→ Traces pro Team

Aufruf:
    python3 visualizer/pitch.py \
        --tracking output/track_test.json \
        --clip clips/test.mp4 \
        --ema 0.25
"""

import sys, os, argparse, json
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import CLIPS_DIR, OUTPUT_DIR

FIELD_WIDTH  = 68.0
FIELD_LENGTH = 105.0

BG           = (10,  10,  10)
FIELD_BG     = (18,  62,  18)
FIELD_STRIPE = (20,  68,  20)
FIELD_LINE   = (210, 210, 210)
TEAM_A       = (220,  70,  70)
TEAM_A_GLOW  = (255, 130, 130)
TEAM_B       = ( 70, 130, 220)
TEAM_B_GLOW  = (130, 175, 255)
GK_A         = (220, 170,  50)
GK_B         = ( 50, 200, 170)
REF_COL      = (200, 200,  50)
BALL_COL     = (255, 255, 255)
BALL_INTERP  = (140, 140, 140)
TEXT_DIM     = (120, 120, 120)


class EMASmooth:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.vals  = {}

    def smooth(self, key, x, y):
        if key not in self.vals:
            self.vals[key]=(x,y); return x,y
        ox,oy=self.vals[key]
        nx=self.alpha*x+(1-self.alpha)*ox
        ny=self.alpha*y+(1-self.alpha)*oy
        self.vals[key]=(nx,ny); return nx,ny


def make_field(w=960, h=660):
    img=np.zeros((h,w,3),dtype=np.uint8); img[:]=FIELD_BG
    px_=int(w*0.055); py_=int(h*0.055); fw=w-2*px_; fh=h-2*py_

    def px(x): return int(px_+(x/FIELD_LENGTH)*fw)
    def py(y): return int(py_+(y/FIELD_WIDTH)*fh)

    # Rasenstreifen
    stripe_h = fh//10
    for i in range(10):
        if i%2==0:
            y1=py_+i*stripe_h; y2=y1+stripe_h
            cv2.rectangle(img,(px_,y1),(px_+fw,y2),FIELD_STRIPE,-1)

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
    # Tore
    gd=2.44; gw=7.32
    cv2.rectangle(img,(px(-gd),py((FIELD_WIDTH-gw)/2)),(px(0),py((FIELD_WIDTH+gw)/2)),FIELD_LINE,lw)
    cv2.rectangle(img,(px(FIELD_LENGTH),py((FIELD_WIDTH-gw)/2)),(px(FIELD_LENGTH+gd),py((FIELD_WIDTH+gw)/2)),FIELD_LINE,lw)
    # Mittelkreis Punkt
    cv2.circle(img,(px(FIELD_LENGTH/2),py(FIELD_WIDTH/2)),4,(255,255,255),-1)

    return img, px, py


def draw_glow(field, x, y, color, glow_color, r=10, intensity=0.3):
    """Spieler mit Glow-Effekt"""
    for i in range(4,0,-1):
        ov=field.copy()
        cv2.circle(ov,(x,y),r+i*3,glow_color,-1)
        cv2.addWeighted(ov,intensity*0.08,field,1-intensity*0.08,0,field)
    cv2.circle(field,(x,y),r,color,-1)
    cv2.circle(field,(x,y),r,(255,255,255),1)


def draw_ball_glow(field, x, y, interp=False):
    col = BALL_INTERP if interp else BALL_COL
    for i in range(5,0,-1):
        ov=field.copy()
        cv2.circle(ov,(x,y),6+i*2,(255,255,200),-1)
        cv2.addWeighted(ov,0.06,field,0.94,0,field)
    cv2.circle(field,(x,y),7,col,-1)
    cv2.circle(field,(x,y),7,(80,80,80),1)


def create_video(tracking_data, output_path, fps=25.0,
                 field_w=960, field_h=660, clip_path=None,
                 ema_alpha=0.25, trace_len=20):

    side_by_side = clip_path is not None and Path(clip_path).exists()
    if side_by_side:
        cap      = cv2.VideoCapture(str(clip_path))
        vid_fps  = cap.get(cv2.CAP_PROP_FPS)
        vid_w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vscale   = field_h / vid_h
        vid_w_sc = int(vid_w * vscale)
        fps      = vid_fps
        out_w    = vid_w_sc + field_w
        out_h    = field_h
    else:
        out_w=field_w; out_h=field_h

    fourcc=cv2.VideoWriter_fourcc(*"avc1")
    writer=cv2.VideoWriter(str(output_path),fourcc,fps,(out_w,out_h))
    if not writer.isOpened():
        fourcc=cv2.VideoWriter_fourcc(*"mp4v")
        writer=cv2.VideoWriter(str(output_path),fourcc,fps,(out_w,out_h))

    base_field, px_f, py_f = make_field(field_w, field_h)
    ema        = EMASmooth(alpha=ema_alpha)
    traces     = defaultdict(list)
    ball_trace = []

    print(f"\n  2D Video: {out_w}x{out_h} @ {fps:.1f}fps")

    for fd in tracking_data:
        field = base_field.copy()

        # ── Traces ────────────────────────────────────────────────
        for key, trace in traces.items():
            if len(trace)<2: continue
            recent = trace[-trace_len:]
            team   = recent[-1][2]
            col    = TEAM_A if team=="A" else TEAM_B
            for i in range(1,len(recent)):
                alpha=(i/len(recent))*0.4
                c=tuple(int(x*alpha) for x in col)
                cv2.line(field,(recent[i-1][0],recent[i-1][1]),
                         (recent[i][0],recent[i][1]),c,1)

        if len(ball_trace)>=2:
            recent=ball_trace[-trace_len:]
            for i in range(1,len(recent)):
                alpha=(i/len(recent))*0.5
                cv2.line(field,recent[i-1],recent[i],
                         tuple(int(x*alpha) for x in BALL_COL),2)

        # ── Spieler ───────────────────────────────────────────────
        for p in fd["players"]:
            if "field_pos" not in p: continue
            fm_x,fm_y = p["field_pos"]
            team      = p.get("team","?")
            cls       = p.get("class","player")

            # Smoothing
            key  = f"{team}_{round(fm_x,0)}_{round(fm_y,0)}"
            sx,sy= ema.smooth(key, px_f(fm_x), py_f(fm_y))
            x=int(sx); y=int(sy)

            is_gk = cls=="goalkeeper"
            if team=="A":   col=GK_A if is_gk else TEAM_A; gcol=TEAM_A_GLOW
            elif team=="B": col=GK_B if is_gk else TEAM_B; gcol=TEAM_B_GLOW
            elif team=="ref": col=REF_COL; gcol=REF_COL
            else:           col=(120,120,120); gcol=(120,120,120)

            r = 13 if is_gk else 10
            draw_glow(field,x,y,col,gcol,r)

            # Trace
            traces[key].append((x,y,team))
            if len(traces[key])>trace_len*3:
                traces[key]=traces[key][-trace_len*3:]

        # ── Ball ──────────────────────────────────────────────────
        if fd.get("ball") and "field_pos" in fd["ball"]:
            bfx,bfy=fd["ball"]["field_pos"]
            interp =fd["ball"].get("interpolated",False)
            sbx,sby=ema.smooth("ball",px_f(bfx),py_f(bfy))
            bx=int(sbx); by=int(sby)
            ball_trace.append((bx,by))
            if len(ball_trace)>trace_len*3: ball_trace=ball_trace[-trace_len*3:]
            draw_ball_glow(field,bx,by,interp)

        # ── Info ──────────────────────────────────────────────────
        n_a=sum(1 for p in fd["players"] if p.get("team")=="A")
        n_b=sum(1 for p in fd["players"] if p.get("team")=="B")
        cv2.circle(field,(16,field_h-22),7,TEAM_A,-1)
        cv2.putText(field,f"A  {n_a}",(28,field_h-17),cv2.FONT_HERSHEY_SIMPLEX,0.45,TEAM_A,1)
        cv2.circle(field,(100,field_h-22),7,TEAM_B,-1)
        cv2.putText(field,f"B  {n_b}",(112,field_h-17),cv2.FONT_HERSHEY_SIMPLEX,0.45,TEAM_B,1)
        cv2.putText(field,f"#{fd['frame']}",(field_w-55,field_h-10),
            cv2.FONT_HERSHEY_SIMPLEX,0.35,(60,60,60),1)

        # ── Canvas ────────────────────────────────────────────────
        if side_by_side:
            ret,vf=cap.read()
            vf=cv2.resize(vf,(vid_w_sc,field_h)) if ret else np.zeros((field_h,vid_w_sc,3),dtype=np.uint8)
            canvas=np.zeros((out_h,out_w,3),dtype=np.uint8)
            canvas[:,:vid_w_sc]=vf; canvas[:,vid_w_sc:]=field
            cv2.line(canvas,(vid_w_sc,0),(vid_w_sc,field_h),(40,40,40),2)
        else:
            canvas=field

        writer.write(canvas)
        fi=fd["frame"]
        if fi%50==0: print(f"    Frame {fi}/{len(tracking_data)}")

    writer.release()
    if side_by_side: cap.release()
    print(f"  ✅ {output_path.name}")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--tracking",type=str,required=True)
    parser.add_argument("--clip",    type=str,default=None)
    parser.add_argument("--ema",     type=float,default=0.25)
    parser.add_argument("--trace",   type=int,  default=20)
    args=parser.parse_args()

    track_path=Path(args.tracking)
    if not track_path.exists(): track_path=OUTPUT_DIR/args.tracking
    if not track_path.exists(): print("❌ Tracking nicht gefunden"); return

    print(f"\n{'='*50}\n  2D PITCH VISUALIZER\n{'='*50}")
    with open(track_path) as f: data=json.load(f)
    print(f"  Frames: {len(data)}")

    clip_path=None
    if args.clip:
        clip_path=Path(args.clip)
        if not clip_path.exists(): clip_path=CLIPS_DIR/args.clip
        if not clip_path.exists(): clip_path=None

    suffix="_sidebyside" if clip_path else "_2d"
    out=OUTPUT_DIR/f"pitch_{track_path.stem}{suffix}.mp4"

    create_video(data,out,clip_path=clip_path,
                 ema_alpha=args.ema,trace_len=args.trace)
    print(f"\n{'='*50}\n  FERTIG: {out.name}\n{'='*50}\n")


if __name__=="__main__":
    main()