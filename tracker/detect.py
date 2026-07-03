# tracker/detect.py
"""
Spieler-Detection mit YOLOv8
→ lädt vortrainiertes Modell
→ erkennt Spieler, Torwart, Ball, Schiedsrichter
→ gibt Detection-Ergebnisse pro Frame zurück

Test:
    python3 tracker/detect.py --clip clips/test.mp4
"""

import sys, os, argparse
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
))
from config import MODELS_DIR, CLIPS_DIR, OUTPUT_DIR

# ── Klassen aus dem Roboflow Football Modell ──────────────────────
# Passe diese an je nach Modell das du verwendest
CLASS_NAMES = {
    0: "ball",
    1: "goalkeeper",
    2: "player",
    3: "referee",
}

CLASS_COLORS = {
    "ball":       (255, 50,  50),  # Rot
    "goalkeeper": (0, 255, 100),   # Grün
    "player":     (0, 200, 255),   # Cyan
    "referee":    (200, 200, 200), # Grau
}


# ══════════════════════════════════════════════════════════════════
#  DETECTOR KLASSE
# ══════════════════════════════════════════════════════════════════

class PlayerDetector:
    def __init__(self, model_path, conf=0.3):
        """
        model_path: Pfad zur .pt Datei
        conf:       Mindest-Konfidenz (0.0 - 1.0)
        """
        print(f"  Lade Modell: {Path(model_path).name}")
        self.model = YOLO(model_path)
        self.conf  = conf
        print(f"  Konfidenz-Schwelle: {conf}")

    def detect_frame(self, frame):
        """
        Erkennt Objekte in einem Frame.
        Gibt Liste von Dicts zurück:
        [{"class": "player", "bbox": [x1,y1,x2,y2], "conf": 0.85}, ...]
        """
        results = self.model(frame, conf=self.conf, verbose=False)[0]
        detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")

            detections.append({
                "class":  cls_name,
                "bbox":   [x1, y1, x2, y2],
                "conf":   round(conf, 3),
                "center": ((x1 + x2) // 2, (y1 + y2) // 2),
                "foot":   ((x1 + x2) // 2, y2),  # Fußpunkt
            })

        return detections

    def draw_detections(self, frame, detections):
        """Zeichnet Bounding Boxes auf Frame"""
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            color = CLASS_COLORS.get(det["class"], (255, 255, 255))
            label = f"{det['class']} {det['conf']:.2f}"

            # Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label Background
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(frame,
                (x1, y1 - th - 6), (x1 + tw + 4, y1),
                color, -1
            )

            # Label Text
            cv2.putText(frame, label,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 1
            )

            # Fußpunkt
            cv2.circle(frame, det["foot"], 4, color, -1)

        return frame


# ══════════════════════════════════════════════════════════════════
#  TEST — Video verarbeiten und Output speichern
# ══════════════════════════════════════════════════════════════════

def test_detection(clip_path, model_path, conf=0.3, max_frames=None):
    detector = PlayerDetector(model_path, conf)

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        print(f"❌ Video nicht gefunden: {clip_path}")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n  Video: {Path(clip_path).name}")
    print(f"  Auflösung: {width}x{height} @ {fps:.1f} fps")
    print(f"  Frames: {total}")

    # Output Video
    out_path = OUTPUT_DIR / f"detect_{Path(clip_path).stem}.mp4"
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    frame_idx   = 0
    total_dets  = {"player": 0, "goalkeeper": 0, "ball": 0, "referee": 0}

    print(f"\n  Verarbeite Frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames and frame_idx >= max_frames:
            break

        # Frame skalieren für schnellere Detection
        scale   = min(1280 / frame.shape[1], 1280 / frame.shape[0])
        if scale < 1.0:
            small = cv2.resize(frame,
                (int(frame.shape[1]*scale), int(frame.shape[0]*scale)))
        else:
            small = frame

        detections_small = detector.detect_frame(small)

        # Koordinaten zurück auf Original-Größe skalieren
        detections = []
        for det in detections_small:
            x1, y1, x2, y2 = det["bbox"]
            if scale < 1.0:
                x1 = int(x1 / scale)
                y1 = int(y1 / scale)
                x2 = int(x2 / scale)
                y2 = int(y2 / scale)
            det["bbox"]   = [x1, y1, x2, y2]
            det["center"] = ((x1+x2)//2, (y1+y2)//2)
            det["foot"]   = ((x1+x2)//2, y2)
            detections.append(det)

        # Stats
        for det in detections:
            cls = det["class"]
            if cls in total_dets:
                total_dets[cls] += 1

        # Frame annotieren
        annotated = detector.draw_detections(frame.copy(), detections)

        # Frame-Info
        n_players = sum(1 for d in detections
                        if d["class"] in ["player","goalkeeper"])
        n_ball    = sum(1 for d in detections if d["class"] == "ball")
        cv2.putText(annotated,
            f"Frame {frame_idx} | Spieler: {n_players} | Ball: {n_ball}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (255, 255, 255), 2
        )

        writer.write(annotated)
        frame_idx += 1

        if frame_idx % 30 == 0:
            print(f"    Frame {frame_idx}/{min(total, max_frames or total)}")

    cap.release()
    writer.release()

    # Zusammenfassung
    print(f"\n{'='*50}")
    print(f"  DETECTION ERGEBNISSE")
    print(f"{'='*50}")
    print(f"  Frames verarbeitet: {frame_idx}")
    print(f"  Spieler erkannt:    {total_dets['player']} "
          f"(avg {total_dets['player']/max(frame_idx,1):.1f}/Frame)")
    print(f"  Torhüter erkannt:   {total_dets['goalkeeper']} "
          f"(avg {total_dets['goalkeeper']/max(frame_idx,1):.1f}/Frame)")
    print(f"  Ball erkannt:       {total_dets['ball']} "
          f"(avg {total_dets['ball']/max(frame_idx,1):.1f}/Frame)")
    print(f"\n  Output: {out_path}")
    print(f"{'='*50}\n")


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip",   type=str, required=True,
                        help="Pfad zum Video-Clip")
    parser.add_argument("--model",  type=str, default=None,
                        help="Pfad zur .pt Modelldatei")
    parser.add_argument("--conf",   type=float, default=0.3,
                        help="Konfidenz-Schwelle (default: 0.3)")
    parser.add_argument("--frames", type=int, default=None,
                        help="Max. Frames (default: alle)")
    args = parser.parse_args()

    # Modell finden
    if args.model:
        model_path = Path(args.model)
    else:
        # Automatisch erstes .pt File in models/ suchen
        models = list(MODELS_DIR.glob("*.pt"))
        if not models:
            print("❌ Kein Modell gefunden!")
            print(f"   Lege .pt Datei in {MODELS_DIR} ab")
            print(f"   oder nutze --model Pfad")
            print(f"\n   Tipp: YOLOv8n als Default-Modell laden:")
            print(f"   → einfach --model yolov8n.pt angeben")
            print(f"     (wird automatisch heruntergeladen)")
            return
        model_path = models[0]
        print(f"  Modell gefunden: {model_path.name}")

    # Clip finden
    clip_path = Path(args.clip)
    if not clip_path.exists():
        # In clips/ suchen
        clip_path = CLIPS_DIR / args.clip
        if not clip_path.exists():
            print(f"❌ Clip nicht gefunden: {args.clip}")
            return

    print(f"\n{'='*50}")
    print(f"  PLAYER DETECTOR TEST")
    print(f"{'='*50}")

    test_detection(
        clip_path  = clip_path,
        model_path = model_path,
        conf       = args.conf,
        max_frames = args.frames,
    )


if __name__ == "__main__":
    main()