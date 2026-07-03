# tracker/homography.py
"""
Homographie — Kamera → 2D Spielfeld
→ Side-by-Side Interface:
  Links: Video-Frame | Rechts: 2D Spielfeld
→ User klickt Punkt-Paare an
→ OpenCV berechnet Transformation

Aufruf:
    python3 tracker/homography.py --clip clips/test.mp4 --calibrate
    python3 tracker/homography.py --clip clips/test.mp4 --tracking output/track_test.json
"""

import sys, os, argparse, json
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
))
from config import CLIPS_DIR, OUTPUT_DIR

# ── Feldmaße ──────────────────────────────────────────────────────
FIELD_WIDTH  = 68.0
FIELD_LENGTH = 105.0

# Farben
BG          = (13,  13,  13)
FIELD_GREEN = (30,  100, 30)
FIELD_LINE  = (200, 200, 200)
CYAN        = (255, 200, 0)
MAGENTA     = (200, 50,  200)
TEXT_COLOR  = (230, 230, 230)
POINT_COLORS = [
    (255, 80,  80),
    (80,  255, 80),
    (80,  80,  255),
    (255, 255, 80),
    (255, 80,  255),
    (80,  255, 255),
    (255, 160, 80),
    (160, 80,  255),
]


# ══════════════════════════════════════════════════════════════════
#  2D SPIELFELD ZEICHNEN
# ══════════════════════════════════════════════════════════════════

def draw_2d_field(width=600, height=400):
    """Zeichnet ein 2D Spielfeld mit Linien"""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = FIELD_GREEN

    # Feld-Padding
    pad_x = int(width  * 0.05)
    pad_y = int(height * 0.05)
    fw    = width  - 2 * pad_x
    fh    = height - 2 * pad_y

    def fx(x_meter):
        return int(pad_x + (x_meter / FIELD_LENGTH) * fw)

    def fy(y_meter):
        return int(pad_y + (y_meter / FIELD_WIDTH) * fh)

    lw = 2  # Linienbreite

    # Außenlinien
    cv2.rectangle(img,
        (fx(0), fy(0)), (fx(FIELD_LENGTH), fy(FIELD_WIDTH)),
        FIELD_LINE, lw)

    # Mittellinie
    cv2.line(img,
        (fx(FIELD_LENGTH/2), fy(0)),
        (fx(FIELD_LENGTH/2), fy(FIELD_WIDTH)),
        FIELD_LINE, lw)

    # Mittelkreis
    r = int((9.15 / FIELD_WIDTH) * fh)
    cv2.circle(img,
        (fx(FIELD_LENGTH/2), fy(FIELD_WIDTH/2)),
        r, FIELD_LINE, lw)
    cv2.circle(img,
        (fx(FIELD_LENGTH/2), fy(FIELD_WIDTH/2)),
        3, FIELD_LINE, -1)

    # Linker Strafraum (16.5m)
    cv2.rectangle(img,
        (fx(0), fy(13.85)),
        (fx(16.5), fy(FIELD_WIDTH - 13.85)),
        FIELD_LINE, lw)

    # Rechter Strafraum
    cv2.rectangle(img,
        (fx(FIELD_LENGTH - 16.5), fy(13.85)),
        (fx(FIELD_LENGTH), fy(FIELD_WIDTH - 13.85)),
        FIELD_LINE, lw)

    # Linkes Toraus (5.5m)
    cv2.rectangle(img,
        (fx(0), fy(24.85)),
        (fx(5.5), fy(FIELD_WIDTH - 24.85)),
        FIELD_LINE, lw)

    # Rechtes Toraus
    cv2.rectangle(img,
        (fx(FIELD_LENGTH - 5.5), fy(24.85)),
        (fx(FIELD_LENGTH), fy(FIELD_WIDTH - 24.85)),
        FIELD_LINE, lw)

    # Elfmeterpunkte
    cv2.circle(img, (fx(11), fy(FIELD_WIDTH/2)), 3, FIELD_LINE, -1)
    cv2.circle(img,
        (fx(FIELD_LENGTH - 11), fy(FIELD_WIDTH/2)),
        3, FIELD_LINE, -1)

    # Eckkreise
    for cx, cy in [(0,0),(FIELD_LENGTH,0),
                   (0,FIELD_WIDTH),(FIELD_LENGTH,FIELD_WIDTH)]:
        cv2.ellipse(img,
            (fx(cx), fy(cy)),
            (int((1.0/FIELD_LENGTH)*fw),
             int((1.0/FIELD_WIDTH)*fh)),
            0, 0, 90, FIELD_LINE, lw)

    return img, fx, fy


# ══════════════════════════════════════════════════════════════════
#  SIDE-BY-SIDE KALIBRIERUNG
# ══════════════════════════════════════════════════════════════════

class SideBySideCalibrator:
    def __init__(self, video_frame, display_w=1400, display_h=500):
        self.display_w   = display_w
        self.display_h   = display_h
        self.half_w      = display_w // 2

        # Video Frame skalieren (links)
        scale = min(
            self.half_w / video_frame.shape[1],
            display_h   / video_frame.shape[0]
        )
        self.video_scale = scale
        vw = int(video_frame.shape[1] * scale)
        vh = int(video_frame.shape[0] * scale)
        self.video_frame = cv2.resize(video_frame, (vw, vh))
        self.video_w     = vw
        self.video_h     = vh

        # 2D Spielfeld (rechts)
        self.field_img, self.fx, self.fy = draw_2d_field(
            self.half_w, display_h
        )

        # Punkt-Paare
        self.video_points = []
        self.field_points = []
        self.mode         = "video"  # "video" oder "field"
        self.n_pairs      = 0
        self.window       = "Kalibrierung — Links: Video | Rechts: 2D Feld"

    def get_combined_frame(self):
        """Baut Side-by-Side Frame zusammen"""
        canvas = np.zeros(
            (self.display_h, self.display_w, 3), dtype=np.uint8
        )
        canvas[:] = BG

        # Video links (zentriert)
        y_off = (self.display_h - self.video_h) // 2
        canvas[
            y_off:y_off+self.video_h,
            0:self.video_w
        ] = self.video_frame

        # 2D Feld rechts
        canvas[:, self.half_w:] = self.field_img.copy()

        # Trennlinie
        cv2.line(canvas,
            (self.half_w, 0), (self.half_w, self.display_h),
            (60, 60, 60), 2)

        # Header
        if self.mode == "video":
            cv2.putText(canvas,
                "1. Klicke auf Punkt im VIDEO",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, CYAN, 2)
            cv2.putText(canvas,
                "danach: gleichen Punkt im 2D Feld",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (150, 150, 150), 1)
        else:
            cv2.putText(canvas,
                "2. Klicke denselben Punkt im 2D FELD",
                (self.half_w + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, MAGENTA, 2)

        # Info unten
        cv2.putText(canvas,
            f"Paare: {self.n_pairs} | "
            f"ENTER = Homographie berechnen | "
            f"R = letztes Paar loeschen | "
            f"ESC = Abbrechen",
            (10, self.display_h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55, TEXT_COLOR, 1)

        # Markierte Punkte zeichnen
        for i, (vp, fp) in enumerate(
            zip(self.video_points, self.field_points)
        ):
            color = POINT_COLORS[i % len(POINT_COLORS)]
            label = str(i + 1)

            # Video-Punkt
            y_off = (self.display_h - self.video_h) // 2
            vx = int(vp[0])
            vy = int(vp[1]) + y_off
            cv2.circle(canvas, (vx, vy), 8, color, -1)
            cv2.circle(canvas, (vx, vy), 8, (0,0,0), 2)
            cv2.putText(canvas, label,
                (vx+10, vy-5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, color, 2)

            # Feld-Punkt
            fpx = int(fp[0]) + self.half_w
            fpy = int(fp[1])
            cv2.circle(canvas, (fpx, fpy), 8, color, -1)
            cv2.circle(canvas, (fpx, fpy), 8, (0,0,0), 2)
            cv2.putText(canvas, label,
                (fpx+10, fpy-5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, color, 2)

        # Aktueller Video-Punkt (noch kein Feld-Paar)
        if len(self.video_points) > len(self.field_points):
            vp    = self.video_points[-1]
            color = POINT_COLORS[
                len(self.field_points) % len(POINT_COLORS)
            ]
            y_off = (self.display_h - self.video_h) // 2
            vx = int(vp[0])
            vy = int(vp[1]) + y_off
            cv2.circle(canvas, (vx, vy), 8, color, -1)
            cv2.circle(canvas, (vx, vy), 10, (255,255,255), 2)
            cv2.putText(canvas,
                f"{len(self.video_points)}",
                (vx+10, vy-5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, color, 2)

        return canvas

    def mouse_callback(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        y_off = (self.display_h - self.video_h) // 2

        if x < self.half_w and self.mode == "video":
            # Klick im Video
            # Koordinaten zurück auf Original-Frame skalieren
            orig_x = x / self.video_scale
            orig_y = (y - y_off) / self.video_scale
            if 0 <= orig_y <= self.video_h / self.video_scale:
                self.video_points.append((orig_x, orig_y))
                self.mode = "field"
                print(f"  Video-Punkt {len(self.video_points)}: "
                      f"({int(orig_x)}, {int(orig_y)})")

        elif x >= self.half_w and self.mode == "field":
            # Klick im 2D Feld
            field_x = x - self.half_w
            field_y = y
            self.field_points.append((field_x, field_y))
            self.n_pairs += 1
            self.mode     = "video"
            print(f"  Feld-Punkt {self.n_pairs}: "
                  f"({field_x}, {field_y})")

    def run(self):
        print(f"\n  Side-by-Side Kalibrierung:")
        print(f"  1. Klicke auf einen Punkt im VIDEO (links)")
        print(f"  2. Klicke auf denselben Punkt im 2D FELD (rechts)")
        print(f"  Wiederhole für mind. 4 Punkte")
        print(f"  ENTER = fertig | R = letztes Paar löschen")

        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window, self.display_w, self.display_h)
        cv2.setMouseCallback(self.window, self.mouse_callback)

        while True:
            frame = self.get_combined_frame()
            cv2.imshow(self.window, frame)
            key   = cv2.waitKey(1) & 0xFF

            if key == 13 and self.n_pairs >= 4:
                # ENTER — fertig
                break
            elif key == ord('r') and self.n_pairs > 0:
                # Letztes Paar löschen
                self.video_points.pop()
                self.field_points.pop()
                self.n_pairs -= 1
                self.mode     = "video"
                print(f"  Letztes Paar gelöscht "
                      f"({self.n_pairs} Paare)")
            elif key == 27:
                cv2.destroyAllWindows()
                return None, None

        cv2.destroyAllWindows()
        print(f"\n  {self.n_pairs} Punkt-Paare markiert")
        return self.video_points, self.field_points


# ══════════════════════════════════════════════════════════════════
#  HOMOGRAPHIE KLASSE
# ══════════════════════════════════════════════════════════════════

class Homography:
    def __init__(self):
        self.H          = None
        self.H_inv      = None
        self.field_w    = None
        self.field_h    = None
        self.field_img  = None
        self.fx         = None
        self.fy         = None

    def calibrate(self, video_points, field_points_px,
                  field_w, field_h):
        """
        video_points:     Pixel-Koordinaten im Original-Video
        field_points_px:  Pixel-Koordinaten im 2D Spielfeld
        field_w/h:        Größe des 2D Spielfelds in Pixel
        """
        self.field_w = field_w
        self.field_h = field_h

        src = np.array(video_points,     dtype=np.float32)
        dst = np.array(field_points_px,  dtype=np.float32)

        self.H, mask = cv2.findHomography(
            src, dst, cv2.RANSAC, 5.0
        )
        if self.H is not None:
            self.H_inv = np.linalg.inv(self.H)
            print(f"  Homographie berechnet!")
            print(f"  Inlier: {mask.sum()}/{len(video_points)}")
            return True
        print(f"  Homographie fehlgeschlagen!")
        return False

    def pixel_to_field_px(self, pixel_x, pixel_y):
        """Video-Pixel → 2D Feld-Pixel"""
        if self.H is None:
            return None
        pt     = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
        result = cv2.perspectiveTransform(pt, self.H)
        x, y   = result[0][0]
        return float(x), float(y)

    def pixel_to_field_meter(self, pixel_x, pixel_y,
                              field_w, field_h):
        """Video-Pixel → Feldkoordinaten in Meter"""
        pos = self.pixel_to_field_px(pixel_x, pixel_y)
        if pos is None:
            return None
        fx_m = (pos[0] / field_w) * FIELD_LENGTH
        fy_m = (pos[1] / field_h) * FIELD_WIDTH
        return round(fx_m, 2), round(fy_m, 2)

    def save(self, path, field_w, field_h,
             video_points, field_points_px):
        data = {
            "H":               self.H.tolist(),
            "field_w":         field_w,
            "field_h":         field_h,
            "video_points":    [[float(x), float(y)]
                                for x,y in video_points],
            "field_points_px": [[float(x), float(y)]
                                for x,y in field_points_px],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Gespeichert: {Path(path).name}")

    def load(self, path):
        with open(path) as f:
            data = json.load(f)
        self.H       = np.array(data["H"], dtype=np.float32)
        self.H_inv   = np.linalg.inv(self.H)
        self.field_w = data["field_w"]
        self.field_h = data["field_h"]
        print(f"  Homographie geladen: {Path(path).name}")

    def transform_tracking(self, tracking_data):
        """Alle Tracking-Positionen → Feldkoordinaten"""
        result = []
        for frame_data in tracking_data:
            frame_result = {
                "frame":   frame_data["frame"],
                "players": [],
                "ball":    None,
            }
            for player in frame_data["players"]:
                foot_x, foot_y = player["foot"]
                pos = self.pixel_to_field_meter(
                    foot_x, foot_y,
                    self.field_w, self.field_h
                )
                if pos:
                    fx, fy = pos
                    if (0 <= fx <= FIELD_LENGTH and
                            0 <= fy <= FIELD_WIDTH):
                        p = player.copy()
                        p["field_pos"] = [fx, fy]
                        frame_result["players"].append(p)

            if frame_data.get("ball"):
                cx, cy = frame_data["ball"]["center"]
                pos = self.pixel_to_field_meter(
                    cx, cy, self.field_w, self.field_h
                )
                if pos:
                    b = frame_data["ball"].copy()
                    b["field_pos"] = list(pos)
                    frame_result["ball"] = b

            result.append(frame_result)
        return result

    def visualize_test(self, video_frame):
        """Test-Bild: Video + Feldpunkte zurückprojiziert"""
        overlay = video_frame.copy()
        field_img, fx, fy = draw_2d_field(600, 400)

        # Einige Feldpunkte zurückprojizieren
        test_pts_m = [
            (0, 0), (FIELD_LENGTH, 0),
            (FIELD_LENGTH, FIELD_WIDTH), (0, FIELD_WIDTH),
            (FIELD_LENGTH/2, 0), (FIELD_LENGTH/2, FIELD_WIDTH),
        ]
        for mx, my in test_pts_m:
            px = (mx / FIELD_LENGTH) * self.field_w
            py = (my / FIELD_WIDTH)  * self.field_h
            pt = np.array([[[px, py]]], dtype=np.float32)
            result = cv2.perspectiveTransform(pt, self.H_inv)
            vx, vy = map(int, result[0][0])
            if (0 <= vx < video_frame.shape[1] and
                    0 <= vy < video_frame.shape[0]):
                cv2.circle(overlay, (vx, vy), 12,
                           (0, 255, 0), -1)
        return overlay


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip",      type=str, required=True)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--tracking",  type=str, default=None)
    args = parser.parse_args()

    clip_path = Path(args.clip)
    if not clip_path.exists():
        clip_path = CLIPS_DIR / args.clip
        if not clip_path.exists():
            print(f"❌ Clip nicht gefunden: {args.clip}")
            return

    homo_path = OUTPUT_DIR / f"homography_{clip_path.stem}.json"

    print(f"\n{'='*50}")
    print(f"  HOMOGRAPHIE — {clip_path.name}")
    print(f"{'='*50}")

    homography = Homography()

    if args.calibrate or not homo_path.exists():
        # Ersten Frame laden
        cap = cv2.VideoCapture(str(clip_path))
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("❌ Frame nicht ladbar")
            return

        # 2D Feld Größe bestimmen
        field_w, field_h = 600, 400

        # Side-by-Side Kalibrierung
        calibrator = SideBySideCalibrator(frame, 1400, field_h)
        video_pts, field_pts = calibrator.run()

        if video_pts is None or len(video_pts) < 4:
            print("❌ Kalibrierung abgebrochen")
            return

        # Homographie berechnen
        success = homography.calibrate(
            video_pts, field_pts, field_w, field_h
        )
        if not success:
            return

        # Speichern
        homography.save(homo_path, field_w, field_h,
                        video_pts, field_pts)

        # Test-Visualisierung
        test = homography.visualize_test(frame)
        test_path = OUTPUT_DIR / \
            f"homography_test_{clip_path.stem}.jpg"
        cv2.imwrite(str(test_path), test)
        print(f"  Test-Bild: {test_path.name}")
        print(f"  → Grüne Punkte = Feldecken zurückprojiziert")

    else:
        homography.load(homo_path)

    # Tracking transformieren
    if args.tracking:
        track_path = Path(args.tracking)
        if not track_path.exists():
            track_path = OUTPUT_DIR / args.tracking
        if track_path.exists():
            print(f"\n  Transformiere Tracking-Daten...")
            with open(track_path) as f:
                tracking_data = json.load(f)

            transformed = homography.transform_tracking(
                tracking_data
            )
            out_path = OUTPUT_DIR / \
                f"field_{track_path.stem}.json"
            with open(out_path, "w") as f:
                json.dump(transformed, f)
            print(f"  Feldpositionen: {out_path.name}")

            # Beispiel
            for p in transformed[0]["players"][:3]:
                print(f"    Spieler #{p['id']}: "
                      f"{p['field_pos']} m")
            if transformed[0]["ball"]:
                print(f"    Ball: "
                      f"{transformed[0]['ball']['field_pos']} m")

    print(f"\n{'='*50}")
    print(f"  FERTIG")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()