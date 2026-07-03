# extract_frames.py
"""
Frames aus Clips extrahieren für Annotation
→ nimmt jeden N-ten Frame
→ speichert als JPEG für Roboflow Upload

Aufruf:
    python3 extract_frames.py \
        --clip clips/test.mp4 \
        --every 5
"""

import cv2
import argparse
from pathlib import Path
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def extract(clip_path, output_dir, every_n=5, max_frames=None):
    cap   = cv2.VideoCapture(str(clip_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Clip:    {clip_path.name}")
    print(f"  Frames:  {total} @ {fps:.1f}fps")
    print(f"  Jeden:   {every_n}. Frame")

    i = 0; saved = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if max_frames and i >= max_frames: break

        if i % every_n == 0:
            fname = output_dir / f"{clip_path.stem}_f{i:04d}.jpg"
            cv2.imwrite(str(fname), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1

        i += 1

    cap.release()
    print(f"  ✅ {saved} Frames → {output_dir}")
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip",   type=str, required=True)
    parser.add_argument("--every",  type=int, default=5,
                        help="Jeden N-ten Frame (default: 5)")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--frames", type=int, default=None)
    args = parser.parse_args()

    from config import CLIPS_DIR, OUTPUT_DIR

    clip_path = Path(args.clip)
    if not clip_path.exists():
        clip_path = CLIPS_DIR / args.clip
    if not clip_path.exists():
        print(f"❌ Clip nicht gefunden: {args.clip}")
        return

    out_dir = Path(args.output) if args.output else \
              OUTPUT_DIR.parent / "annotation_frames" / clip_path.stem

    extract(clip_path, out_dir, args.every, args.frames)

    print(f"\n  Nächster Schritt:")
    print(f"  → roboflow.com öffnen")
    print(f"  → Neues Projekt: 'football-tactical-personal'")
    print(f"  → Klassen: ball, goalkeeper, player, referee")
    print(f"  → Frames hochladen aus: {out_dir}")
    print(f"  → Annotieren starten\n")


if __name__ == "__main__":
    main()