# Football Tracking System 🎯

Computer Vision Pipeline zur automatischen Erkennung und Verfolgung von Fußballspielern in Videoaufnahmen — mit moderner 2D Spielfeld-Visualisierung.

```
Input:  Fußball-Videoaufnahme (Broadcast / Tactical Camera)
Output: 2D Spielfeld-Visualisierung mit teamfarbigen Spielerpunkten
```

![Pipeline](docs/pipeline.png)

## Features

- **YOLOv9e Detection** — State-of-the-art Objekterkennung, trainiert auf fußballspezifischen Daten (Ball, Torhüter, Spieler, Schiedsrichter)
- **ByteTrack** — Stabile Spieler-IDs über mehrere Frames mit Team-Zuordnung nach 15 Frames einfrieren
- **Homographie-Kalibrierung** — Manuelle Feldpunkt-Markierung mit Hough-Transform Linienerkennungs-Unterstützung
- **Team-Zuordnung** — Referenzfarben-basierte Zuweisung mit Vote-Stabilisierung über mehrere Frames
- **Ball-Interpolation** — Glatte Ball-Trajektorie auch bei kurzzeitiger Nicht-Erkennung (bis zu 20 Frames)
- **EMA Smoothing** — Exponentieller gleitender Durchschnitt für stabile Spielerpositionen auf dem 2D Spielfeld
- **Kalman-Filter** — Für robuste Ball-Positionsschätzung
- **Modernes 2D Rendering** — Glow-Effekte, Spieler-Traces und Side-by-Side Ausgabe

## Workflow

```bash
# Schritt 1 — Kalibrieren (einmalig pro Clip)
python3 tracker/calibrate.py \
    --clip clips/spiel.mp4 \
    --model models/football_v3.pt

# Schritt 2 — Tracking
python3 tracker/track.py \
    --clip clips/spiel.mp4 \
    --calibration output/calibration_spiel.json \
    --model models/football_v3.pt

# Schritt 3 — 2D Visualisierung
python3 visualizer/pitch.py \
    --tracking output/track_spiel.json \
    --clip clips/spiel.mp4 \
    --ema 0.25
```

## Kalibrierungs-Interface

Das Kalibrierungs-Interface öffnet sich interaktiv in 3 Schritten:

| Schritt | Aktion |
|---------|--------|
| 1 | Mind. 8 Feldpunkte markieren (Video ↔ 2D Feld) — Hough-Transform schlägt Kandidaten vor |
| 2 | Team-Zuordnung korrigieren (Klick = Toggle A↔B) |
| 3 | Trainer/Fotografen markieren zum Ignorieren |

`L` = Hough-Linien ein/ausblenden als Orientierungshilfe.

## Modell

Das Erkennungsmodell (`football_v3.pt`) ist ein **YOLOv9e**, trainiert auf dem [Football Players Detection Dataset](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc) (Roboflow):

| Klasse | mAP50 |
|--------|-------|
| Gesamt | 0.888 |
| Ball | 0.636 |
| Torhüter | 0.956 |
| Spieler | 0.992 |
| Schiedsrichter | 0.967 |

Training: 200 Epochen, YOLOv9e, imgsz=1280, A100 GPU (Google Colab)

## Technologien

```
Detection:      YOLOv9e (Ultralytics)
Tracking:       ByteTrack (Supervision)
Kalibrierung:   OpenCV Homographie + Hough Transform
Clustering:     scikit-learn K-Means
Smoothing:      Kalman-Filter (Ball) + EMA (Spieler)
Visualisierung: OpenCV
```

## Installation

```bash
pip install ultralytics supervision opencv-python scikit-learn numpy
```

## Projektstruktur

```
football-tracking/
├── tracker/
│   ├── calibrate.py     # Interaktives Kalibrierungs-Interface
│   └── track.py         # Detection + Tracking Pipeline
├── visualizer/
│   └── pitch.py         # 2D Spielfeld Rendering
├── models/              # Modellgewichte (nicht im Repo)
├── clips/               # Input-Videos (nicht im Repo)
└── output/              # Kalibrierung + Tracking + Videos
```

## Limitierungen

Dies ist ein Forschungs- und Portfolio-Projekt mit bekannten Einschränkungen:

- **Statische Kamera erforderlich** — Homographie wird einmalig berechnet; Kameraschwenks reduzieren die Genauigkeit
- **Teamfarben** — Funktioniert am besten bei klar unterscheidbaren Trikotfarben
- **Occlusion** — Sich überlappende Spieler können kurz verloren gehen
- **Ball-Erkennung** — Ball in der Luft oder bei großer Distanz schwer erkennbar (63.6% Erkennungsrate)
- **Kein Re-ID** — Spieler die verschwinden und wieder auftauchen erhalten ggf. eine neue ID

Professionelle Tracking-Systeme (Tracab, Second Spectrum) lösen diese Probleme mit fest installierten Multi-Kamera-Setups und jahrelanger Entwicklung.

## Verwandte Projekte

- [Football Coaching Analytics](https://github.com/AF0203/football-coaching-analytics) — Taktisches Profiling und Trainer-Matching-System auf Basis von Event-Daten

---

**Adrian Friedrich**
[LinkedIn](https://www.linkedin.com/in/adrian-friedrich-4a318141b/) · [GitHub](https://github.com/AF0203) · Adrianfriedrich12@gmail.com
