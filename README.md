# Gaze-Change Detection Experiment

Peripheral gaze-change detection: psychophysical 2AFC study with eye tracking.
Master's thesis — Amira Shymbolatova, University of Twente, 2026.

## Requirements

- Python 3.8+
- YETI24 eye tracker
- Dependencies: `pip install pygame numpy pandas scipy opencv-python Pillow`

## Setup

1. Clone this repository
2. Download the stimuli from OSF: [https://osf.io/mfd2k/overview?view_only=7124056eb7194772b8df58562982308c]
3. Place the Stimuli folder in the project root directory
4. Run `python generate_stimuli_csv.py` to generate Stimuli/Stimuli.csv
5. Update `Config.csv` with your monitor specs and viewing distance
6. Run `python run.py`

## Repository structure

- `run.py` — main experiment script
- `staircase.py` — adaptive staircase logic
- `geometry.py` — screen geometry utilities
- `haarcascade_eye.xml` — OpenCV cascade used for eye detection
- `libyeti24.py` — YETI24 eye tracker interface
- `Config.csv` — experiment configuration
- `exploratory_analysis.qmd` — R analysis script
- `generate_stimuli_csv.py` — generates stimulus index file

## Stimuli

Stimuli are hosted on OSF Files due to size (2.2GB).
Download from: [https://osf.io/mfd2k/overview?view_only=7124056eb7194772b8df58562982308c]

Two conditions:
- `gaze_control_frames/` — normal face stimuli
- `gaze_quadbright_frames/` — QuadBright condition stimuli

Each stimulus folder contains 30 PNG frames (1920×1080px).

## Hardware

This experiment requires the YETI24 eye tracker.
- YETI24 build instructions and hardware setup: [https://github.com/amirashv/YET/tree/main/yeti24]
- The `libyeti24.py` driver file is included in this repository (extended for dual-camera support)
