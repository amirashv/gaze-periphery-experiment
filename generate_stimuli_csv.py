"""
generate_stimuli_csv.py
-----------------------
Run this once to generate Stimuli/Stimuli.csv from your existing stimulus folders.

Your folder structure:
  Stimuli/
    gaze_control_frames/
      gaze_0_control/   (30 frames: 0001.png ... 0030.png)
      gaze_1_control/
      ...
      gaze_30_control/
    gaze_quadbright_frames/
      gaze_0_quadbright/
      gaze_1_quadbright/
      ...
      gaze_30_quadbright/

Output: Stimuli/Stimuli.csv with columns:
  folder, condition, gaze_magnitude, n_frames, onset_frame
"""

import os
import pandas as pd

STIM_DIR = "Stimuli"
CONTROL_DIR   = os.path.join(STIM_DIR, "gaze_control_frames")
QUADBRIGHT_DIR = os.path.join(STIM_DIR, "gaze_quadbright_frames")

ONSET_FRAME = 15   # frame index where gaze change begins (0-indexed)
N_FRAMES    = 30   # total frames per stimulus

rows = []

for magnitude in range(0, 31):  # 0 to 30 degrees gaze change
    # Control
    folder_name = f"gaze_{magnitude}_control"
    folder_path = os.path.join(CONTROL_DIR, folder_name)
    if os.path.isdir(folder_path):
        frames = sorted([f for f in os.listdir(folder_path) if f.endswith(".png")])
        rows.append({
            "folder":        os.path.join("gaze_control_frames", folder_name),
            "condition":     "control",
            "gaze_magnitude": magnitude,
            "n_frames":      len(frames),
            "onset_frame":   ONSET_FRAME,
        })
    else:
        print(f"WARNING: missing {folder_path}")

    # QuadBright
    folder_name = f"gaze_{magnitude}_quadbright"
    folder_path = os.path.join(QUADBRIGHT_DIR, folder_name)
    if os.path.isdir(folder_path):
        frames = sorted([f for f in os.listdir(folder_path) if f.endswith(".png")])
        rows.append({
            "folder":        os.path.join("gaze_quadbright_frames", folder_name),
            "condition":     "quadbright",
            "gaze_magnitude": magnitude,
            "n_frames":      len(frames),
            "onset_frame":   ONSET_FRAME,
        })
    else:
        print(f"WARNING: missing {folder_path}")

df = pd.DataFrame(rows)
out_path = os.path.join(STIM_DIR, "Stimuli.csv")
df.to_csv(out_path, index=False)
print(f"Written {len(df)} stimulus entries to {out_path}")
print(df.head(10).to_string())
