"""
run.py  —  YETI24 Gaze-Change Detection Experiment
====================================================
Peripheral gaze-change detection: psychophysical 2AFC with eye tracking.

State machine
-------------
  Detect  →  Calibration  →  Validate
    →  Intro  →  prepareStimulus  →  Quick
    →  Fixation  →  Stimulus  →  Response  →  ITI
    →  (every 200 trials) Break → Calibration → Validate → Quick
    →  (when both staircases complete) ThankYou

Key design points
-----------------
- One continuous session — no blocks
- Condition (control / quadbright) randomised per trial
- Two staircases run in parallel, one per condition
- Break every 200 trials: 5-min countdown, then re-calibration
- All data saved continuously to single files per participant
- Q key = left changed, P key = right changed
"""

import logging as log
import os
import sys
import time
import random
import math

import numpy as np
import pandas as pd
import pygame as pg
from pygame.locals import *

import libyeti24 as yeti24
from libyeti24 import draw_text
from geometry import ScreenGeometry
from staircase import StaircaseBlock

FACE_DISPLAY_SIZE = (150, 261)
CROP_LEFT, CROP_TOP, CROP_RIGHT, CROP_BOTTOM = 672, 5, 1252, 996
FIXATION_DOT_RADIUS   = 10
FIXATION_DOT_COLOR    = (255, 120, 0)
RESPONSE_KEYS         = {K_q: 'left', K_p: 'right'}
SACCADE_THRESHOLD_DEG = 8.0
DRIFT_QUALITY_MIN     = 0.05
DRIFT_THRESHOLD       = 60
BREAK_EVERY_N_TRIALS  = 200
BREAK_DURATION_S      = 300
TEXT_COLOR = (255, 255, 255)  # white



def eye_quality_ok(yet):
    try:
        qb = yet.quad_bright
        if qb is None or len(qb) < 4:
            return False
        mean = sum(qb) / 4
        variance = sum((v - mean) ** 2 for v in qb) / 4
        return variance > DRIFT_QUALITY_MIN
    except Exception:
        return False


class AnimatedStimulus:
    _cache = {}   # shared across all instances

    def __init__(self, folder, surface, face_center, display_size=FACE_DISPLAY_SIZE, mirror=False):
        self.folder        = os.path.join("Stimuli", folder)
        self.surface       = surface
        self.face_center   = face_center
        self.display_size  = display_size
        self.mirror        = mirror
        self.frames        = []
        self.current_frame = 0

    # Crop region — face location in 1920x1080 stimulus frames
    def load(self):
        cache_key = (self.folder, self.mirror)
        if cache_key in AnimatedStimulus._cache:
            self.frames = AnimatedStimulus._cache[cache_key]
            self.current_frame = 0
            return True
        try:
            files = sorted([f for f in os.listdir(self.folder) if f.endswith(".png")])
            if not files:
                log.error(f"No PNG frames in {self.folder}")
                return False
            self.frames = []
            for f in files:
                img = pg.image.load(os.path.join(self.folder, f)).convert_alpha()
                # Crop to face region before scaling
                crop_rect = pg.Rect(CROP_LEFT, CROP_TOP,
                                    CROP_RIGHT - CROP_LEFT,
                                    CROP_BOTTOM - CROP_TOP)
                img = img.subsurface(crop_rect).copy()
                img = pg.transform.smoothscale(img, self.display_size)
                if self.mirror:
                    img = pg.transform.flip(img, True, False)
                self.frames.append(img)
            AnimatedStimulus._cache[cache_key] = self.frames
            self.current_frame = 0
            return True
        except Exception as e:
            log.error(f"Failed to load {self.folder}: {e}")
            return False

    def n_frames(self):
        return len(self.frames)

    def set_frame(self, idx):
        self.current_frame = max(0, min(idx, len(self.frames) - 1))

    def draw(self):
        if not self.frames:
            return
        img = self.frames[self.current_frame]
        w, h = img.get_size()
        x = max(0, self.face_center[0] - w // 2)
        x = min(x, self.surface.get_width() - w)
        y = max(0, self.face_center[1] - h // 2)
        y = min(y, self.surface.get_height() - h)
        self.surface.blit(img, (x, y))

def preload_all_stimuli(stim_table):
    print("Preloading stimuli...")
    all_folders = stim_table['folder'].unique()
    dummy_surface = pg.Surface(FACE_DISPLAY_SIZE)
    for folder in all_folders:
        for mirror in [True, False]:
            stim = AnimatedStimulus(folder, dummy_surface, (0,0))
            stim.mirror = mirror
            stim.load()
            pg.event.pump()
    print(f"Preloaded {len(AnimatedStimulus._cache)} stimulus sets into cache.")

def detect_saccade_toward(yet, geo, face_center_x):
    if not hasattr(yet, 'eye_pos'):
        return False
    dist_px  = abs(yet.eye_pos[0] - face_center_x)
    dist_deg = geo.px_to_deg(dist_px)
    return dist_deg <= SACCADE_THRESHOLD_DEG


def main():

    Yet = yeti24.YETI24(USB, SURF)
    if not Yet.connected:
        log.error("YETI24 could not connect with USB " + str(USB))
        sys.exit()

    geo = ScreenGeometry(
        screen_w_px     = SURF_SIZE[0],
        screen_h_px     = SURF_SIZE[1],
        screen_w_cm     = SCREEN_W_CM,
        viewing_dist_cm = VIEWING_DIST_CM,
    )
    print(geo.summary())

    max_ecc       = geo.max_eccentricity_deg(face_half_width_px=FACE_DISPLAY_SIZE[0] // 2)
    fix_radius_px = geo.fixation_radius_px(FIXATION_RADIUS_DEG)

    stim_table      = pd.read_csv(STIM_PATH)
    control_mags    = sorted(stim_table[(stim_table.condition == 'control')
                             & (stim_table.gaze_magnitude > 0)]['gaze_magnitude'].unique().tolist())
    quadbright_mags = sorted(stim_table[(stim_table.condition == 'quadbright')
                             & (stim_table.gaze_magnitude > 0)]['gaze_magnitude'].unique().tolist())

    ecc_range = (5.0, min(28.0, max_ecc))

    staircases = {
        'control':    StaircaseBlock('control',    control_mags,    ecc_range, n_trials=TRIALS_PER_STAIRCASE),
        'quadbright': StaircaseBlock('quadbright', quadbright_mags, ecc_range, n_trials=TRIALS_PER_STAIRCASE),
    }

    Cal  = yeti24.Calib(SURF)
    QCal = yeti24.Calib(SURF, pro_positions=[0.5, 0.5])
    Yet.init_eye_detection(EYECASC)

    total_trial_count    = 0
    next_break_at        = BREAK_EVERY_N_TRIALS
    break_start_time     = None
    current_condition    = None
    trial_ecc            = None
    trial_ecc_proposed   = None
    trial_mag            = None
    trial_side           = None
    stim_left            = None
    stim_right           = None
    pre_change_dur       = None
    t_fixation_start     = 0.0
    t_stim_start         = 0.0
    t_onset              = None
    onset_eye_pos        = None
    srt_ms               = None
    keyboard_rt_ms = None
    fixation_ok          = True
    trial_valid          = True
    key_response         = None
    drift_frame_count    = 0
    frame_idx            = 0
    t_iti_start          = 0.0
    next_stim_left  = None
    next_stim_right = None
    next_trial_ready = False
    _frame_cache = {}
    stimuli_preloaded   = False
    t_fixation_start_absolute = 0.0

    # Preload all stimuli into cache before experiment starts
    preload_all_stimuli(stim_table)
    stimuli_preloaded = True
    print("Stimuli ready.")

    STATE = "Detect"
    clock = pg.time.Clock()

    def get_stim_folder(condition, magnitude):
        rows = stim_table[stim_table.condition == condition].copy()
        rows['diff'] = (rows['gaze_magnitude'] - magnitude).abs()
        return rows.loc[rows['diff'].idxmin(), 'folder']

    def both_complete():
        return all(sc.is_complete() for sc in staircases.values())

    def choose_condition():
        active = [c for c, sc in staircases.items() if not sc.is_complete()]
        return random.choice(active)

    def prepare_trial():
        nonlocal current_condition, trial_ecc, trial_ecc_proposed, trial_mag
        nonlocal trial_side, stim_left, stim_right, pre_change_dur
        nonlocal frame_idx, t_onset, onset_eye_pos, srt_ms
        nonlocal fixation_ok, trial_valid, key_response
        t_fixation_start_absolute = time.time()


        current_condition           = choose_condition()
        sc                          = staircases[current_condition]
        trial_ecc_proposed, trial_mag = sc.propose_trial()
        trial_ecc  = min(trial_ecc_proposed, max_ecc)
        trial_side = random.choice(['left', 'right'])

        pos_left  = geo.ecc_to_screen_pos(trial_ecc, 'left')
        pos_right = geo.ecc_to_screen_pos(trial_ecc, 'right')

        folder_change = get_stim_folder(current_condition, trial_mag)
        folder_stable = get_stim_folder(current_condition, 0)

        if trial_side == 'left':
            stim_left  = AnimatedStimulus(folder_change, SURF, pos_left,  mirror=True)
            stim_right = AnimatedStimulus(folder_stable, SURF, pos_right, mirror=False)
        else:
            stim_left  = AnimatedStimulus(folder_stable, SURF, pos_left,  mirror=True)
            stim_right = AnimatedStimulus(folder_change, SURF, pos_right, mirror=False)

        stim_left.load()
        stim_right.load()
        stim_left.set_frame(0)
        stim_right.set_frame(0)

        pre_change_dur = random.randint(PRE_CHANGE_MIN_MS, PRE_CHANGE_MAX_MS)
        frame_idx      = 0
        t_onset        = None
        onset_eye_pos  = None
        srt_ms         = None
        fixation_ok    = True
        trial_valid    = True
        key_response   = None

        pg.event.pump()   # keep pygame responsive
        print(f"DEBUG trial {total_trial_count+1}: cond={current_condition}, "
            f"ecc={trial_ecc:.1f}°, mag={trial_mag}, side={trial_side}")

    def draw_fixation():
        pg.draw.circle(SURF, FIXATION_DOT_COLOR, (geo.cx, geo.cy), FIXATION_DOT_RADIUS)

    def fixation_maintained():
        if not hasattr(Yet, 'eye_pos'):
            return True
        dx = Yet.eye_pos[0] - geo.cx
        dy = Yet.eye_pos[1] - geo.cy
        dist = math.sqrt(dx*dx + dy*dy)
        dist_deg = geo.px_to_deg(dist)
        return dist <= fix_radius_px

    def save_data():
        if not Yet.data.empty:
            # Append if file exists, write header only if new file
            write_header = not os.path.exists(RESULT_FILE_GAZE)
            Yet.data.to_csv(RESULT_FILE_GAZE, index=False,
                            mode='a', header=write_header)
        all_trials = []
        for sc in staircases.values():
            all_trials.extend(sc.history)
        if all_trials:
            pd.DataFrame(all_trials).to_csv(RESULT_FILE_TRIALS, index=False)
        summaries = []
        for cond, sc in staircases.items():
            df = sc.threshold_summary()
            df['condition'] = cond
            summaries.append(df)
        if summaries:
            pd.concat(summaries, ignore_index=True).to_csv(RESULT_FILE_THRESHOLDS, index=False)

    # ==================================================================
    # MAIN LOOP
    # ==================================================================

    while True:

        for event in pg.event.get():
            if event.type == QUIT:
                save_data(); Yet.release(); pg.quit(); sys.exit()

            key_down    = event.type == KEYDOWN
            key_forward = key_down and event.key == K_SPACE
            key_back    = key_down and event.key == K_BACKSPACE

            if key_down and event.key == K_ESCAPE:
                save_data(); Yet.release(); pg.quit(); sys.exit()

            if key_down and event.key == K_r:      # ← ADD THIS BLOCK
                Cal.reset()
                Yet.reset()
                STATE = "Calibration"

            if STATE == "Detect":
                if Yet.eye_detected and key_forward:
                    Yet.update_eye_frame()
                    STATE = "Calibration"

            elif STATE == "Calibration":
                if key_forward:
                    Yet.update_frame()
                    Yet.update_eye_frame()
                    Yet.update_quad_bright()
                    Yet.record_calib_data(Cal.active_pos())
                    if Cal.remaining() > 0:
                        Cal.next()
                    else:
                        Yet.train()
                        test_input = np.array(Yet.quad_bright[0:4]).reshape(1,4)
                        test_pred  = Yet.model_L.predict(test_input)
                        STATE = "Validate"
                elif key_back:
                    STATE = "Detect"

            elif STATE == "Validate":
                if key_forward:
                    if total_trial_count == 0:
                        STATE = "Intro"
                    else:
                        prepare_trial()
                        STATE = "Quick"
                elif key_back:
                    Cal.reset(); Yet.reset()
                    STATE = "Calibration"

            elif STATE == "Intro":
                if key_forward:
                    prepare_trial()
                    STATE = "Quick"

            elif STATE == "Quick":
                if key_forward:
                    Yet.update_offsets(QCal.active_pos())
                    t_fixation_start = time.time()
                    STATE = "Fixation"

            elif STATE == "Response":
                if key_down and event.key in RESPONSE_KEYS:
                    key_response = RESPONSE_KEYS[event.key]
                    correct      = (key_response == trial_side)
                    keyboard_rt_ms = (time.time() - t_onset) * 1000 if t_onset else None
                    if srt_ms is None:
                        srt_ms = (time.time() - t_onset) * 1000 if t_onset else None
                    staircases[current_condition].register_response(
                        eccentricity = trial_ecc_proposed,
                        magnitude    = trial_mag,
                        correct      = correct,
                        srt_ms       = srt_ms,
                        valid        = trial_valid,
                        pre_change_ms = pre_change_dur,
                        keyboard_rt_ms = keyboard_rt_ms,
                    )
                    total_trial_count += 1
                    save_data()
                    t_iti_start = time.time()
                    iti_dur_ms  = random.randint(ITI_MIN_MS, ITI_MAX_MS)
                    STATE = "ITI"

            elif STATE == "Break":
                elapsed_break = time.time() - break_start_time
                if key_forward and elapsed_break >= BREAK_DURATION_S:
                    save_data()          # save everything first
                    Yet.data = Yet.data.iloc[0:0]  # clear gaze buffer
                    Cal.reset()
                    Yet.reset()
                    STATE = "Calibration"

            elif STATE == "ThankYou":
                if key_forward:
                    save_data(); Yet.release(); pg.quit(); sys.exit()

        # --- Automatic transitions ---

        if STATE == "ITI":
            if not next_trial_ready and not both_complete():
                prepare_trial()
                next_trial_ready = True
            if (time.time() - t_iti_start) * 1000 >= iti_dur_ms:
                next_trial_ready = False
                if both_complete():
                    STATE = "ThankYou"
                elif total_trial_count >= next_break_at:
                    break_start_time = time.time()
                    next_break_at   += BREAK_EVERY_N_TRIALS
                    STATE = "Break"
                else:
                    STATE = "Quick"

        if STATE == "Fixation":
            if not fixation_maintained():
                t_fixation_start = time.time()
            if (time.time() - t_fixation_start) * 1000 >= pre_change_dur:
                t_stim_start = time.time()
                frame_idx = 0
                stim_left.set_frame(0)
                stim_right.set_frame(0)
                STATE = "Stimulus"

        if STATE == "Stimulus":
            elapsed_stim_ms   = (time.time() - t_stim_start) * 1000
            total_ms_per_frame = (GAZE_CHANGE_DURATION_MS + POST_CHANGE_HOLD_MS) / N_TOTAL_FRAMES
            new_frame_idx = min(int(elapsed_stim_ms / total_ms_per_frame), N_TOTAL_FRAMES - 1)
            if new_frame_idx != frame_idx:
                frame_idx = new_frame_idx
                if trial_side == 'left':
                    stim_left.set_frame(frame_idx); stim_right.set_frame(0)
                else:
                    stim_right.set_frame(frame_idx); stim_left.set_frame(0)
            if frame_idx >= ONSET_FRAME and t_onset is None:
                t_onset       = time.time()
                onset_eye_pos = getattr(Yet, 'eye_pos', (geo.cx, geo.cy))
            if t_onset is None and not fixation_maintained():
                fixation_ok = False; trial_valid = False
            if t_onset is not None and srt_ms is None:
                if hasattr(Yet, 'eye_pos'):
                    face_center_x = stim_left.face_center[0] if trial_side == 'left' else stim_right.face_center[0]
                    dist_to_face = geo.px_to_deg(abs(Yet.eye_pos[0] - face_center_x))
                    print(f"dist_to_face={dist_to_face:.2f}° threshold={SACCADE_THRESHOLD_DEG}°")
                face_center_x = stim_left.face_center[0] if trial_side == 'left' else stim_right.face_center[0]
                if detect_saccade_toward(Yet, geo, face_center_x):
                    candidate_srt = (time.time() - t_onset) * 1000
                    if candidate_srt > 50:  # ignore detections within 50ms of onset (false positives)
                        srt_ms = candidate_srt
                        print(f"SRT detected: {srt_ms:.0f}ms")
            if elapsed_stim_ms >= GAZE_CHANGE_DURATION_MS + POST_CHANGE_HOLD_MS:
                STATE = "Response"

        # --- Frame processing ---

        if STATE == "Detect":
            Yet.update_frame(); Yet.detect_eye()
            if Yet.eye_detected:
                Yet.update_eye_frame()

        elif STATE in ("Validate", "Quick", "Fixation"):
            Yet.update_frame(); Yet.update_eye_frame()
            Yet.update_quad_bright(); Yet.update_eye_pos()
            drift_frame_count = drift_frame_count + 1 if not eye_quality_ok(Yet) else 0

        elif STATE == "Stimulus":
            Yet.update_frame(); Yet.update_eye_frame()
            Yet.update_quad_bright(); Yet.update_eye_pos()
            if stim_left is not None:
                Yet.eye_stim_L = getattr(Yet, 'eye_pos', (0, 0))
                Yet.eye_stim_R = getattr(Yet, 'eye_pos', (0, 0))
                Yet.eye_stim_F = getattr(Yet, 'eye_pos', (0, 0))
                Yet.eye_stim   = getattr(Yet, 'eye_pos', (0, 0))
                Yet.eye_stim_L_pro = (0.0, 0.0)
                Yet.eye_stim_R_pro = (0.0, 0.0)
                Yet.eye_stim_F_pro = (0.0, 0.0)
                stim_id = (f"trial={total_trial_count+1}|cond={current_condition}|"
                           f"ecc={trial_ecc:.1f}|mag={trial_mag:.0f}|side={trial_side}")
                Yet.record(EXP_ID + EXPERIMENTER, PART_ID, stim_id)

        elif STATE == "Response":
            Yet.update_frame(); Yet.update_eye_frame()
            Yet.update_quad_bright(); Yet.update_eye_pos()

        # --- Rendering ---

        SURF.fill(BACKGR_COL)

        if STATE == "Detect":
            if Yet.eye_detected:
                draw_text("Eyes detected! Press Space to continue.", SURF, (0.05, 0.85), FONT, color=TEXT_COLOR)
            else:
                draw_text("Detecting eyes — please look at camera.", SURF, (0.05, 0.85), FONT, color=TEXT_COLOR)
            if Yet.frame is not None:
                preview = yeti24.frame_to_surf(Yet.frame,
                    (int(SURF_SIZE[0]*0.5), int(SURF_SIZE[1]*0.5)))
                SURF.blit(preview, (int(SURF_SIZE[0]*0.25), int(SURF_SIZE[1]*0.25)))

        elif STATE == "Calibration":
            Cal.draw()
            draw_text("Follow the orange dot and press Space.", SURF, (0.05, 0.9), Font, color=TEXT_COLOR)

        elif STATE == "Validate":
            Yet.draw_follow(SURF)
            draw_text("Validation: look around. Space = OK, Backspace = redo calibration.",
                      SURF, (0.05, 0.9), Font, color=TEXT_COLOR)

        elif STATE == "Intro":
            draw_text("Gaze-Change Detection", SURF, (0.5, 0.20), FONT, color=TEXT_COLOR, center=True)
            draw_text("Keep your gaze on the central orange dot at all times.", SURF, (0.5, 0.35), Font, color=TEXT_COLOR, center=True)
            draw_text("Two faces will appear. One of them will change gaze direction.", SURF, (0.5, 0.43), Font, color=TEXT_COLOR, center=True)
            draw_text("Look at the face that changed.", SURF, (0.5, 0.51), Font, color=TEXT_COLOR, center=True)
            draw_text("Then press  Q  if the LEFT face changed,", SURF, (0.5, 0.59), Font, color=TEXT_COLOR, center=True)
            draw_text("or press  P  if the RIGHT face changed.", SURF, (0.5, 0.67), Font, color=TEXT_COLOR, center=True)
            draw_text("Press Space to begin.", SURF, (0.5, 0.80), Font, color=TEXT_COLOR, center=True)

        elif STATE == "Quick":
            if stim_left and stim_right:
                stim_left.draw(); stim_right.draw()
            QCal.draw(); Yet.draw_follow(SURF)
            draw_text("Look at the orange dot and press Space.", SURF, (0.05, 0.75), Font, color=TEXT_COLOR)
            if drift_frame_count > DRIFT_THRESHOLD // 2:
                draw_text("Warning: eye signal unstable", SURF, (0.1, 0.05), Font, color=TEXT_COLOR)

        elif STATE == "Fixation":
            if stim_left and stim_right:
                stim_left.draw()
                stim_right.draw()
            draw_fixation()
            elapsed_abs = time.time() - t_fixation_start_absolute
            if elapsed_abs > 5.0:
                draw_text("Please look at the orange dot...",
                          SURF, (0.5, 0.85), Font,
                          color=TEXT_COLOR, center=True)

        elif STATE == "Stimulus":
            if stim_left and stim_right:
                stim_left.draw()
                stim_right.draw()
            draw_fixation()

        elif STATE == "Response":
            draw_text("Which face changed?   Q = Left    P = Right", SURF, (0.5, 0.5), FONT, color=TEXT_COLOR, center=True)


        elif STATE == "ITI":
            pass

        elif STATE == "Break":
            elapsed = time.time() - break_start_time
            remaining = max(0, BREAK_DURATION_S - elapsed)
            mins, secs = int(remaining // 60), int(remaining % 60)
            draw_text("Mandatory break — please rest your eyes.", SURF, (0.5, 0.30), FONT, color=TEXT_COLOR, center=True)
            draw_text(f"Time remaining: {mins}:{secs:02d}", SURF, (0.5, 0.45), Font, color=TEXT_COLOR, center=True)
            draw_text(f"Trials completed: {total_trial_count}", SURF, (0.5, 0.55), Font, color=TEXT_COLOR, center=True)
            if remaining <= 0:
                draw_text("Break over — press Space to continue.",
                          SURF, (0.5, 0.68), Font, color=TEXT_COLOR, center=True)

        elif STATE == "ThankYou":
            draw_text("Thank you for participating!", SURF, (0.5, 0.45), FONT, color=TEXT_COLOR, center=True)
            draw_text("Data has been saved. Press Space to exit.",
                      SURF, (0.5, 0.6), Font, color=TEXT_COLOR, center=True)

        pg.display.update()
        clock.tick(60)


def read_config(path="Config.csv"):
    global USB, EXP_ID, EXPERIMENTER, SURF_SIZE, STIM_FILE
    global SCREEN_W_CM, SCREEN_H_CM, VIEWING_DIST_CM
    global TRIALS_PER_STAIRCASE, FIXATION_RADIUS_DEG
    global PRE_CHANGE_MIN_MS, PRE_CHANGE_MAX_MS
    global N_TOTAL_FRAMES, ONSET_FRAME, N_GAZE_CHANGE_FRAMES
    global GAZE_CHANGE_DURATION_MS, POST_CHANGE_HOLD_MS
    global ITI_MIN_MS, ITI_MAX_MS

    cfg = pd.read_csv(path, index_col=0).squeeze().to_dict()
    USB                     = (int(cfg["USB_L"]), int(cfg["USB_R"]))
    EXP_ID                  = str(cfg["EXP_ID"])
    EXPERIMENTER            = str(cfg["EXPERIMENTER"])
    SURF_SIZE               = (int(cfg["WIDTH"]), int(cfg["HEIGHT"]))
    SCREEN_W_CM             = float(cfg["SCREEN_W_CM"])
    SCREEN_H_CM             = float(cfg["SCREEN_H_CM"])
    VIEWING_DIST_CM         = float(cfg["VIEWING_DIST_CM"])
    STIM_FILE               = str(cfg["STIM_FILE"])
    TRIALS_PER_STAIRCASE    = int(cfg["TRIALS_PER_STAIRCASE"])
    FIXATION_RADIUS_DEG     = float(cfg["FIXATION_RADIUS_DEG"])
    PRE_CHANGE_MIN_MS       = int(cfg["PRE_CHANGE_MIN_MS"])
    PRE_CHANGE_MAX_MS       = int(cfg["PRE_CHANGE_MAX_MS"])
    N_TOTAL_FRAMES          = int(cfg["GAZE_CHANGE_FRAMES"])
    ONSET_FRAME             = int(cfg["GAZE_CHANGE_ONSET_FRAME"])
    N_GAZE_CHANGE_FRAMES    = N_TOTAL_FRAMES - ONSET_FRAME
    GAZE_CHANGE_DURATION_MS = 100
    POST_CHANGE_HOLD_MS     = int(cfg["POST_CHANGE_HOLD_MS"])
    ITI_MIN_MS              = int(cfg["ITI_MIN_MS"])
    ITI_MAX_MS              = int(cfg["ITI_MAX_MS"])


def setup():
    global WD, STIM_DIR, STIM_PATH, RESULT_DIR
    global PART_ID, RESULT_FILE_GAZE, RESULT_FILE_TRIALS, RESULT_FILE_THRESHOLDS
    global EYECASC
    WD         = "."
    os.chdir(WD)
    STIM_DIR   = os.path.join(WD, "Stimuli")
    STIM_PATH  = os.path.join(STIM_DIR, "Stimuli.csv")
    RESULT_DIR = "Data"
    os.makedirs(RESULT_DIR, exist_ok=True)
    PART_ID                = str(int(time.time()))
    RESULT_FILE_GAZE       = os.path.join(RESULT_DIR, f"gaze_{EXP_ID}_{PART_ID}.csv")
    RESULT_FILE_TRIALS     = os.path.join(RESULT_DIR, f"trials_{EXP_ID}_{PART_ID}.csv")
    RESULT_FILE_THRESHOLDS = os.path.join(RESULT_DIR, f"thresholds_{EXP_ID}_{PART_ID}.csv")
    EYECASC                = "haarcascade_eye.xml"
    log.basicConfig(filename="Yet.log", level=log.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")


def init_pygame():
    global FONT, Font, font, BACKGR_COL, SURF
    pg.init()
    FONT = pg.font.Font("freesansbold.ttf", int(min(SURF_SIZE) / 20))
    Font = pg.font.Font("freesansbold.ttf", int(min(SURF_SIZE) / 40))
    font = pg.font.Font("freesansbold.ttf", int(min(SURF_SIZE) / 60))
    pg.display.set_mode(SURF_SIZE)
    pg.display.set_caption("Gaze-Change Detection")
    SURF       = pg.display.get_surface()
    BACKGR_COL = (57, 57, 57)


read_config()
setup()
init_pygame()
main()
