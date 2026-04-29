import os
import random

import numpy as np
from numpy import array as ary

"""Numpy for data manipulation"""

import itertools

from sklearn import linear_model as lm

"""Using linear models from Sklearn"""
import pandas as pd

"""Using Pandas data frames"""
import logging as log
from time import sleep, time

# CV
import cv2 as cv
import pygame as pg
from pygame.draw import circle

"""OpenCV computer vision library"""


def main():
    print("main")
    pg.init()
    pg.display.set_mode((800, 800))
    SCREEN = pg.display.get_surface()
    Cal = Calib(SCREEN)
    print(str(Cal.targets))
    print("active: " + str(Cal.active))
    Cal.draw()
    pg.display.update()
    sleep(2)


def draw_text(
    text: str,
    Surf: pg.Surface,
    rel_pos: tuple,
    Font: pg.font.Font,
    color=(0, 0, 0),
    center=False,
):
    surf_size = Surf.get_size()
    x, y = np.array(rel_pos) * np.array(surf_size)
    rendered_text = Font.render(text, True, color)
    # retrieving the abstract rectangle of the text box
    box = rendered_text.get_rect()
    # this sets the x and y coordinates
    if center:
        box.center = (x, y)
    else:
        box.topleft = (x, y)
    # This puts the pre-rendered object on the surface
    Surf.blit(rendered_text, box)


class Stimulus:
    stim_dir = "Stimuli/"

    def __init__(self, entry):
        if isinstance(entry, pd.DataFrame):
            entry = entry.to_dict()
        self.file = entry["File"]
        self.path = os.path.join(self.stim_dir, self.file)
        self.size = ary((entry["width"], entry["height"]))

    def load(self, surface: pg.Surface, scale=True):
        image = pg.image.load(self.path)
        # image = pg.image.convert()
        self.surface = surface
        self.surf_size = ary(self.surface.get_size())
        if scale:
            self.scale = min(self.surf_size / self.size)
            scale_to = ary(self.size * self.scale).astype(int)
            self.image = pg.transform.smoothscale(image, scale_to)
            self.size = self.image.get_size()
        else:
            self.scale = 1
        self.pos = ary((self.surf_size - self.size) / 2).astype(int)

    def draw(self):
        self.surface.blit(self.image, self.pos)

    def draw_preview(self):
        blur = ary(self.surf_size / 4).astype("int")  # strong blur (~25% of screen dims)
        img = pg.surfarray.array3d(self.image)
        img = cv.blur(img, blur).astype("uint8")
        # img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        img = pg.surfarray.make_surface(img)
        self.surface.blit(img, self.pos)

    def average_brightness(self):
        return pg.surfarray.array3d(self.image).mean()


class StimulusSet:
    def __init__(self, path):
        self.table = pd.read_csv(path)
        self.Stimuli = []
        for index, row in self.table.iterrows():
            this_stim = Stimulus(row)
            self.Stimuli.append(this_stim)
        self.active = 0

    def n(self):
        return len(self.Stimuli)

    def remaining(self):
        return len(self.Stimuli) - self.active

    def next(self):
        if self.active < len(self.Stimuli):
            this_stim = self.Stimuli[self.active]
            self.active += 1
            return True, this_stim
        else:
            return False, None

    def reset(self):
        self.active = 0

    def pop(self):
        return self.Stimuli.pop()

    def shuffle(self, reset=True):
        self.reset()
        random.shuffle(self.Stimuli)


def frame_to_surf(frame, dim):
    img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  # convert BGR (cv) to RGB (Pygame)
    img = np.rot90(img)  # rotate coordinate system
    surf = pg.surfarray.make_surface(img)
    surf = pg.transform.smoothscale(surf, dim)
    return surf


class YETI24:
    """
    Dual-camera (left/right) eye-tracking prototype using two USB cameras.

    Pipeline (typical loop):
        1) update_frame()        -> read frames from both cameras
        2) detect_eye()          -> optionally / typically during detection phase; ROI may then be reused
        3) update_eye_frame()    -> crop eye ROIs (L/R), grayscale, resize to fixed size
        4) update_quad_bright()  -> compute 4 quadrant-mean brightness features per eye (8 total)
        5) (calibration) record_calib_data(target_pos) over multiple targets
        6) train()               -> train two monocular linear models:
                                  - model_L: 4 features -> (x,y)
                                  - model_R: 4 features -> (x,y)
        7) update_eye_pos()      -> predict left/right gaze, fuse by averaging, clip fused point to screen bounds
        8) update_eye_stim()     -> transform gaze from screen pixels to stimulus coordinates
        9) record()              -> append a row to self.data (L/R + fused)

    Coordinate conventions:
        - eye_pos_* : screen pixel coordinates (x,y) on the pygame surface
        - eye_pro_* : proportional coordinates for L/R (screen-relative after update_eye_pos; stimulus-relative after update_eye_stim)
        - eye_stim_*: stimulus-local pixel coordinates (x,y) relative to drawn stimulus
        - eye_pro   : proportional coordinates for the fused point (same convention as above)

    Notes:
        - Eye detection is considered successful only if BOTH cameras detect exactly one eye.
        - The "fused" point (self.eye_pos) is the average of left/right predictions.
    """

    frame = None
    new_frame = False
    connected = False
    cascade = False
    eye_detection = False
    eye_detected = False
    eye_frame_coords = (0, 0, 0, 0)
    eye_frame = []
    quad_bright = (0, 0, 0, 0, 0, 0, 0, 0)
    offsets = (0, 0)
    data_cols = (
        "Exp",
        "Part",
        "Stim",
        "time",
        "xL",
        "yL",
        "xL_pro",
        "yL_pro",
        "xR",
        "yR",
        "xR_pro",
        "yR_pro",
        # keep fused
        "xF",
        "yF",
        "xF_pro",
        "yF_pro",
        "blink"
    )

    def __init__(self, usb: int, surface: pg.Surface) -> None:
        """
        Create a dual-camera YETI24 instance and connect to both cameras.

        Parameters
        ----------
        usb :
            Camera selector. Can be:
              - int: uses (usb) as left and (usb+1) as right (fallback convention)
              - tuple/list of length 2: (usb_left, usb_right)
        surface :
            Pygame surface used as screen reference for pixel coordinates.

        Side effects / State created
        ----------------------------
        - device_L, device_R : cv.VideoCapture objects
        - frame_L, frame_R   : most recent frames
        - calib_data         : numpy array with shape (n_samples, 10)
                              [4 quad L, 4 quad R, target_x, target_y]
        - data               : pandas DataFrame for recorded samples
        """
        self.connected = False
        self.surface = surface
        self.surf_size = self.surface.get_size()
        self.eye_detected_L = False
        self.eye_detected_R = False
        self._roi_stable_L = False  # ADD HERE
        self._roi_stable_R = False  # ADD HERE
        self._blink_holdoff = 0        # ADD
        self._last_stable_pos = None   # ADD

        # Interpreting usb argument
        if isinstance(usb, (tuple, list)) and len(usb) == 2:
            self.usb_L, self.usb_R = int(usb[0]), int(usb[1])
        else:
            self.usb_L = int(usb)
            self.usb_R = int(usb) + 1  # fallback convention

        try:
            self.device_L = cv.VideoCapture(self.usb_L)
            self.device_R = cv.VideoCapture(self.usb_R)
            self.connected = self.device_L.isOpened() and self.device_R.isOpened()
        except Exception as e:
            log.error(f"Could not connect USB devices L={self.usb_L}, R={self.usb_R}: {e}")
            self.connected = False

        # --- init state ---
        self.new_frame = False
        self.frame_L = None
        self.frame_R = None
        self.eye_frame_L = None
        self.eye_frame_R = None

        self.eye_detected_L = False
        self.eye_detected_R = False
        self.eye_frame_coords_L = (0, 0, 0, 0)
        self.eye_frame_coords_R = (0, 0, 0, 0)

        self.frame = None
        self.eye_frame = None

        self.quad_bright = (
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        )  # this will become 8 values (4 left + 4 right)

        if self.connected:
            self.fps = self.device_L.get(cv.CAP_PROP_FPS)
            self.frame_size = (
                int(self.device_L.get(cv.CAP_PROP_FRAME_WIDTH)),
                int(self.device_L.get(cv.CAP_PROP_FRAME_HEIGHT)),
            )
            self.calib_data = np.zeros(
                shape=(0, 10)
            )  # now 8 features + 2 target coords = 10 columns

            self.data = pd.DataFrame(columns=YETI24.data_cols, dtype="float64")
            self.data["Exp"].astype("category")
            self.data["Part"].astype("category")
            self.data["Stim"].astype("category")

            self.update_frame()

    def release(self):
        """
        Release both camera devices (if present).

        Call this when closing the application to free the USB cameras.
        """
        if hasattr(self, "device_L"):
            self.device_L.release()
        if hasattr(self, "device_R"):
            self.device_R.release()

    def init_eye_detection(self, cascade_file: str):
        """
        Load the Haar cascade used for eye detection.

        Parameters
        ----------
        cascade_file :
            Path to an OpenCV Haar cascade XML file (e.g. eye detector).

        Side effects
        ------------
        - self.cascade is set (cv.CascadeClassifier)
        - self.eye_detection flag enabled
        """
        self.eye_detection = False
        self.cascade = cv.CascadeClassifier(cascade_file)
        self.eye_detection = True

    def update_frame(self) -> np.ndarray:
        """
        Read a new frame from BOTH cameras.

        Returns
        -------
        bool
            True if both cameras delivered a non-empty frame on this call, else False.

        Side effects
        ------------
        - self.frame_L, self.frame_R updated when successful
        - self.new_frame set accordingly
        - self.frame may be set to a debug composite (side-by-side) for visualization
        """
        self.new_frame = False

        new_L, frame_L = self.device_L.read()
        new_R, frame_R = self.device_R.read()

        if (
            new_L
            and new_R
            and (not np.sum(frame_L) == 0)
            and (not np.sum(frame_R) == 0)
        ):
            self.new_frame = True
            self.frame_L = frame_L
            self.frame_R = frame_R

            try:
                h = min(self.frame_L.shape[0], self.frame_R.shape[0])
                L = getattr(self, "debug_L", self.frame_L)
                R = getattr(self, "debug_R", self.frame_R)
                self.frame = np.hstack([L[:h], R[:h]])
            except Exception:
                self.frame = self.frame_L

        return self.new_frame


    def _pad_roi(self, x, y, w, h, frame_shape, pad_x=0.2, pad_y=0.1):
        """
        Expand a detected ROI by `pad` fraction on each side.
        Gives the iris room to move without leaving the crop.
        """
        px = int(w * pad_x)
        py = int(h * pad_y)
        x0 = max(0, x - px)
        y0 = max(0, y - py)
        x1 = min(frame_shape[1], x + w + px)
        y1 = min(frame_shape[0], y + h + py)
        return (x0, y0, x1 - x0, y1 - y0)

    def _best_eye(self, detections, frame_shape):
        frame_h, frame_w = frame_shape[:2]
        min_area = (frame_w * frame_h) * 0.03  # raised from 0.01 to 0.03 — kills tiny boxes
        candidates = []
        for (x, y, w, h) in detections:
            area = w * h
            aspect = w / h if h > 0 else 0
            # Accept both landscape AND portrait orientations (sideways cameras)
            if area >= min_area and 0.25 <= aspect <= 4.0:
                candidates.append((x, y, w, h))
        if not candidates:
            return None
        return max(candidates, key=lambda e: e[2] * e[3])

    def detect_eye(self) -> bool:
        self.eye_detected_L = False
        self.eye_detected_R = False
        if not self.new_frame:
            return False

        gray_L = cv.cvtColor(self.frame_L, cv.COLOR_BGR2GRAY)
        gray_R = cv.cvtColor(self.frame_R, cv.COLOR_BGR2GRAY)
        gray_L = cv.equalizeHist(gray_L)
        gray_R = cv.equalizeHist(gray_R)

        Eyes_L = self.cascade.detectMultiScale(
            gray_L, scaleFactor=1.1, minNeighbors=2, minSize=(30, 30)
        )
        Eyes_R = self.cascade.detectMultiScale(
            gray_R, scaleFactor=1.1, minNeighbors=2, minSize=(30, 30)
        )

        self.debug_L = self.frame_L.copy()
        self.debug_R = self.frame_R.copy()

        # Draw ONLY the winning detection, not all raw cascade outputs
        best_L = self._best_eye(Eyes_L, self.frame_L.shape)
        if best_L is not None:
            # Fresh detection — update ROI and mark stable
            x, y, w, h = best_L
            self.eye_frame_coords_L = self._pad_roi(x, y, w, h, self.frame_L.shape)
            self._roi_stable_L = True
            cv.rectangle(self.debug_L,
                         (self.eye_frame_coords_L[0], self.eye_frame_coords_L[1]),
                         (self.eye_frame_coords_L[0] + self.eye_frame_coords_L[2],
                          self.eye_frame_coords_L[1] + self.eye_frame_coords_L[3]),
                         (0, 255, 0), 2)
        elif self._roi_stable_L:
            # No detection this frame but we had a good ROI — keep it, draw in amber
            cv.rectangle(self.debug_L,
                         (self.eye_frame_coords_L[0], self.eye_frame_coords_L[1]),
                         (self.eye_frame_coords_L[0] + self.eye_frame_coords_L[2],
                          self.eye_frame_coords_L[1] + self.eye_frame_coords_L[3]),
                         (0, 165, 255), 2)  # orange = holding last known position

        self.eye_detected_L = self._roi_stable_L  # detected as long as we ever had a ROI

        best_R = self._best_eye(Eyes_R, self.frame_R.shape)
        if best_R is not None:
            x, y, w, h = best_R
            self.eye_frame_coords_R = self._pad_roi(x, y, w, h, self.frame_R.shape)
            self._roi_stable_R = True
            cv.rectangle(self.debug_R,
                         (self.eye_frame_coords_R[0], self.eye_frame_coords_R[1]),
                         (self.eye_frame_coords_R[0] + self.eye_frame_coords_R[2],
                          self.eye_frame_coords_R[1] + self.eye_frame_coords_R[3]),
                         (0, 255, 0), 2)
        elif self._roi_stable_R:
            cv.rectangle(self.debug_R,
                         (self.eye_frame_coords_R[0], self.eye_frame_coords_R[1]),
                         (self.eye_frame_coords_R[0] + self.eye_frame_coords_R[2],
                          self.eye_frame_coords_R[1] + self.eye_frame_coords_R[3]),
                         (0, 165, 255), 2)

        self.eye_detected_R = self._roi_stable_R

        self.eye_detected = self.eye_detected_L and self.eye_detected_R
        return self.eye_detected

    def update_eye_frame(self) -> np.ndarray:
        """
        Crop and preprocess the eye regions (ROIs) for left and right cameras.

        Steps:
            - crop ROI from each frame using eye_frame_coords_L/R
            - convert to grayscale
            - resize to a fixed size for consistent feature extraction

        Returns
        -------
        tuple of (eye_frame_L, eye_frame_R) or None

        Side effects
        ------------
        - self.eye_frame_L and self.eye_frame_R updated (grayscale, fixed size)
        """

        if not self.new_frame:
            return None

        if self.frame_L is None or self.frame_R is None:
            return None


        FIXED_EYE_SIZE = (64, 48)  # (width, height). Try (60,36) or (80,60) if needed.

        def safe_crop(frame, coords):
                x, y, w, h = coords
                if w < 4 or h < 4:          # ROI too small to be valid
                    return None
                crop = frame[y:y+h, x:x+w]
                if crop.size == 0:
                    return None
                gray = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
                return cv.resize(gray, FIXED_EYE_SIZE, interpolation=cv.INTER_AREA)

        result_L = safe_crop(self.frame_L, self.eye_frame_coords_L)
        result_R = safe_crop(self.frame_R, self.eye_frame_coords_R)

        # Only update if both crops succeeded — keeps last good frame otherwise
        if result_L is not None and result_R is not None:
            self.eye_frame_L = result_L
            self.eye_frame_R = result_R

        return self.eye_frame_L, self.eye_frame_R

    def update_quad_bright(self) -> tuple:
        """
        Compute quadrant-mean brightness features for both eyes.

        For each eye ROI (fixed-size grayscale image), compute mean brightness of:
            NW, NE, SW, SE
        and concatenate:
            (L_NW, L_NE, L_SW, L_SE, R_NW, R_NE, R_SW, R_SE)

        Returns
        -------
        tuple
            8 brightness features (floats). If frames/ROIs are not ready, returns the last value.

        Side effects
        ------------
        - self.quad_bright updated
        """
        if not self.new_frame:
            return self.quad_bright

        # guard: eye frames might not be available yet
        if self.eye_frame_L is None or self.eye_frame_R is None:
            return self.quad_bright

        def quad(img):
            h, w = img.shape
            h2, w2 = h // 2, w // 2
            b_NW = np.mean(img[0:h2, 0:w2])
            b_SW = np.mean(img[h2:h, 0:w2])
            b_NE = np.mean(img[0:h2, w2:w])
            b_SE = np.mean(img[h2:h, w2:w])
            return (b_NW, b_NE, b_SW, b_SE)

        qb_L = quad(self.eye_frame_L)
        qb_R = quad(self.eye_frame_R)
        self.quad_bright = qb_L + qb_R

        return self.quad_bright

    def record_calib_data(self, target_pos: tuple) -> ary:
        """
        Append one calibration sample.

        Parameters
        ----------
        target_pos :
            (x,y) pixel position of the displayed calibration target on the pygame surface.

        Returns
        -------
        np.ndarray
            The appended row: [8 brightness features, target_x, target_y]

        Side effects
        ------------
        - self.calib_data grows by one row
        """
        new_data = np.append(self.quad_bright, ary(target_pos))
        self.calib_data = np.append(self.calib_data, [new_data], axis=0)
        return new_data

    def train(self):
        """
        Train two monocular linear regression models from calibration data.

        Uses:
            X_L = calib_data[:, 0:4]   (left eye quad brightness)
            X_R = calib_data[:, 4:8]   (right eye quad brightness)
            Y   = calib_data[:, 8:10]  (target screen position)

        Returns
        -------
        (LinearRegression, LinearRegression)
            (model_L, model_R)

        Side effects
        ------------
        - self.model_L and self.model_R created and stored
        """
        X_L = self.calib_data[:, 0:4]
        X_R = self.calib_data[:, 4:8]
        Y = self.calib_data[:, 8:10]

        self.model_L = lm.LinearRegression().fit(X_L, Y)
        self.model_R = lm.LinearRegression().fit(X_R, Y)

        return self.model_L, self.model_R

    def update_offsets(self, target_pos: tuple) -> tuple:
        """
        Compute and store a constant screen-space offset to align predictions to a known target.

        Convention:
            offsets = target_pos - eye_raw
            eye_pos = eye_raw + offsets

        Parameters
        ----------
        target_pos :
            (x,y) pixel position of a known target on screen.

        Returns
        -------
        tuple
            (offset_x, offset_y)

        Side effects
        ------------
        - self.offsets updated
        """
        if not hasattr(self, 'eye_raw'):
            log.warning("update_offsets called before eye_raw exists — offsets not updated")
            return self.offsets
        new_offsets = ary(target_pos) - ary(self.eye_raw)
        self.offsets = tuple(new_offsets)
        return self.offsets

    def reset_offsets(self) -> None:
        """
        Reset offsets to (0,0).
        """
        self.offsets = (0, 0)

    def is_blinking(self) -> bool:
        return False

    def update_eye_pos(self) -> tuple:
        """
        Predict gaze position from current brightness features.

        Steps:
            1) Split features into left (4) and right (4)
            2) Predict raw gaze positions with model_L and model_R
            3) Apply shared offsets to each monocular prediction
            4) Fuse by averaging left/right
            5) Clip fused point to screen bounds

        Returns
        -------
        (tuple, tuple)
            (eye_pos_L, eye_pos_R) in SCREEN PIXELS.
        """
        if not hasattr(self, 'model_L') or not hasattr(self, 'model_R'):
            return (0, 0), (0, 0)

        quad = ary(self.quad_bright)

        quad_L = quad[0:4].reshape(1, 4)
        quad_R = quad[4:8].reshape(1, 4)

        raw_L = self.model_L.predict(quad_L)[0, :]
        raw_R = self.model_R.predict(quad_R)[0, :]

        self.eye_raw_L = tuple(raw_L)
        self.eye_raw_R = tuple(raw_R)
        self.eye_raw = tuple((ary(self.eye_raw_L) + ary(self.eye_raw_R)) / 2.0)

        # Check blink BEFORE computing positions
        raw_fused = (ary(raw_L) + ary(raw_R)) / 2.0 + ary(self.offsets)

        BLINK_HOLDOFF_FRAMES = 8

        if self.is_blinking():
            self._blink_holdoff = BLINK_HOLDOFF_FRAMES
            if hasattr(self, '_last_stable_pos') and self._last_stable_pos is not None:
                self.eye_pos = self._last_stable_pos
            else:
                self.eye_pos = (self.surf_size[0] // 2, self.surf_size[1] // 2)  # safe fallback
            return getattr(self, 'eye_pos_L', (0,0)), getattr(self, 'eye_pos_R', (0,0))

        if hasattr(self, '_blink_holdoff') and self._blink_holdoff > 0:
            self._blink_holdoff -= 1
            if hasattr(self, '_last_stable_pos') and self._last_stable_pos is not None:
                self.eye_pos = self._last_stable_pos
            else:
                self.eye_pos = (self.surf_size[0] // 2, self.surf_size[1] // 2)  # safe fallback
            return getattr(self, 'eye_pos_L', (0,0)), getattr(self, 'eye_pos_R', (0,0))

        self.eye_pos_L = tuple(ary(self.eye_raw_L) + ary(self.offsets))
        self.eye_pos_R = tuple(ary(self.eye_raw_R) + ary(self.offsets))

        self.eye_pro_L = tuple(ary(self.eye_pos_L) / ary(self.surf_size))
        self.eye_pro_R = tuple(ary(self.eye_pos_R) / ary(self.surf_size))

        x = int(np.clip(raw_fused[0], 0, self.surf_size[0] - 1))
        y = int(np.clip(raw_fused[1], 0, self.surf_size[1] - 1))
        self.eye_pos = (x, y)
        self.eye_pro = (x / self.surf_size[0], y / self.surf_size[1])
        self._last_stable_pos = self.eye_pos


        return self.eye_pos_L, self.eye_pos_R

    def update_eye_stim(self, Stim: Stimulus) -> tuple:
        """
        Transform gaze position from screen coordinates to stimulus-local coordinates.

        Parameters
        ----------
        Stim :
            Stimulus object that has been loaded and positioned on the pygame surface.

        Returns
        -------
        (tuple, tuple)
            (eye_stim_L, eye_stim_R) in STIMULUS PIXELS.

        Side effects
        ------------
        - self.eye_stim_L, self.eye_stim_R updated
        - self.eye_stim (fused stim point) updated
        - self.eye_pro_* updated relative to Stim.size
        """
        if self._blink_holdoff > 0:
            # Ensure attributes exist even if never set yet
            if not hasattr(self, 'eye_stim_L'):
                self.eye_stim_L = (0, 0)
                self.eye_stim_R = (0, 0)
                self.eye_stim   = (0, 0)
                self.eye_pro_L  = (0, 0)
                self.eye_pro_R  = (0, 0)
                self.eye_pro    = (0, 0)
            return self.eye_stim_L, self.eye_stim_R

        offsets = ary(Stim.pos)
        scale = ary(Stim.scale)

        self.eye_stim_L = tuple((ary(self.eye_pos_L) - offsets) / scale)
        self.eye_stim_R = tuple((ary(self.eye_pos_R) - offsets) / scale)

        self.eye_pro_L = tuple(ary(self.eye_stim_L) / ary(Stim.size))
        self.eye_pro_R = tuple(ary(self.eye_stim_R) / ary(Stim.size))

        self.eye_stim = tuple((ary(self.eye_stim_L) + ary(self.eye_stim_R)) / 2.0)
        self.eye_pro  = tuple(ary(self.eye_stim) / ary(Stim.size))

        return self.eye_stim_L, self.eye_stim_R

    def record(self, Exp_ID: str, Part_ID: str, Stim_ID: str) -> pd.DataFrame:
        """
        Append one time-stamped sample to the recording table.

        Records:
            - left stimulus coords + proportional coords
            - right stimulus coords + proportional coords
            - fused stimulus coords + proportional coords

        Parameters
        ----------
        Exp_ID, Part_ID, Stim_ID :
            Identifiers stored alongside gaze samples.

        Returns
        -------
        pd.DataFrame
            One-row DataFrame that was appended to self.data.

        Side effects
        ------------
        - self.data grows by one row
        """
        new_data = pd.DataFrame(
            {
                "Exp": Exp_ID,
                "Part": Part_ID,
                "Stim": Stim_ID,
                "time": time(),
                "xL": self.eye_stim_L[0],
                "yL": self.eye_stim_L[1],
                "xL_pro": self.eye_pro_L[0],
                "yL_pro": self.eye_pro_L[1],
                "xR": self.eye_stim_R[0],
                "yR": self.eye_stim_R[1],
                "xR_pro": self.eye_pro_R[0],
                "yR_pro": self.eye_pro_R[1],
                # optional fused
                "xF": self.eye_stim[0],
                "yF": self.eye_stim[1],
                "xF_pro": self.eye_pro[0],
                "yF_pro": self.eye_pro[1],
                "blink": int(self._blink_holdoff > 0),
            },
            index=[0],
        )

        self.data = pd.concat([self.data, new_data], ignore_index=True)
        return new_data

    def reset_calib(self) -> None:
        """
        Clear calibration samples and delete trained models (if present).
        """
        self.calib_data = np.zeros(shape=(0, 10))
        for m in ("model_L", "model_R"):
            if hasattr(self, m):
                delattr(self, m)

    def reset_data(self) -> None:
        """
        Clear recorded gaze samples DataFrame.
        """
        self.data = pd.DataFrame(columns=YETI24.data_cols, dtype="float64")

    def reset(self) -> None:
        """
        Reset calibration and recorded data (does not disconnect cameras).
        """
        self.reset_calib()
        self.reset_data()
        self._roi_stable_L = False
        self._roi_stable_R = False
        self._blink_holdoff = 0
        self._last_stable_pos = None
        self.eye_stim_L = (0, 0)
        self.eye_stim_R = (0, 0)
        self.eye_stim   = (0, 0)
        self.eye_pro_L  = (0, 0)
        self.eye_pro_R  = (0, 0)
        self.eye_pro    = (0, 0)

    def draw_follow(self, surface: pg.Surface, add_raw=False, add_stim=False) -> None:
        """
        Draw the current gaze point(s) onto a pygame surface.

        Draws:
            - self.eye_pos (fused) in red
            - optionally self.eye_raw in green
            - optionally self.eye_stim in blue

        Parameters
        ----------
        surface :
            Pygame surface to draw on.
        add_raw :
            If True, also draw raw fused (pre-offset) estimate.
        add_stim :
            If True, also draw stimulus-local gaze estimate.
        """

        if not hasattr(self, "eye_pos"):
            return  # nothing to draw yet

        surf_w, surf_h = surface.get_size()

        circ_size = int(min(surf_w, surf_h) / 50)
        circ_stroke = int(min(surf_w, surf_h) / 200)

        # Clip to screen bounds
        x = int(np.clip(self.eye_pos[0], 0, surf_w - 1))
        y = int(np.clip(self.eye_pos[1], 0, surf_h - 1))

        circle(surface, (255, 0, 0), (x, y), circ_size, circ_stroke)

        if add_raw and hasattr(self, "eye_raw"):
            rx = int(np.clip(self.eye_raw[0], 0, surf_w - 1))
            ry = int(np.clip(self.eye_raw[1], 0, surf_h - 1))
            circle(surface, (0, 255, 0), (rx, ry), circ_size, circ_stroke)

        if add_stim and hasattr(self, "eye_stim"):
            sx = int(np.clip(self.eye_stim[0], 0, surf_w - 1))
            sy = int(np.clip(self.eye_stim[1], 0, surf_h - 1))
            circle(surface, (0, 0, 255), (sx, sy), circ_size, circ_stroke)


class Calib:

    color = (160, 160, 160)
    active_color = (255, 120, 0)
    radius = 20
    stroke = 10

    """
    Creates a square calib surface using relative positions
    ...

    Attributes
    ----------
    surface : pygame.Surface
        a Pygame surface object for drawing
    pro_positions : tuple[float]
        relative target positions used for creating a square calibration surface
    targets : numpy.array
        actual target positions (x,y)
    active : int
        index of the active targets

    Methods
    -------
    active_pos()
        returns the coordinates of the active targets
    reset()
        resets the active position to 0
    n()
        returns the number of targets to
    remaining()
        returns the number of remaining targets
    next()
        advances active position by 1
    draw()
        draws the calibration surface

    """

    def __init__(self, surface: pg.Surface, pro_positions=(0.125, 0.5, 0.875)) -> None:
        self.surface = surface
        self.surface_size = ary(self.surface.get_size())
        self.pro_positions = ary(pro_positions)
        x_pos = self.pro_positions * self.surface_size[0]
        y_pos = self.pro_positions * self.surface_size[1]
        self.targets = ary(
            list(itertools.product(x_pos, y_pos))
        )
        self.active = 0


    def shuffle(self, reset=True):
        self.reset()
        np.random.shuffle(self.targets)

    def active_pos(self) -> int:
        return self.targets[self.active]

    def reset(self) -> None:
        self.active = 0

    def n(self) -> int:
        return len(self.targets[:, 0])

    def remaining(self) -> int:
        return self.n() - self.active - 1

    def next(self) -> tuple:
        if self.remaining():
            this_target = self.targets[self.active]
            self.active += 1
            return True, this_target
        else:
            return False, None

    def draw(self) -> None:
        index = 0
        for target in self.targets:
            pos = list(map(int, target))
            if index == self.active:
                color = self.active_color
            else:
                color = self.color
            index += 1
            circle(self.surface, color, pos, self.radius, self.stroke)
