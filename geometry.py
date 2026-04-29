"""
geometry.py
-----------
Screen geometry utilities: converts visual degrees <-> pixels
given viewing distance and physical screen dimensions.

All values come from Config.csv so swapping to the lab monitor
is a one-line config change — no code edits needed.

Usage:
    from geometry import ScreenGeometry
    geo = ScreenGeometry(screen_w_px, screen_h_px, screen_w_cm, viewing_dist_cm)
    x_px = geo.deg_to_px(eccentricity_deg)          # pixels from screen center
    deg  = geo.px_to_deg(pixel_offset)               # degrees from screen center
    pos  = geo.ecc_to_screen_pos(ecc_deg, side)      # absolute screen (x, y) for face center
"""

import math


class ScreenGeometry:
    def __init__(self, screen_w_px: int, screen_h_px: int,
                 screen_w_cm: float, viewing_dist_cm: float):
        self.screen_w_px     = screen_w_px
        self.screen_h_px     = screen_h_px
        self.screen_w_cm     = screen_w_cm
        self.viewing_dist_cm = viewing_dist_cm

        self.px_per_cm = screen_w_px / screen_w_cm
        self.cx        = screen_w_px // 2   # screen center x
        self.cy        = screen_h_px // 2   # screen center y

    def deg_to_px(self, deg: float) -> float:
        """Convert visual angle (degrees from center) to pixels from center."""
        cm = math.tan(math.radians(abs(deg))) * self.viewing_dist_cm
        return cm * self.px_per_cm

    def px_to_deg(self, px: float) -> float:
        """Convert pixel offset from center to visual angle in degrees."""
        cm = abs(px) / self.px_per_cm
        return math.degrees(math.atan(cm / self.viewing_dist_cm))

    def ecc_to_screen_pos(self, ecc_deg: float, side: str) -> tuple:
        """
        Return absolute (x, y) screen position for a face stimulus center
        placed at eccentricity ecc_deg on the given side.

        Parameters
        ----------
        ecc_deg : float
            Eccentricity in degrees visual angle from screen center.
        side : str
            'left' or 'right'

        Returns
        -------
        (x, y) in screen pixels (suitable for pygame blit center)
        """
        offset_px = int(self.deg_to_px(ecc_deg))
        if side == 'left':
            x = self.cx - offset_px
        else:
            x = self.cx + offset_px
        y = self.cy
        return (x, y)

    def max_eccentricity_deg(self, face_half_width_px: int = 150) -> float:
        """
        Return the maximum eccentricity (degrees) that fits on screen,
        given that a face stimulus has a half-width of face_half_width_px.
        """
        usable_half_px = self.cx - face_half_width_px
        return self.px_to_deg(usable_half_px)

    def fixation_radius_px(self, radius_deg: float) -> int:
        """Convert a fixation tolerance radius from degrees to pixels."""
        return int(self.deg_to_px(radius_deg))

    def summary(self) -> str:
        lines = [
            f"Screen: {self.screen_w_px}x{self.screen_h_px}px, "
            f"{self.screen_w_cm}cm wide, d={self.viewing_dist_cm}cm",
            f"  px/cm: {self.px_per_cm:.1f}",
            f"  Center: ({self.cx}, {self.cy})",
            f"  Max eccentricity: {self.max_eccentricity_deg():.1f}°",
            "  Eccentricity map:",
        ]
        for deg in [1, 5, 10, 15, 20, 25, 30]:
            px = self.deg_to_px(deg)
            fits = "✓" if px < self.cx - 150 else "✗ off-screen"
            lines.append(f"    {deg:>3}° = {px:>5.0f}px from center  {fits}")
        return "\n".join(lines)
