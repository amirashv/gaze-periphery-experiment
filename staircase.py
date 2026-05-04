"""
staircase.py
------------
Bayesian adaptive staircase for 2AFC gaze-change detection.
Uses 2-up/1-down rule with step-size halving after reversals.

Two key outcomes per condition:
  1. Maximum detectable eccentricity (furthest bin not abandoned)
  2. Detection threshold magnitude at each active eccentricity

Staircase rule
--------------
- 2-up/1-down: magnitude decreases after 2 consecutive correct responses,
  increases after any single wrong response
- Converges on ~70.7% correct threshold
- Starting magnitude: 20° (near maximum, works downward)
- Step size: 3° halving after every 2 reversals, minimum 1°

Abandon logic
-------------
- Each bin evaluated independently — no cascade assumption
- Bayesian Beta-Binomial model after MIN_TRIALS_BEFORE_ABANDON (10) trials
- Prior: Beta(1,1) — flat, no assumption about participant ability
- Posterior: Beta(k+1, n-k+1) updated with observed responses
- Abandon if P(true detection rate > 0.5 | data) < ABANDON_THRESHOLD (0.80)
- At 10 trials: abandons if ≤5 correct (≤50% accuracy)

Stopping rule
-------------
A staircase is complete when EITHER:
  - All active (non-abandoned) bins have >= MIN_TRIALS_PER_BIN (15) trials, OR
  - n_trials hard limit (400) is reached
"""

import math
import random
import numpy as np
import pandas as pd
from scipy.stats import beta
from typing import Optional, Tuple, List

ECC_MIN = 1.0
ECC_MAX = 30.0
MAG_MIN = 1.0
MAG_MAX = 30.0


def psychometric(magnitude: float, threshold: float, slope: float = 2.0) -> float:
    if magnitude <= 0:
        return 0.5
    chance = 0.5
    d = max(0.0, magnitude / threshold) ** slope
    return chance + (1 - chance) * (1 - math.exp(-d))


class StaircaseBlock:
    """
    Adaptive staircase for one condition (control or quadbright).

    Parameters
    ----------
    condition            : 'control' or 'quadbright'
    available_magnitudes : list of float — gaze-change magnitudes with stimuli
    ecc_range            : (min, max) eccentricity in degrees
    n_trials             : hard upper limit on trials
    min_trials_per_bin   : trials needed per active bin for convergence
    seed                 : random seed
    """

    def __init__(self,
                 condition: str,
                 available_magnitudes: List[float],
                 ecc_range: Tuple[float, float] = (ECC_MIN, ECC_MAX),
                 n_trials: int = 400,
                 min_trials_per_bin: int = 15,
                 seed: Optional[int] = None):

        self.condition            = condition
        self.n_trials             = n_trials
        self.MIN_TRIALS_PER_BIN   = min_trials_per_bin
        self.MIN_TRIALS_BEFORE_ABANDON = 10   # same as convergence criterion
        self.ABANDON_THRESHOLD    = 0.80      # P(above chance) must exceed this to keep bin
        self.rng                  = random.Random(seed)
        self.np_rng               = np.random.default_rng(seed)

        self.available_magnitudes = sorted(available_magnitudes)
        self.ecc_min, self.ecc_max = ecc_range

        # Eccentricity grid — integer degrees
        self.ecc_grid      = np.arange(ecc_range[0], ecc_range[1] + 1, 1.0)
        n_bins             = len(self.ecc_grid)

        # Per-bin state
        self.threshold_est = np.full(n_bins, 20.0)   # start at mid-range magnitude
        self.threshold_n   = np.zeros(n_bins)         # trials per bin
        self._correct_counts = np.zeros(n_bins)
        self._total_counts   = np.zeros(n_bins)
        self._abandoned      = np.zeros(n_bins, dtype=bool)

        # 1-up/1-down staircase state
        self._last_correct = {}    # bin_idx -> last bool response
        self._reversals    = {}    # bin_idx -> reversal count
        self._step_size    = 3.0  # starting step size in degrees
        self._consecutive_correct = {}    # bin_idx -> consecutive correct count


        # Trial history
        self.history: List[dict] = []
        self.trial_count = 0

    # ------------------------------------------------------------------
    # Proposal
    # ------------------------------------------------------------------

    def propose_trial(self) -> Tuple[float, float]:
        """
        Propose next (eccentricity, magnitude).
        Only proposes from active (non-abandoned) bins.
        Prefers least-tested bins for exploration.
        """
        active = ~self._abandoned
        if not np.any(active):
            # All abandoned — fall back to bin 0 (should not happen in practice)
            active = np.ones(len(self.ecc_grid), dtype=bool)

        # Among active bins, pick the one with fewest trials
        active_n = np.where(active, self.threshold_n, np.inf)
        idx = int(np.argmin(active_n))

        ecc        = float(self.ecc_grid[idx])
        target_mag = float(self.threshold_est[idx])
        mag        = self._nearest_magnitude(target_mag)

        return ecc, mag

    def _nearest_magnitude(self, target: float) -> float:
        diffs = [abs(m - target) for m in self.available_magnitudes]
        return self.available_magnitudes[int(np.argmin(diffs))]

    def _ecc_bin(self, ecc: float) -> int:
        diffs = np.abs(self.ecc_grid - ecc)
        return int(np.argmin(diffs))

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def register_response(self,
                          eccentricity: float,
                          magnitude: float,
                          correct: bool,
                          srt_ms: Optional[float] = None,
                          valid: bool = True,
                          pre_change_ms: Optional[float] = None,
                          keyboard_rt_ms: Optional[float] = None) -> None:
        """Record trial outcome and update threshold + abandon logic."""
        self.trial_count += 1

        row = {
            "trial":        self.trial_count,
            "condition":    self.condition,
            "eccentricity": eccentricity,
            "magnitude":    magnitude,
            "correct":      int(correct),
            "srt_ms":       srt_ms,
            "keyboard_rt_ms": keyboard_rt_ms,
            "pre_change_ms":  pre_change_ms,
            "valid":        int(valid),
        }
        self.history.append(row)

        if not valid:
            return

        idx = self._ecc_bin(eccentricity)
        self.threshold_n[idx]     += 1
        self._total_counts[idx]   += 1
        self._correct_counts[idx] += int(correct)

        # --- 2-up/1-down threshold update ---
        step = self._get_step(idx)

        # Track consecutive correct responses
        if correct:
            self._consecutive_correct[idx] = self._consecutive_correct.get(idx, 0) + 1
        else:
            self._consecutive_correct[idx] = 0

        if not correct:
            # 1 wrong → increase magnitude (easier)
            self.threshold_est[idx] = min(
                float(self.available_magnitudes[-1]),
                self.threshold_est[idx] + step
            )
        elif self._consecutive_correct[idx] >= 2:
            # 2 consecutive correct → decrease magnitude (harder)
            self._consecutive_correct[idx] = 0
            self.threshold_est[idx] = max(
                float(self.available_magnitudes[0]),
                self.threshold_est[idx] - step
            )

        # Track reversals for step-size halving
        prev = self._last_correct.get(idx)
        if prev is not None and prev != correct:
            self._reversals[idx] = self._reversals.get(idx, 0) + 1
        self._last_correct[idx] = correct

        # --- Independent bin abandon check (Bayesian Beta-Binomial) ---
        # No cascade — each bin evaluated independently
        # Prior: Beta(1,1), Posterior: Beta(k+1, n-k+1)
        # Abandon if P(true detection rate > 0.5 | data) < ABANDON_THRESHOLD
        if self._total_counts[idx] >= self.MIN_TRIALS_BEFORE_ABANDON:
            n                 = int(self._total_counts[idx])
            k                 = int(self._correct_counts[idx])
            prob_above_chance = 1 - beta.cdf(0.5, k + 1, n - k + 1)
            if prob_above_chance < self.ABANDON_THRESHOLD and not self._abandoned[idx]:
                self._abandoned[idx] = True
                import logging
                logging.info(
                    f"[{self.condition}] Abandoned ecc={self.ecc_grid[idx]:.1f}° "
                    f"({k}/{n} correct, P(above chance)={prob_above_chance:.3f} < {self.ABANDON_THRESHOLD})"
                )

    def _get_step(self, idx: int) -> float:
        """Step size halves after every 2 reversals, minimum 1°. Part of 2-up/1-down rule."""
        rev = self._reversals.get(idx, 0)
        halvings = rev // 2
        return max(1.0, self._step_size / (2 ** halvings))

    # ------------------------------------------------------------------
    # Stopping rule
    # ------------------------------------------------------------------

    def is_converged(self) -> bool:
        """
        True if every active (non-abandoned) bin has >= MIN_TRIALS_PER_BIN trials.
        This is the natural stopping condition.
        """
        active = ~self._abandoned
        if not np.any(active):
            return True
        return bool(np.all(self.threshold_n[active] >= self.MIN_TRIALS_PER_BIN))

    def is_complete(self) -> bool:
        """True if converged OR hard trial limit reached."""
        return self.is_converged() or self.trial_count >= self.n_trials

    def remaining(self) -> int:
        return max(0, self.n_trials - self.trial_count)

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def max_detectable_eccentricity(self) -> Optional[float]:
        """
        The highest eccentricity bin that is NOT abandoned.
        This is the participant's maximum detectable eccentricity.
        Returns None if all bins are abandoned.
        """
        active_indices = np.where(~self._abandoned)[0]
        if len(active_indices) == 0:
            return None
        return float(self.ecc_grid[active_indices[-1]])

    def threshold_summary(self) -> pd.DataFrame:
        """
        One row per eccentricity bin with:
        - threshold_est : estimated minimum detectable magnitude
        - n_trials      : trials at this bin
        - accuracy      : proportion correct
        - abandoned     : 1 if abandoned (at chance), 0 if active
        - detectable_range : 'threshold_est° to 30°' for active bins, NaN for abandoned
        """
        n = len(self.ecc_grid)
        accuracy = np.where(
            self._total_counts > 0,
            self._correct_counts / np.maximum(self._total_counts, 1),
            np.nan
        )
        # Compute P(above chance) per bin
        prob_above_chance_list = []
        for i in range(n):
            ni = int(self._total_counts[i])
            ki = int(self._correct_counts[i])
            if ni >= self.MIN_TRIALS_BEFORE_ABANDON:
                prob = round(float(1 - beta.cdf(0.5, ki + 1, ni - ki + 1)), 4)
                prob_above_chance_list.append(prob)
            else:
                prob_above_chance_list.append(None)

        detectable_range = []
        for i in range(n):
            if self._abandoned[i] or self._total_counts[i] == 0:
                detectable_range.append(None)
            else:
                lo = round(float(self.threshold_est[i]), 1)
                detectable_range.append(f"{lo}° to 30°")

        return pd.DataFrame({
            "eccentricity":    self.ecc_grid,
            "threshold_est":   self.threshold_est,
            "n_trials":        self.threshold_n,
            "accuracy":        accuracy,
            "P_above_chance":  prob_above_chance_list,
            "abandoned":       self._abandoned.astype(int),
            "detectable_range":detectable_range,
        })

    def convergence_status(self) -> str:
        """Human-readable summary of staircase progress."""
        active   = ~self._abandoned
        n_active = int(np.sum(active))
        n_ready  = int(np.sum(self.threshold_n[active] >= self.MIN_TRIALS_PER_BIN))
        n_aband  = int(np.sum(self._abandoned))
        max_ecc  = self.max_detectable_eccentricity()
        return (f"[{self.condition}] {n_ready}/{n_active} bins converged, "
                f"{n_aband} abandoned, "
                f"max ecc={max_ecc}°, "
                f"trials={self.trial_count}, "
                f"complete={self.is_complete()}")

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)
