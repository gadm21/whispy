"""Guaranteed-rate CSI resampler — magnitude-only port of the upstream
WiFiSensing ``CSI_Loader._resample_equal_intervals`` used verbatim by the
ICC occupancy pipeline (``run_csi_100hz.load_resampler``).

Semantics preserved exactly:
  * fixed ``guaranteed_sr`` Hz grid anchored at the first timestamp
  * samples in the same 1/sr bin are averaged
  * empty bins are linearly interpolated between populated bins
    (edge slots clamp to the nearest populated bin)
"""
from __future__ import annotations

import numpy as np

GUARANTEED_SR = 100.0


def resample(amplitude: np.ndarray, times_seconds: np.ndarray,
             sr: float = GUARANTEED_SR):
    """Resample amplitude to an equal ``sr`` Hz grid.

    Parameters
    ----------
    amplitude : (n, k) float array — CSI magnitudes.
    times_seconds : (n,) float — epoch or relative timestamps (seconds).

    Returns
    -------
    (values, times, samples_per_bin, stats)
        values : (n_out, k) float64 resampled magnitudes
        times  : (n_out,) float64 grid timestamps (same epoch as input)
        samples_per_bin : (n_out,) int64 raw samples per output bin
        stats : dict of upstream resampling statistics
    """
    amplitude = np.asarray(amplitude, dtype=np.float64)
    times_seconds = np.asarray(times_seconds, dtype=np.float64)
    n_orig = len(times_seconds)
    stats = {
        'original_samples': n_orig, 'resampled_samples': 0,
        'empty_slots': 0, 'empty_slots_pct': 0.0,
        'overlapping_samples': 0, 'actual_sampling_rate': 0.0,
        'target_sampling_rate': sr, 'duration_sec': 0.0,
    }
    if n_orig == 0:
        return (amplitude[:0], times_seconds[:0],
                np.zeros(0, np.int64), stats)

    start, end = times_seconds[0], times_seconds[-1]
    duration = end - start
    n_out = int(np.ceil(duration * sr))
    if n_out < 2:
        # upstream trivial path: too-short span
        stats.update(original_samples=n_orig, resampled_samples=n_orig,
                     actual_sampling_rate=n_orig / max(duration, 1e-8),
                     duration_sec=duration)
        spb = np.ones(n_orig, dtype=np.int64)
        return amplitude, times_seconds, spb, stats

    target_t = start + np.arange(n_out) / sr
    dt = 1.0 / sr
    n_sc = amplitude.shape[1]

    # vectorized bin assignment (verbatim upstream semantics)
    bin_edges = target_t - dt / 2
    bin_idx = np.clip(np.searchsorted(bin_edges, times_seconds,
                                      side='right') - 1, 0, n_out - 1)
    samples_per_bin = np.bincount(bin_idx, minlength=n_out).astype(np.int64)

    acc = np.zeros((n_out, n_sc), dtype=np.float64)
    np.add.at(acc, bin_idx, amplitude)
    populated = samples_per_bin > 0
    acc[populated] /= samples_per_bin[populated, None]

    empty_idx = np.flatnonzero(~populated)
    if len(empty_idx):
        valid_idx = np.flatnonzero(populated)
        if len(valid_idx) >= 2:
            for sc in range(n_sc):
                acc[empty_idx, sc] = np.interp(
                    target_t[empty_idx], target_t[valid_idx],
                    acc[valid_idx, sc])
        elif len(valid_idx) == 1:
            acc[empty_idx] = acc[valid_idx[0]]

    stats.update(resampled_samples=n_out,
                 empty_slots=int(len(empty_idx)),
                 empty_slots_pct=100.0 * len(empty_idx) / n_out,
                 overlapping_samples=int(
                     (samples_per_bin[samples_per_bin > 1] - 1).sum()),
                 actual_sampling_rate=n_orig / max(duration, 1e-8),
                 duration_sec=duration)
    return acc, target_t, samples_per_bin, stats


def verify() -> None:
    """Upstream self-check (identical to run_csi_100hz.verify_resampler)."""
    t = np.array([0., .001, .02, .03])
    a = np.array([[1., 2.], [3., 4.], [8., 10.], [10., 12.]])
    values, times, counts, stats = resample(a, t)
    assert counts.tolist() == [2, 0, 2]
    np.testing.assert_allclose(values, [[2, 3], [5.5, 7], [9, 11]],
                               rtol=1e-6)
    np.testing.assert_allclose(times, [0, .01, .02], atol=1e-6)
    assert stats['empty_slots'] == 1
    assert np.isclose(np.array([[0, 0], [3, 4], [6, 8]]).var(axis=0).sum(),
                      50 / 3)
