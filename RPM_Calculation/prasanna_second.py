import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
 
def stft_mag_db(x, fs, nperseg=4096, hop=1024, window="hann", eps=1e-12):
    if window == "hann":
        w = np.hanning(nperseg)
    else:
        w = np.ones(nperseg)
 
    # frame the signal
    n_frames = 1 + (len(x) - nperseg) // hop if len(x) >= nperseg else 0
    if n_frames <= 0:
        raise ValueError("Signal too short for chosen nperseg/hop.")
 
    frames = np.lib.stride_tricks.as_strided(
        x,
        shape=(n_frames, nperseg),
        strides=(x.strides[0]*hop, x.strides[0]),
        writeable=False
    ).copy()
 
    frames *= w[None, :]
    X = np.fft.rfft(frames, axis=1)
    mag = np.abs(X)
 
    mag_db = 20*np.log10(mag + eps)
    mag_db -= mag_db.max()  # normalize overall to 0 dB
    freqs = np.fft.rfftfreq(nperseg, d=1/fs)
    times = (np.arange(n_frames)*hop + nperseg/2) / fs
    return freqs, times, mag_db.T  # (freq_bins, time_bins)
 
def local_sum_energy(S_db, freqs, f_center, bw_hz=3.0):
    """Sum linear energy in a narrow band around f_center."""
    if f_center <= 0:
        return 0.0
    idx = np.where((freqs >= f_center - bw_hz) & (freqs <= f_center + bw_hz))[0]
    if idx.size == 0:
        return 0.0
    # convert dB back to linear amplitude proxy
    lin = 10**(S_db[idx, :]/20.0)
    return float(np.mean(lin))  # average over time
 
def score_rpm(S_db, freqs, rpm, orders=(1, 2.5, 5, 10), harmonics=3, bw_hz=3.0):
    f1 = rpm / 60.0
    score = 0.0
    for o in orders:
        base = o * f1
        # score base + a few harmonics (helps reject resonances)
        for h in range(1, harmonics+1):
            score += local_sum_energy(S_db, freqs, h*base, bw_hz=bw_hz) / h
    return score
 
def estimate_rpm_from_exhaust_wav(
    wav_path,
    fs_expected=50000,
    rpm_min=300,
    rpm_max=2500,
    nperseg=4096,
    hop=1024,
    f_search=(55, 2000),
    fmax_plot=1200,
    orders= (1, 2.5, 5, 10),
    bw_hz=3.0
):
    x, fs = sf.read(wav_path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
 
    if fs != fs_expected:
        print(f"[warn] WAV fs={fs}, expected {fs_expected}. Continuing with fs={fs}.")
 
    freqs, times, S_db = stft_mag_db(x, fs, nperseg=nperseg, hop=hop)
 
    # Limit freq range for candidate peak picking
    f_lo, f_hi = f_search
    band = np.where((freqs >= f_lo) & (freqs <= f_hi))[0]
    S_band = S_db[band, :]
 
    # Average spectrum over time and pick top peaks
    avg_spec = np.mean(10**(S_band/20.0), axis=1)  # linear average
    # take top K bins as candidates
    K = 25
    cand_idx = np.argpartition(avg_spec, -K)[-K:]
    cand_freqs = freqs[band][cand_idx]
    cand_freqs = np.unique(np.round(cand_freqs, 2))
 
    # Convert candidate freqs to candidate RPMs by assuming they could be 10×, 5×, 2.5×, 1×
    cand_rpms = []
    firing_order = min(o for o in orders if o > 1.0)
    for f in cand_freqs:
        for o in orders:
            rpm = 60.0 * f / o
            if (firing_order *rpm/60.0) <f_lo:
                continue
            if rpm_min <= rpm <= rpm_max:
                cand_rpms.append(rpm)
    cand_rpms = np.unique(np.round(cand_rpms, 1))
    print(cand_rpms)
 
    # Score candidates
    scores = []
    for rpm in cand_rpms:
        s = score_rpm(S_db, freqs, rpm, orders=orders, harmonics=3, bw_hz=bw_hz)
        scores.append(s)
 
    if len(scores) == 0:
        raise RuntimeError("No RPM candidates found. Widen f_search or rpm_min/rpm_max.")
 
    best_i = int(np.argmax(scores))
    print(best_i)
    best_rpm = float(cand_rpms[best_i])
 
    # Diagnostics: which order is most supported?
    f1 = best_rpm/60.0
    order_freqs = {o: o*f1 for o in orders}
 
    # Plot spectrogram + order overlays
    fig, ax = plt.subplots(figsize=(11, 5))
    fmask = freqs <= fmax_plot
    im = ax.imshow(
        S_db[fmask, :],
        origin="lower",
        aspect="auto",
        extent=[times[0], times[-1], freqs[fmask][0], freqs[fmask][-1]],
        cmap="magma",
        vmin=-80, vmax=0
    )
    plt.colorbar(im, ax=ax, label="dB (normalized)")
    ax.set_title(f"Spectrogram with order lines — Estimated RPM ≈ {best_rpm:.1f}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
 
    # Horizontal order lines (works best if RPM ~ steady)
    for o, fo in order_freqs.items():
        ax.hlines(fo, times[0], times[-1], colors="cyan", linestyles="--", linewidth=1)
        ax.text(times[0], fo, f"{o}×", color="cyan", va="bottom")
 
    plt.tight_layout()
    plt.show()
 
    # Return result + a simple confidence ratio
    sorted_scores = np.sort(scores)[::-1]
    confidence = float(sorted_scores[0] / (sorted_scores[1] + 1e-9)) if len(sorted_scores) > 1 else 999.0
    return best_rpm, confidence, order_freqs
 
# Example:
# rpm, conf, order_freqs = estimate_rpm_from_exhaust_wav("your_exhaust.wav1", fs_expected=50000)
# print(rpm, conf, order_freqs)
rpm, conf, order_freqs = estimate_rpm_from_exhaust_wav("MJC04 - 1800 rpm_august 13 2025 G3520_008_1_1_2.wav", fs_expected=1000000)
print(rpm, conf, order_freqs)
