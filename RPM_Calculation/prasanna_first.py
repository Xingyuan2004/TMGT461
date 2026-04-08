import matplotlib
matplotlib.use("TkAgg")  # VS Code GUI backend
 
import numpy as np
import soundfile as sf
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
from matplotlib.gridspec import GridSpec
from collections import deque
 
# ----------------------------
# User config
# ----------------------------
WAV_PATH = "MJC04 - 1800 rpm_august 13 2025 G3520_010_1_1_2.wav"   # <-- change this
# 950L_TransmissionNoise_2Fto5FManualShiftSweep.wav
BLOCK_MS = 30                 # update interval (ms) and audio block duration
WINDOW_SEC = 1.0              # rolling window length for waveform/FFT (s)
FFT_FMAX = 8000               # max frequency shown (Hz)
EPS = 1e-12                   # for log stability
F0_MIN_HZ = 20.0              # ignore below this when picking the "fundamental"
 
# Realtime spectrogram (running columns)
RT_SPEC_NFFT = 1024
RT_SPEC_COLS = 220
RT_SPEC_DB_FLOOR = -80
RT_SPEC_DB_CEIL = 0
 
# Full-file spectrogram settings
FULL_SPEC_NFFT = 2048
FULL_SPEC_OVERLAP = 1536  # 75% overlap
 
# Safeguard for full-file plots (downsample if too long to keep UI responsive)
MAX_FULL_PLOT_SAMPLES = 1_500_000  # adjust if you want
 
# ----------------------------
# Load WAV
# ----------------------------
audio, fs = sf.read(WAV_PATH, dtype="float32", always_2d=True)
audio = audio.mean(axis=1)  # mono
n_total = len(audio)
print(fs)
block_size = max(256, int(fs * BLOCK_MS / 1000.0))
win_size = int(fs * WINDOW_SEC)
 
ring = deque([0.0] * win_size, maxlen=win_size)
idx = 0  # playhead
 
# ----------------------------
# Prepare "full-file" data (possibly downsampled for static plots)
# ----------------------------
audio_full = audio
fs_full = fs
 
ds_factor = 1
if len(audio_full) > MAX_FULL_PLOT_SAMPLES:
    ds_factor = int(np.ceil(len(audio_full) / MAX_FULL_PLOT_SAMPLES))
    audio_full = audio_full[::ds_factor]
    fs_full = fs / ds_factor
 
# ----------------------------
# Figure + GridSpec layout: 3 rows x 2 cols
# Left col = real-time (3 rows)
# Right col = full-file FFT (top), full-file spectrogram (rows 2-3)
# GridSpec supports spanning rows/cols. [1](https://matplotlib.org/stable/gallery/subplots_axes_and_figures/gridspec_multicolumn.html)
# ----------------------------
plt.style.use("fast")
fig = plt.figure(figsize=(16, 9))
gs = GridSpec(3, 2, figure=fig, width_ratios=[1.15, 1.0], wspace=0.22, hspace=0.45)
fig.subplots_adjust(bottom=0.18)
 
ax_time_rt = fig.add_subplot(gs[0, 0])
ax_fft_rt  = fig.add_subplot(gs[1, 0])
ax_spec_rt = fig.add_subplot(gs[2, 0])
 
ax_fft_full  = fig.add_subplot(gs[0, 1])
ax_spec_full = fig.add_subplot(gs[1:, 1])  # span rows 1 and 2 on right
 
# ----------------------------
# LEFT COLUMN: Real-time Waveform
# ----------------------------
t = np.arange(win_size) / fs
(line_time,) = ax_time_rt.plot(t, np.zeros(win_size), lw=1)
ax_time_rt.set_title("Realtime: Waveform (rolling window)")
ax_time_rt.set_xlabel("Time (s)")
ax_time_rt.set_ylabel("Amplitude")
ax_time_rt.set_xlim(t[0], t[-1])
ax_time_rt.set_ylim(-1.05, 1.05)
ax_time_rt.grid(True, alpha=0.3)
 
# ----------------------------
# LEFT COLUMN: Real-time FFT
# ----------------------------
freqs_rt = np.fft.rfftfreq(win_size, d=1/fs)
mask_fft_rt = freqs_rt <= FFT_FMAX
 
(line_fft,) = ax_fft_rt.plot(freqs_rt[mask_fft_rt], np.zeros(mask_fft_rt.sum()),
                            lw=1, color="crimson")
ax_fft_rt.set_title("Realtime: FFT (rolling window)")
ax_fft_rt.set_xlabel("Frequency (Hz)")
ax_fft_rt.set_ylabel("Magnitude (dB, normalized)")
ax_fft_rt.set_xlim(0, FFT_FMAX)
ax_fft_rt.set_ylim(-120, 0)
ax_fft_rt.grid(True, alpha=0.3)
 
# Fundamental peak marker
(peak_marker,) = ax_fft_rt.plot([], [], marker="o", color="gold", markersize=8, zorder=5)
peak_vline = ax_fft_rt.axvline(0, color="gold", lw=1.5, alpha=0.8)
peak_text = ax_fft_rt.text(
    0.02, 0.92, "", transform=ax_fft_rt.transAxes,
    fontsize=11, color="gold",
    bbox=dict(facecolor="black", alpha=0.35, edgecolor="none")
)
 
def find_fundamental(freqs_visible, mag_db_visible):
    valid = freqs_visible >= F0_MIN_HZ
    if not np.any(valid):
        return None, None
    sub_mag = mag_db_visible[valid]
    sub_freq = freqs_visible[valid]
    k = np.argmax(sub_mag)
    return float(sub_freq[k]), float(sub_mag[k])
 
# ----------------------------
# LEFT COLUMN: Real-time Spectrogram (running)
# Uses imshow + set_data for efficient updates. [2](https://stackoverflow.com/questions/17835302/how-to-update-matplotlibs-imshow-window-interactively)[3](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html)
# ----------------------------
rt_spec_freqs = np.fft.rfftfreq(RT_SPEC_NFFT, d=1/fs)
mask_spec_rt = rt_spec_freqs <= FFT_FMAX
rt_spec_fmax = rt_spec_freqs[mask_spec_rt][-1] if np.any(mask_spec_rt) else rt_spec_freqs[-1]
 
rt_spec_db = np.full((rt_spec_freqs.size, RT_SPEC_COLS), RT_SPEC_DB_FLOOR, dtype=np.float32)
 
hop_sec = block_size / fs
rt_time_span = RT_SPEC_COLS * hop_sec
 
rt_img = ax_spec_rt.imshow(
    rt_spec_db[mask_spec_rt, :],
    origin="lower",
    aspect="auto",
    extent=[-rt_time_span, 0.0, 0.0, rt_spec_fmax],
    cmap="magma",
    vmin=RT_SPEC_DB_FLOOR,
    vmax=RT_SPEC_DB_CEIL,
)
ax_spec_rt.set_title("Realtime: Spectrogram (newest at right)")
ax_spec_rt.set_xlabel("Time (s) [relative]")
ax_spec_rt.set_ylabel("Frequency (Hz)")
cbar_rt = fig.colorbar(rt_img, ax=ax_spec_rt, pad=0.01)
cbar_rt.set_label("dB (normalized)")
 
# ----------------------------
# RIGHT COLUMN: Full-file FFT (static)
# ----------------------------
# Window entire signal to reduce leakage (single segment FFT).
w_full = np.hanning(len(audio_full))
Y_full = np.fft.rfft(audio_full * w_full)
mag_full = np.abs(Y_full)
freqs_full = np.fft.rfftfreq(len(audio_full), d=1/fs_full)
 
mask_fft_full = freqs_full <= FFT_FMAX
mag_db_full = 20 * np.log10(mag_full + EPS)
mag_db_full -= np.max(mag_db_full)
 
ax_fft_full.plot(freqs_full[mask_fft_full], mag_db_full[mask_fft_full], color="teal", lw=1)
ax_fft_full.set_title("Full file: FFT")
ax_fft_full.set_xlabel("Frequency (Hz)")
ax_fft_full.set_ylabel("Magnitude (dB, normalized)")
ax_fft_full.set_xlim(0, FFT_FMAX)
ax_fft_full.set_ylim(-120, 0)
ax_fft_full.grid(True, alpha=0.3)
 
# ----------------------------
# RIGHT COLUMN: Full-file Spectrogram (static)
# Axes.specgram computes and plots a spectrogram and returns (Pxx, freqs, bins, im). [4](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.specgram.html)[5](https://matplotlib.org/stable/gallery/images_contours_and_fields/specgram_demo.html)
# ----------------------------
Pxx, f_spec, t_spec, im_spec = ax_spec_full.specgram(
    audio_full,
    NFFT=FULL_SPEC_NFFT,
    Fs=fs_full,
    noverlap=FULL_SPEC_OVERLAP,
    cmap="magma",
    scale="dB",
    mode="magnitude",
)
ax_spec_full.set_title("Full file: Spectrogram")
ax_spec_full.set_xlabel("Time (s)")
ax_spec_full.set_ylabel("Frequency (Hz)")
ax_spec_full.set_ylim(0, FFT_FMAX)
cbar_full = fig.colorbar(im_spec, ax=ax_spec_full, pad=0.01)
cbar_full.set_label("Magnitude (dB)")
 
# ----------------------------
# Audio output stream (realtime playback)
# ----------------------------
stream = sd.OutputStream(samplerate=fs, channels=1, dtype="float32", blocksize=block_size)
stream.start()
 
state = {"stopped": False, "paused": False}
 
def stop_everything():
    state["stopped"] = True
    try:
        anim.event_source.stop()
    except Exception:
        pass
    try:
        stream.stop()
        stream.close()
    except Exception:
        pass
 
def update(_frame):
    global idx, rt_spec_db
 
    if state["stopped"]:
        return
 
    if idx >= n_total:
        stop_everything()
        return
 
    # Get next audio block
    block = audio[idx: idx + block_size]
    idx += len(block)
    if len(block) < block_size:
        block = np.pad(block, (0, block_size - len(block)))
 
    # Play audio (pausing animation pauses audio because write happens here)
    stream.write(block.reshape(-1, 1))
 
    # Update rolling buffer
    ring.extend(block.tolist())
    y = np.asarray(ring, dtype=np.float32)
 
    # --- waveform update ---
    line_time.set_ydata(y)
 
    # --- FFT update ---
    w = np.hanning(len(y))
    Y = np.fft.rfft(y * w)
    mag = np.abs(Y)
    mag_db = 20 * np.log10(mag + EPS)
    mag_db -= np.max(mag_db)
 
    mag_db_vis = mag_db[mask_fft_rt]
    freqs_vis = freqs_rt[mask_fft_rt]
    line_fft.set_ydata(mag_db_vis)
 
    f0, m0 = find_fundamental(freqs_vis, mag_db_vis)
    if f0 is None:
        peak_marker.set_data([], [])
        peak_vline.set_xdata([0, 0])
        peak_text.set_text("")
    else:
        peak_marker.set_data([f0], [m0])
        peak_vline.set_xdata([f0, f0])
        peak_text.set_text(f"Fundamental peak: f0 = {f0:.1f} Hz")
 
    # --- realtime spectrogram update (one new column) ---
    if len(y) >= RT_SPEC_NFFT:
        seg = y[-RT_SPEC_NFFT:]
    else:
        seg = np.pad(y, (RT_SPEC_NFFT - len(y), 0))
 
    seg = seg * np.hanning(RT_SPEC_NFFT)
    S = np.fft.rfft(seg)
    Smag = np.abs(S)
 
    Sdb = 20 * np.log10(Smag + EPS)
    Sdb -= np.max(Sdb)
 
    rt_spec_db = np.roll(rt_spec_db, -1, axis=1)
    rt_spec_db[:, -1] = np.clip(Sdb, RT_SPEC_DB_FLOOR, RT_SPEC_DB_CEIL)
 
    rt_img.set_data(rt_spec_db[mask_spec_rt, :])
 
    # blit=False so return not required
 
# ----------------------------
# Animation
# ----------------------------
anim = FuncAnimation(fig, update, interval=BLOCK_MS, blit=False, cache_frame_data=False)
 
# ----------------------------
# Buttons (keep references!)
# ----------------------------
ax_pause = fig.add_axes([0.72, 0.04, 0.10, 0.06])
btn_pause = Button(ax_pause, "Pause", hovercolor="0.95")
 
ax_stop = fig.add_axes([0.84, 0.04, 0.10, 0.06])
btn_stop = Button(ax_stop, "Stop", hovercolor="0.95")
 
def on_pause(_event):
    if state["stopped"]:
        return
    if not state["paused"]:
        anim.pause()
        state["paused"] = True
        btn_pause.label.set_text("Resume")
    else:
        anim.resume()
        state["paused"] = False
        btn_pause.label.set_text("Pause")
    fig.canvas.draw_idle()
 
def on_stop(_event):
    stop_everything()
    plt.close(fig)
 
btn_pause.on_clicked(on_pause)
btn_stop.on_clicked(on_stop)
 
plt.show()
