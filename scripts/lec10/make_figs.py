"""Lecture 10 figures (spectral analysis)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from svgplot import Fig, Panel, BLUE, RED, GREEN, GRAY, ORANGE, PURPLE, BLACK

OUT = "docs/figures/lec10/"
os.makedirs(OUT, exist_ok=True)
AMBER = "#BA7517"
rng = np.random.default_rng(10)

# ---------- 01: the signal zoo and its spectra ----------
fig = Fig(860, 520)
t = np.linspace(0, 3, 600)
rows = [
    ("sinusoidal", np.sin(2*np.pi*2*t), [(2, 1.0)], None),
    ("complex periodic", np.sin(2*np.pi*1*t) + 0.6*np.sin(2*np.pi*2*t) + 0.8*np.sin(2*np.pi*4*t),
     [(1, 1.0), (2, 0.6), (4, 0.8)], None),
    ("almost periodic", np.sin(2*t*2*np.pi/ (2*np.pi) *2*np.pi) , None, None),
    ("transient", np.exp(-1.5*t)*np.cos(2*np.pi*3*t), None, "cont"),
]
# fix almost periodic properly: sin(2t)+sin(3t)+sin(sqrt(50) t) in rad/s
rows[2] = ("almost periodic", np.sin(2*t*2)+0.7*np.sin(3*t*2)+0.8*np.sin(np.sqrt(50)*t*2),
           [(2/np.pi, 1.0), (3/np.pi, 0.7), (np.sqrt(50)/np.pi, 0.8)], None)
y0 = 40
for i, (name, sig, lines, cont) in enumerate(rows):
    p = Panel(fig, 55, y0 + i*118, 420, 86, (0, 3), (-2.3, 2.3), None, xticks=None, yticks=None, grid=False)
    p.line(t, sig, BLUE, 1.8)
    fig.text(58, y0 + i*118 - 6, name, 12, "start", BLACK, weight="bold")
    q = Panel(fig, 560, y0 + i*118, 260, 86, (0, 4.2), (0, 1.25), None, xticks=None, yticks=None, grid=False)
    fig.parts.append(f'<path d="M{q.x} {q.Y(0)}H{q.x+q.w}" stroke="{GRAY}" stroke-width="0.8"/>')
    if lines:
        for f_, a_ in lines:
            q.vseg(f_, 0, a_, RED, 2.5, None)
            fig.parts.append(f'<circle cx="{q.X(f_):.1f}" cy="{q.Y(a_):.1f}" r="3" fill="{RED}"/>')
    if cont:
        ff = np.linspace(0, 4.2, 100)
        q.line(ff, 1.1/(1 + ((ff-3)/0.45)**2), RED, 2)
    if i == 0: fig.text(690, y0 - 14, "spectrum", 12, "middle", GRAY)
fig.text(430, 505, "Down the zoo, spectra fatten: one line → harmonic lines (integer multiples) → incommensurate lines (period never repeats) → a continuous band (transients).", 11.5, "middle", GRAY)
fig.text(265, 505 - 485, "", 11, "middle", GRAY)
print("01", fig.save(OUT + "01_signal_zoo.svg"))

# ---------- 02: autocorrelation digs a period out of noise ----------
fig = Fig(860, 360)
fs = 200.0
tt = np.arange(0, 6, 1/fs)
clean = np.sin(2*np.pi*1.5*tt)
noisy = clean + 1.2*rng.standard_normal(len(tt))
p = Panel(fig, 55, 45, 360, 250, (0, 6), (-4.6, 4.6), "signal + heavy noise (SNR ≈ 0.3)",
          xticks=[0, 2, 4, 6], yticks=[-4, 0, 4], xlabel="t (s)")
p.line(tt, noisy, GRAY, 0.9)
p.line(tt, clean, BLUE, 1.8)
R = np.correlate(noisy - noisy.mean(), noisy - noisy.mean(), "full")
R = R[len(R)//2:]; R /= R[0]
lag = np.arange(len(R))/fs
q = Panel(fig, 490, 45, 330, 250, (0, 3), (-0.75, 1.1), "its autocorrelation R(τ)",
          xticks=[0, 1, 2, 3], yticks=[-0.5, 0, 0.5, 1], xlabel="lag τ (s)")
q.hline(0, GRAY, 0.8)
q.line(lag[lag <= 3], R[lag <= 3], GREEN, 2.2)
per = 1/1.5
q.vseg(per, -0.75, 1.0, RED, 1.2, "4 3")
q.label(per, 1.0, "τ = 0.667 s = the period", RED, 11, dx=6, dy=-4, weight="bold")
fig.text(430, 352, "The eye can't see the 1.5 Hz sine in the gray mess — R(τ) can: noise only correlates with itself at τ = 0, the signal at every period.", 11.5, "middle", GRAY)
print("02", fig.save(OUT + "02_acf_noise.svg"))

# ---------- 03: leakage and windowing ----------
M = 256
k = np.arange(M)
hann = 0.5 - 0.5*np.cos(2*np.pi*k/M)
def spec(c, w=None):
    s = np.sin(2*np.pi*c*k/M)
    if w is not None: s = s*w
    return np.abs(np.fft.rfft(s))/(M/2)
figs = [("4.0 cycles in the record", spec(4.0), GREEN),
        ("4.5 cycles — leakage", spec(4.5), RED),
        ("4.5 cycles + Hann window", spec(4.5, hann), BLUE)]
fig = Fig(860, 330)
for i, (title, S, col) in enumerate(figs):
    p = Panel(fig, 55 + i*272, 45, 240, 220, (-0.5, 20), (-8, 0.3), title,
              xticks=[0, 4, 10, 20], yticks=[0, -2, -4, -6, -8], logy=True,
              xlabel="frequency bin" if i == 1 else None)
    vals = np.log10(np.maximum(S[:21], 1e-12))
    for b in range(21):
        p.bar(b - 0.28, max(vals[b], -8), 0.56, col)
print("   peak bins:", [f"{S.max():.3f}" for _, S, _ in figs])
fig.text(430, 312, "Log scale. Integer cycles: one clean line. Half a cycle extra: energy smears everywhere. Hann window: skirt ~100× lower.", 11.5, "middle", GRAY)
print("03", fig.save(OUT + "03_leakage.svg"))
