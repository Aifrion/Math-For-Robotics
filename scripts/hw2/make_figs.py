"""HW2 companion figures (compact SVG via the house plotter)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from scipy.interpolate import CubicSpline
from svgplot import Fig, Panel, BLUE, RED, GREEN, GRAY, ORANGE, PURPLE, BLACK

OUT = "docs/figures/hw2/"
W = np.loadtxt("hw2/waypoints.csv", delimiter=",")
n = len(W); t = np.arange(n) / 30.0

def samples(step):
    idx = np.arange(0, n, step)
    if idx[-1] != n - 1: idx = np.append(idx, n - 1)
    return idx

def quad_interp(ts, xs, tq):
    out = np.empty_like(tq); m = len(ts)
    for k in range(0, m - 2, 2):
        t0, t1, t2 = ts[k], ts[k+1], ts[k+2]
        sel = (tq >= t0) & (tq <= t2) if k + 4 <= m else (tq >= t0)
        tt = tq[sel]
        L0 = (tt-t1)*(tt-t2)/((t0-t1)*(t0-t2)); L1 = (tt-t0)*(tt-t2)/((t1-t0)*(t1-t2))
        L2 = (tt-t0)*(tt-t1)/((t2-t0)*(t2-t1))
        out[sel] = xs[k]*L0 + xs[k+1]*L1 + xs[k+2]*L2
    if (m - 1) % 2 == 1:
        sel = tq >= ts[m-2]; out[sel] = np.interp(tq[sel], ts[m-2:], xs[m-2:])
    return out

def recon(idx, method, tq):
    ts = t[idx]
    if method == "linear":
        return np.column_stack([np.interp(tq, ts, W[idx, 0]), np.interp(tq, ts, W[idx, 1])])
    if method == "quadratic":
        return np.column_stack([quad_interp(ts, W[idx, 0], tq), quad_interp(ts, W[idx, 1], tq)])
    return np.column_stack([CubicSpline(ts, W[idx, 0])(tq), CubicSpline(ts, W[idx, 1])(tq)])

# ---------- 01 the ground-truth track ----------
xr = (-560, -150); yr = (-345, 115)
sc = 1.05  # px per metre -> true equal aspect
pw, ph = int((xr[1]-xr[0]) * sc), int((yr[1]-yr[0]) * sc)
f = Fig(pw + 110, ph + 115)
p = Panel(f, 70, 45, pw, ph, xr, yr, "The ground truth: 96.8 s of driving at 30 Hz",
          xticks=[-500, -400, -300, -200], yticks=[-300, -200, -100, 0, 100], xlabel="x (m)", ylabel="y (m)")
p.line(W[:, 0], W[:, 1], GRAY, 2)
p.dots([W[0, 0]], [W[0, 1]], GREEN, 5.5)
p.dots([W[-1, 0]], [W[-1, 1]], BLUE, 5.5)
p.dots([W[1266, 0]], [W[1266, 1]], RED, 4, True)
p.dots([W[1543, 0]], [W[1543, 1]], ORANGE, 4, True)
p.label(-547, 107, "start / finish (FAH)", GREEN, 11, weight="bold")
p.label(W[1266, 0], W[1266, 1], "SME roundabout", RED, 11, dx=12, dy=-8)
p.label(W[1543, 0], W[1543, 1], "trolley stop (turnaround)", ORANGE, 11, anchor="end", dx=-10, dy=14)
f.text((pw + 110) // 2, ph + 108, "2905 waypoints covering 1.4 km. Point spacing 0.10–0.87 m varies with speed.", 11.5, "middle", GRAY)
print("01", f.save(OUT + "01_track.svg"))

# ---------- 02 corner cutting at the roundabout (1 Hz) ----------
xr = (-262, -206); yr = (-242, -208)
ww = 760; hh = ww * (yr[1]-yr[0]) / (xr[1]-xr[0])
f = Fig(860, int(hh) + 120)
p = Panel(f, 55, 50, ww, int(hh), xr, yr, "Zoom: the roundabout, rebuilt from 1 Hz samples",
          xticks=[-260, -250, -240, -230, -220, -210], yticks=[-240, -230, -220, -210], xlabel="x (m)")
tq = t
idx = samples(30)
p.line(W[:, 0], W[:, 1], GRAY, 3.5)
for mname, col in [("linear", BLUE), ("quadratic", PURPLE), ("spline", GREEN)]:
    R = recon(idx, mname, tq)
    p.line(R[:, 0], R[:, 1], col, 1.8)
inr = idx[(W[idx,0]>=xr[0])&(W[idx,0]<=xr[1])&(W[idx,1]>=yr[0])&(W[idx,1]<=yr[1])]
p.dots(W[inr, 0], W[inr, 1], ORANGE, 4.5)
f.parts.append('<g font-size="11">')
for lab, col, y0 in [("ground truth (30 Hz)", GRAY, 62), ("1 Hz samples", ORANGE, 79),
                     ("linear", BLUE, 96), ("quadratic", PURPLE, 113), ("cubic spline", GREEN, 130)]:
    f.parts.append(f'<path d="M640 {y0-4}h20" stroke="{col}" stroke-width="3"/>')
    f.text(666, y0, lab, 11)
f.parts.append('</g>')
f.text(430, int(hh) + 105, "Linear cuts every corner into chords. The quadratic overshoots where parabola windows meet. The spline hugs the curve.", 11.5, "middle", GRAY)
print("02", f.save(OUT + "02_roundabout_1hz.svg"))

# ---------- 03 the same at 0.2 Hz: full southern leg ----------
xr = (-410, -195); yr = (-340, -40)
ww = 500; hh = ww * (yr[1]-yr[0]) / (xr[1]-xr[0])
f = Fig(860, int(hh) + 120)
p = Panel(f, 60, 50, ww, int(hh), xr, yr, "At 0.2 Hz (one sample every 5 s) the road itself is gone",
          xticks=[-400, -300, -200], yticks=[-300, -200, -100], xlabel="x (m)")
idx = samples(150)
p.line(W[:, 0], W[:, 1], GRAY, 3)
for mname, col in [("linear", BLUE), ("spline", GREEN)]:
    R = recon(idx, mname, tq)
    p.line(R[:, 0], R[:, 1], col, 1.8)
inr = idx[(W[idx,0]>=xr[0])&(W[idx,0]<=xr[1])&(W[idx,1]>=yr[0])&(W[idx,1]<=yr[1])]
p.dots(W[inr, 0], W[inr, 1], ORANGE, 5)
f.parts.append('<g font-size="11">')
for lab, col, y0 in [("ground truth", GRAY, 62), ("0.2 Hz samples", ORANGE, 79),
                     ("linear", BLUE, 96), ("cubic spline", GREEN, 113)]:
    f.parts.append(f'<path d="M640 {y0-4}h20" stroke="{col}" stroke-width="3"/>')
    f.text(666, y0, lab, 11)
f.parts.append('</g>')
f.text(430, int(hh) + 105, "21 samples for 1.4 km. Both methods miss the roundabout loop entirely — no interpolation can invent unseen geometry.", 11.5, "middle", GRAY)
print("03", f.save(OUT + "03_leg_02hz.svg"))

# ---------- 04 cumulative error bars (log scale) ----------
from scipy.interpolate import CubicSpline as CS
f = Fig(760, 330)
p = Panel(f, 70, 45, 620, 210, (-0.6, 8.6), (0, 4.6), "Cumulative error: method matters only once sampling gets sparse",
          xticks=None, yticks=[0, 1, 2, 3, 4], ylabel="log10 cum. error (m)")
vals = {}
for r, step in [("10 Hz", 3), ("1 Hz", 30), ("0.2 Hz", 150)]:
    idx = samples(step)
    for mname in ["linear", "quadratic", "spline"]:
        R = recon(idx, mname, t)
        vals[(r, mname)] = np.linalg.norm(R - W, axis=1).sum()
cols = {"linear": BLUE, "quadratic": PURPLE, "spline": GREEN}
gx = 0
for gi, r in enumerate(["10 Hz", "1 Hz", "0.2 Hz"]):
    for mi, mname in enumerate(["linear", "quadratic", "spline"]):
        x0 = gi * 3 + mi
        v = np.log10(vals[(r, mname)])
        p.bar(x0, v, 0.72, cols[mname])
        p.label(x0, v, f"{vals[(r, mname)]:.0f}", BLACK, 10, anchor="middle", dy=-5)
    p.label(gi * 3 + 1, -0.0, r, BLACK, 12, anchor="middle", dy=34, weight="bold")
f.parts.append('<g font-size="11">')
for lab, col, i in [("linear", BLUE, 0), ("quadratic", PURPLE, 1), ("cubic spline", GREEN, 2)]:
    f.parts.append(f'<rect x="{90+i*110}" y="52" width="12" height="12" fill="{col}"/>')
    f.text(107 + i * 110, 62, lab, 11)
f.parts.append('</g>')
f.text(380, 320, "Sum of 2D errors over all 2905 ground-truth timestamps (metres, log scale; numbers above bars are the raw sums).", 11.5, "middle", GRAY)
print("04", f.save(OUT + "04_error_bars.svg"))

# ---------- 05 Q2: error vs h, log-log ----------
hs = np.logspace(-3.2, -0.6, 28)
el, eq2 = [], []
xq = np.linspace(0, np.pi, 400001)
cq = np.cos(xq)
for hh2 in hs:
    m = max(2, int(np.ceil(np.pi / hh2)))
    xs = np.linspace(0, np.pi, m + 1)
    el.append(np.abs(np.interp(xq, xs, np.cos(xs)) - cq).max())
    m2 = m + (m % 2)
    xs2 = np.linspace(0, np.pi, m2 + 1)
    eq2.append(np.abs(quad_interp(xs2, np.cos(xs2), xq) - cq).max())
f = Fig(760, 360)
p = Panel(f, 70, 45, 620, 240, (-3.3, -0.5), (-10.5, -0.5), "cos-table error vs spacing h (both axes log10)",
          xticks=[-3, -2.5, -2, -1.5, -1, -0.5], yticks=[-2, -4, -6, -8, -10], xlabel="log10 h (rad)", ylabel="log10 max error")
p.line(np.log10(hs), np.log10(np.array(el)), BLUE, 2.4)
p.line(np.log10(hs), np.log10(np.array(eq2)), GREEN, 2.4)
p.hline(np.log10(5e-7), RED, 1.5)
p.line(np.log10(hs), 2 * np.log10(hs) - np.log10(8), GRAY, 1.2, "4 3")
p.line(np.log10(hs), 3 * np.log10(hs) - np.log10(9 * np.sqrt(3)), GRAY, 1.2, "2 3")
p.dots([np.log10(2e-3)], [np.log10(5e-7)], BLUE, 5)
p.dots([np.log10(0.019827)], [np.log10(5e-7)], GREEN, 5)
p.label(np.log10(2e-3), np.log10(5e-7), "h = 0.0020 → 1572 entries", BLUE, 11, dx=-10, dy=22, anchor="start", weight="bold")
p.label(np.log10(0.019827), np.log10(5e-7), "h = 0.0198 → 161 entries", GREEN, 11, dx=4, dy=26, weight="bold")
p.label(-1.35, -2.6, "slope 2 (linear: h²/8)", GRAY, 11, anchor="end")
p.label(-0.62, -4.9, "slope 3 (quadratic: h³/9√3)", GRAY, 11, anchor="end")
p.label(-2.48, np.log10(5e-7), "6-digit target 5·10⁻⁷", RED, 11, dy=16)
f.text(380, 345, "Straight lines on a log-log plot = power laws. Measured error sits exactly on the textbook bounds (dashed).", 11.5, "middle", GRAY)
print("05", f.save(OUT + "05_table_spacing.svg"))
