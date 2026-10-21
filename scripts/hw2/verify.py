"""HW2 verification: every number quoted on the HW2 companion page comes from here.

Q1: reconstruct the 30 Hz Voigt-Drive trajectory from 10/1/0.2 Hz downsamples
    with linear / piecewise-quadratic / cubic-spline interpolation.
Q2: table spacing for cos x on [0, pi] with linear / quadratic interpolation.
"""
import numpy as np
from scipy.interpolate import CubicSpline

W = np.loadtxt("hw2/waypoints.csv", delimiter=",")
n = len(W)
t = np.arange(n) / 30.0  # 30 Hz timestamps
print(f"waypoints: {n} points, duration {t[-1]:.1f} s")
seg = np.linalg.norm(np.diff(W, axis=0), axis=1)
print(f"spacing: min {seg.min():.3f} m, median {np.median(seg):.3f} m, max {seg.max():.3f} m")
print(f"track length: {seg.sum():.1f} m")
print(f"speed: median {np.median(seg)*30:.1f} m/s, max {seg.max()*30:.1f} m/s")

def lin_interp(ts, xs, tq):
    return np.interp(tq, ts, xs)

def quad_interp(ts, xs, tq):
    """Piecewise quadratic: Lagrange parabola through consecutive non-overlapping
    triplets (0,1,2), (2,3,4), ... Evaluated on the two intervals each covers."""
    out = np.empty_like(tq)
    m = len(ts)
    for k in range(0, m - 2, 2):
        t0, t1, t2 = ts[k], ts[k+1], ts[k+2]
        y0, y1, y2 = xs[k], xs[k+1], xs[k+2]
        lo = t0 if k == 0 else t0
        hi = t2
        sel = (tq >= lo) & (tq <= hi) if k + 4 <= m else (tq >= lo)
        tt = tq[sel]
        L0 = (tt-t1)*(tt-t2)/((t0-t1)*(t0-t2))
        L1 = (tt-t0)*(tt-t2)/((t1-t0)*(t1-t2))
        L2 = (tt-t0)*(tt-t1)/((t2-t0)*(t2-t1))
        out[sel] = y0*L0 + y1*L1 + y2*L2
    # if m is even the last point starts a leftover single interval -> linear there
    if (m - 1) % 2 == 1:
        sel = tq >= ts[m-2]
        out[sel] = np.interp(tq[sel], ts[m-2:], xs[m-2:])
    return out

def spline_interp(ts, xs, tq):
    return CubicSpline(ts, xs)(tq)

methods = {"linear": lin_interp, "quadratic": quad_interp, "spline": spline_interp}
rates = {"10 Hz": 3, "1 Hz": 30, "0.2 Hz": 150}

print("\n=== Q1: cumulative error (sum over all 30 Hz samples of 2D distance, metres) ===")
print(f"{'rate':>8} {'#samples':>9} | " + " | ".join(f"{m:>16}" for m in methods))
results = {}
for rname, step in rates.items():
    idx = np.arange(0, n, step)
    # include the final ground-truth point so the whole track is covered
    if idx[-1] != n - 1:
        idx = np.append(idx, n - 1)
    ts = t[idx]
    row = []
    for mname, fn in methods.items():
        rec = np.column_stack([fn(ts, W[idx, 0], t), fn(ts, W[idx, 1], t)])
        d = np.linalg.norm(rec - W, axis=1)
        results[(rname, mname)] = dict(cum=d.sum(), mean=d.mean(), mx=d.max(),
                                       argmax=int(d.argmax()))
        row.append(f"cum {d.sum():9.2f}")
    print(f"{rname:>8} {len(idx):>9} | " + " | ".join(f"{r:>16}" for r in row))

print("\nper-method detail (mean / max error in metres, where max occurs):")
for (rname, mname), r in results.items():
    print(f"  {rname:>7} {mname:>10}: cum {r['cum']:10.3f}  mean {r['mean']:7.4f}  "
          f"max {r['mx']:7.3f} at t={r['argmax']/30:6.1f}s")

print("\n=== Q2: cos-table spacing on [0, pi] ===")
# Linear interpolation error bound: max|f''| h^2 / 8 = h^2/8 (|cos''| <= 1).
# Quadratic (3 equally spaced nodes, interior): max |w(x)|/3! * max|f'''|
#   with w = (x-x0)(x-x1)(x-x2); max |w| on [x0,x2] is 2h^3/(3*sqrt(3))
#   -> bound = h^3 / (9*sqrt(3)).
for tol, label in [(0.5e-6, "strict '6 decimal digits' (err <= 0.5e-6)"),
                   (1e-6, "looser reading (err <= 1e-6)")]:
    h_lin = np.sqrt(8 * tol)
    h_qua = (9 * np.sqrt(3) * tol) ** (1 / 3)
    n_lin = int(np.ceil(np.pi / h_lin)) + 1
    n_qua_intervals = int(np.ceil(np.pi / h_qua))
    if n_qua_intervals % 2: n_qua_intervals += 1  # need an even number of intervals
    n_qua = n_qua_intervals + 1
    print(f"\n  {label}")
    print(f"    linear   : h <= {h_lin:.6f} rad -> {n_lin} entries")
    print(f"    quadratic: h <= {h_qua:.6f} rad -> {n_qua} entries "
          f"({n_qua_intervals} intervals, rounded to even)")

# brute-force check of both bounds at the strict tolerance
tol = 0.5e-6
h_lin = np.sqrt(8 * tol)
xs = np.arange(0, np.pi + h_lin, h_lin); xs[-1] = np.pi
xq = np.linspace(0, np.pi, 2_000_001)
err_lin = np.abs(np.interp(xq, xs, np.cos(xs)) - np.cos(xq)).max()
h_qua = (9 * np.sqrt(3) * tol) ** (1 / 3)
m = int(np.ceil(np.pi / h_qua)); m += m % 2
xs2 = np.linspace(0, np.pi, m + 1)
eq = np.abs(quad_interp(xs2, np.cos(xs2), xq) - np.cos(xq)).max()
print(f"\n  brute-force check (strict tol 5e-7):")
print(f"    linear    max err = {err_lin:.3e}  (bound predicts <= {h_lin**2/8:.3e})")
print(f"    quadratic max err = {eq:.3e}  (bound predicts <= {h_qua**3/(9*np.sqrt(3)):.3e})")
