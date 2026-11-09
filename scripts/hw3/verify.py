"""Verify every number quoted on the HW3 companion page.

Q1: dy/dx = 1/(x^2 (1-y)), y(1) = -1, on (0, 1].
Q2: plane fitting in the three point clouds.

Run: python3 scripts/hw3/verify.py
"""
import numpy as np

ok = True
def check(name, got, want, tol=1e-9):
    global ok
    good = abs(float(got) - float(want)) <= tol
    ok &= good
    print(f"{'OK ' if good else 'FAIL'} {name}: {got!r} (want {want!r})")

print("=" * 60)
print("Q1: the ODE")
print("=" * 60)

# ---------------------------------------------------------------
# Exact solution by separation:
#   (1-y) dy = dx / x^2
#   y - y^2/2 = -1/x + C;  y(1) = -1  =>  C = -1/2
#   y^2 - 2y - (1 + 2/x) = 0  =>  y = 1 - sqrt(2 + 2/x)   (branch with y(1)=-1)
# ---------------------------------------------------------------
y_exact = lambda x: 1.0 - np.sqrt(2.0 + 2.0/x)
check("y_exact(1) = -1", y_exact(1.0), -1.0, 1e-15)

# implicit relation holds along the curve
for x in [0.05, 0.1, 0.3, 0.7, 1.0]:
    y = y_exact(x)
    check(f"implicit relation at x={x}", y - y*y/2, -1/x - 0.5, 1e-10)

# it actually solves the ODE: compare dy/dx numerically to 1/(x^2(1-y))
for x in [0.1, 0.5, 0.9]:
    h = 1e-6
    dydx = (y_exact(x+h) - y_exact(x-h)) / (2*h)
    check(f"ODE satisfied at x={x}", dydx, 1/(x*x*(1-y_exact(x))), 1e-5)

# cross-check against a high-accuracy integrator (backward from x=1)
from scipy.integrate import solve_ivp
f = lambda x, y: 1.0/(x*x*(1.0-y))
sol = solve_ivp(f, [1.0, 0.05], [-1.0], rtol=1e-12, atol=1e-12, dense_output=True)
check("solve_ivp matches exact at x=0.05", sol.y[0][-1], y_exact(0.05), 1e-7)

# y(0): diverges. y ~ -sqrt(2/x) -> -inf as x -> 0+
print(f"   y(0.05) = {y_exact(0.05):.6f},  y(0.01) = {y_exact(0.01):.4f}, "
      f"y(0.001) = {y_exact(0.001):.3f}  -> -inf as x->0+")
check("y(0.05) = 1 - sqrt(42)", y_exact(0.05), 1 - np.sqrt(42), 1e-12)
check("y(0.05) value", round(y_exact(0.05), 4), -5.4807, 0)

# ---------------------------------------------------------------
# Numerical methods, h = 0.05, integrating from x=1 DOWN to x=0
# (the step is -h). Grid: x = 1.00, 0.95, ..., 0.05, 0.00.
# ---------------------------------------------------------------
H = 0.05
def euler():
    xs = [1.0]; ys = [-1.0]
    for k in range(20):
        x, y = xs[-1], ys[-1]
        ys.append(y - H*f(x, y))     # f evaluated at current x (never at 0)
        xs.append(round(x - H, 10))
    return np.array(xs), np.array(ys)

def rk4():
    xs = [1.0]; ys = [-1.0]
    for k in range(20):
        x, y = xs[-1], ys[-1]
        xn = round(x - H, 10)
        if xn <= 0:                   # last step: k4 would need f(0, .) -> undefined
            xs.append(xn); ys.append(np.nan); break
        k1 = f(x, y)
        k2 = f(x - H/2, y - H/2*k1)
        k3 = f(x - H/2, y - H/2*k2)
        k4 = f(xn, y - H*k3)
        ys.append(y - H/6*(k1 + 2*k2 + 2*k3 + k4))
        xs.append(xn)
    return np.array(xs), np.array(ys)

xe, ye = euler()
xr, yr = rk4()
check("euler reaches x=0", xe[-1], 0.0, 1e-12)
print(f"   Euler y(0) estimate = {ye[-1]:.4f}  (finite! true y(0) = -inf; "
      f"Euler never evaluates f at x=0)")

# errors at x = 0.05 (last point where the truth is finite on the grid)
i = np.argmin(np.abs(xe - 0.05)); assert abs(xe[i] - 0.05) < 1e-9
err_eul_005 = abs(ye[i] - y_exact(0.05))
err_rk4_005 = abs(yr[i] - y_exact(0.05))
print(f"   at x=0.05: exact {y_exact(0.05):.6f}, Euler {ye[i]:.6f} "
      f"(err {err_eul_005:.4f}), RK4 {yr[i]:.6f} (err {err_rk4_005:.2e})")

# errors away from the singularity, x = 0.5
i5 = np.argmin(np.abs(xe - 0.5))
err_eul_05 = abs(ye[i5] - y_exact(0.5))
err_rk4_05 = abs(yr[i5] - y_exact(0.5))
print(f"   at x=0.50: Euler err {err_eul_05:.2e}, RK4 err {err_rk4_05:.2e}")

# ---------------------------------------------------------------
# Richardson extrapolation on Euler: run h and h/2, combine
#   y_R = 2*y_{h/2} - y_h   (Euler is first order)
# ---------------------------------------------------------------
def euler_h(h, x_end=0.05):
    x, y = 1.0, -1.0
    n = int(round((1.0 - x_end)/h))
    for _ in range(n):
        y = y - h*f(x, y)
        x = round(x - h, 12)
    return y

yh  = euler_h(H)
yh2 = euler_h(H/2)
y_rich = 2*yh2 - yh
err_rich = abs(y_rich - y_exact(0.05))
print(f"   Richardson(Euler) at x=0.05: {y_rich:.6f} (err {err_rich:.2e}) "
      f"vs plain Euler err {abs(yh - y_exact(0.05)):.2e}")
check("Richardson beats Euler at 0.05", float(err_rich < abs(yh - y_exact(0.05))), 1.0, 0)

# Richardson at x=0.5 too (clean region: should gain ~an order)
yh_5, yh2_5 = euler_h(H, 0.5), euler_h(H/2, 0.5)
err_rich_05 = abs(2*yh2_5 - yh_5 - y_exact(0.5))
print(f"   Richardson at x=0.50: err {err_rich_05:.2e} (Euler {err_eul_05:.2e}, RK4 {err_rk4_05:.2e})")

# measured convergence order of Euler on this problem (away from singularity)
e1 = abs(euler_h(0.05, 0.5) - y_exact(0.5))
e2 = abs(euler_h(0.025, 0.5) - y_exact(0.5))
p = np.log2(e1/e2)
print(f"   Euler measured order at x=0.5: {p:.3f} (theory 1)")
check("Euler order ~1", p, 1.0, 0.15)

print()
print("=" * 60)
print("Q2: plane fitting")
print("=" * 60)

# ---------------------------------------------------------------
# Load clouds. Plane fit: centroid + SVD; normal = singular vector
# of the smallest singular value; plane n.(p - c) = 0.
# ---------------------------------------------------------------
def load(path):
    P = np.loadtxt(path)
    assert P.shape[1] == 3
    valid = np.linalg.norm(P, axis=1) > 1e-6   # drop invalid-depth (0,0,0) pixels
    return P[valid], len(P) - valid.sum()

def fit_plane(P):
    c = P.mean(axis=0)
    _, S, Vt = np.linalg.svd(P - c, full_matrices=False)
    n = Vt[2]
    if n[2] < 0: n = -n            # orient consistently (+z toward camera)
    d = -n @ c
    rms = S[2]/np.sqrt(len(P))
    return n, d, rms

def dist(P, n, d):
    return np.abs(P @ n + d)

def robust_fit(P, thresh, iters=10):
    """Fit, keep points within thresh of the plane, refit."""
    Q = P
    for _ in range(iters):
        n, d, _ = fit_plane(Q)
        keep = dist(P, n, d) < thresh
        Qn = P[keep]
        if len(Qn) == len(Q) and np.array_equal(Qn[:1], Q[:1]): break
        Q = Qn
    n, d, rms = fit_plane(Q)
    return n, d, rms, keep.sum()

E, zE = load("hw3/Empty2.asc")
T, zT = load("hw3/TableWithObjects2.asc")
C, zC = load("hw3/CSE.asc")
print(f"   valid points (dropped (0,0,0) invalid-depth pixels):")
print(f"   empty {len(E)} (dropped {zE}), cluttered {len(T)} (dropped {zT}), CSE {len(C)} (dropped {zC})")

# ---------------------------------------------------------------
# CSE building: sequential RANSAC to pull out dominant planes
# ---------------------------------------------------------------
def ransac_plane(P, thresh, trials=300, rng=None):
    rng = np.random.default_rng(0) if rng is None else rng
    best = (0, None, None)
    n_pts = len(P)
    for _ in range(trials):
        idx = rng.choice(n_pts, 3, replace=False)
        A = P[idx]
        n = np.cross(A[1]-A[0], A[2]-A[0])
        nn = np.linalg.norm(n)
        if nn < 1e-9: continue
        n = n/nn
        d = -n @ A[0]
        count = (np.abs(P @ n + d) < thresh).sum()
        if count > best[0]: best = (count, n, d)
    # refine on inliers with LSQ, iterated to a fixed point
    _, n, d = best
    keep = np.abs(P @ n + d) < thresh
    for _ in range(8):
        n, d, rms = fit_plane(P[keep])
        keep2 = np.abs(P @ n + d) < thresh
        if (keep2 == keep).all(): break
        keep = keep2
    return n, d, keep

ang = lambda a, b: np.degrees(np.arccos(np.clip(abs(a @ b), -1, 1)))

# Both table clouds are full camera frames (table + floor + background),
# so a whole-cloud LSQ is NOT the table. Baseline to show why:
for name, P0 in (("empty", E), ("cluttered", T)):
    _, _, rms0 = fit_plane(P0)
    print(f"   naive whole-cloud LSQ ({name}): rms = {rms0*1000:.0f} mm -> not a plane at all")

# The table is the dominant plane in both views: RANSAC + LSQ refine.
TH = 0.01   # 1 cm inlier band (RealSense depth noise is a few mm)
nE, dE, keepE = ransac_plane(E, TH)
nT, dT, keepT = ransac_plane(T, TH)
rmsE = np.sqrt(np.mean(dist(E[keepE], nE, dE)**2))
rmsT = np.sqrt(np.mean(dist(T[keepT], nT, dT)**2))
if nE[2] < 0: nE, dE = -nE, -dE
if nT[2] < 0: nT, dT = -nT, -dT
tilt = ang(nE, nT)
off  = abs(dE - dT)
print(f"   table (empty):     n = {np.round(nE,4)}, d = {dE:.5f}, inliers {keepE.sum()}/{len(E)} ({100*keepE.mean():.0f}%), rms {rmsE*1000:.1f} mm")
print(f"   table (cluttered): n = {np.round(nT,4)}, d = {dT:.5f}, inliers {keepT.sum()}/{len(T)} ({100*keepT.mean():.0f}%), rms {rmsT*1000:.1f} mm")
print(f"   across captures: tilt {tilt:.2f} deg, offset {off*1000:.2f} mm "
      f"(different camera pose between the two scans -> a few deg is expected)")
check("both fits are tight planes (rms < 5 mm)", float(max(rmsE, rmsT) < 0.005), 1.0, 0)
check("clutter rejected: cluttered-view inliers well below 100%",
      float(0.1 < keepT.mean() < 0.9), 1.0, 0)
check("captures roughly agree (tilt < 5 deg)", float(tilt < 5.0), 1.0, 0)

# stability: split each cloud in half at random, fit both halves, compare
rng = np.random.default_rng(1)
for name, P0, bar in (("empty", E, 0.5), ("cluttered", T, 1.5)):
    m = rng.permutation(len(P0))
    h1, h2 = P0[m[:len(P0)//2]], P0[m[len(P0)//2:]]
    n1, d1, _ = ransac_plane(h1, TH)
    n2, d2, _ = ransac_plane(h2, TH)
    a = ang(n1, n2)
    print(f"   split-half stability ({name}): {a:.3f} deg")
    check(f"split-half stable ({name}) < {bar} deg", float(a < bar), 1.0, 0)

P = C.copy()
planes = []
for k in range(4):
    n, d, keep = ransac_plane(P, 0.15)
    planes.append((n, d, keep.sum(), len(P)))
    P = P[~keep]
for i, (n, d, cnt, outof) in enumerate(planes):
    print(f"   CSE plane {i+1}: n = {np.round(n,3)}, d = {d:.3f}, inliers {cnt}/{outof}")
check("plane 1 captures a dominant share (>20%)", float(planes[0][2] > 0.2*len(C)), 1.0, 0)
# dominant planes should be near-orthogonal walls/ground: check plane1 vs plane2 angle
a12 = ang(planes[0][0], planes[1][0])
print(f"   angle(plane1, plane2) = {a12:.1f} deg")

print()
print("ALL OK" if ok else "SOME CHECKS FAILED")
raise SystemExit(0 if ok else 1)
