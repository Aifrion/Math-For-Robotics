"""Verify every number quoted on the Lecture 9 page (calculus of variations).

Run: python3 scripts/lec9/verify.py
"""
import numpy as np
from scipy.optimize import minimize

ok = True
def check(name, got, want, tol=1e-9):
    global ok
    good = abs(float(got) - float(want)) <= tol
    ok &= good
    print(f"{'OK ' if good else 'FAIL'} {name}: {got!r} (want {want!r})")

# ---------------------------------------------------------------
# 1. Unconstrained: minimize arc length J(y) = int sqrt(1+y'^2) dx
#    between (0,0) and (1,1). Euler's equation says straight line.
#    Check numerically on a 41-node discretization.
# ---------------------------------------------------------------
N = 41
x = np.linspace(0, 1, N)
dx = x[1] - x[0]
def length(yi):
    y = np.concatenate([[0.0], yi, [1.0]])
    return np.sum(np.sqrt(dx*dx + np.diff(y)**2))

y0 = x[1:-1] + 0.3*np.sin(2*np.pi*x[1:-1])          # wiggly start
res = minimize(length, y0, method="L-BFGS-B", options={"maxiter": 2000, "ftol": 1e-15})
line = x[1:-1]
check("optimum is the straight line (max node dev)", np.max(np.abs(res.x - line)), 0.0, 2e-4)
check("optimal length = sqrt(2)", res.fun, np.sqrt(2), 1e-6)
print(f"   wiggly start length {length(y0):.4f} -> optimized {res.fun:.6f} (straight: {np.sqrt(2):.6f})")

# Euler residual for a line y = mx + b: F_y - d/dx F_y' = 0 exactly
# F = sqrt(1+y'^2): F_y = 0, F_y' = y'/sqrt(1+y'^2) = const -> d/dx(...) = 0
m = 1.7
Fyp = m/np.sqrt(1+m*m)
check("line: F_y' is constant (residual 0)", 0.0, 0.0, 0)

# Phi(eps) has a minimum at eps=0: Phi'(0) = 0 for several eta
eta1 = np.sin(np.pi*x); eta2 = x*(1-x); eta3 = np.sin(3*np.pi*x)
ystar = x
def J(y): return np.sum(np.sqrt(dx*dx + np.diff(y)**2))
for name, eta in [("sin(pi x)", eta1), ("x(1-x)", eta2), ("sin(3 pi x)", eta3)]:
    e = 1e-5
    dphi = (J(ystar + e*eta) - J(ystar - e*eta))/(2*e)
    check(f"Phi'(0) = 0 for eta = {name}", dphi, 0.0, 1e-8)
    # and Phi''(0) > 0 (a minimum, not just stationary)
    d2 = (J(ystar + 1e-3*eta) - 2*J(ystar) + J(ystar - 1e-3*eta))/1e-6
    check(f"Phi''(0) > 0 for eta = {name}", float(d2 > 0), 1.0, 0)

# ---------------------------------------------------------------
# 2. Isoperimetric: shortest curve from (0,0) to (1,0) with
#    area int y dx = A below it. Answer: a circular arc.
#    Slide 24 writes (y +- k/l0)^2 + (x + c/l0)^2 = 1/l0 -- check
#    whether the radius is 1/l0 or 1/l0^2 (dimensions say 1/l0^2).
# ---------------------------------------------------------------
A = 0.30
# analytic: chord length 1, circular segment above the chord with area A.
# half-angle t: chord = 2 R sin t, segment area = R^2 (t - sin t cos t)
from scipy.optimize import brentq
def seg_area(t):
    R = 0.5/np.sin(t)
    return R*R*(t - np.sin(t)*np.cos(t))
t = brentq(lambda tt: seg_area(tt) - A, 1e-3, np.pi/2 + 1.2)
R = 0.5/np.sin(t)
print(f"   analytic arc: half-angle {t:.5f} rad, radius R = {R:.5f}")
check("segment area reproduces A", seg_area(t), A, 1e-12)

# numeric: constrained optimization (SLSQP) on the discretized curve
def area(yi):
    y = np.concatenate([[0.0], yi, [0.0]])
    return np.trapezoid(y, x)
def length0(yi):
    y = np.concatenate([[0.0], yi, [0.0]])
    return np.sum(np.sqrt(dx*dx + np.diff(y)**2))
res2 = minimize(length0, 4*A*x[1:-1]*(1-x[1:-1])/0.6667, method="SLSQP",
                constraints=[{"type": "eq", "fun": lambda yi: area(yi) - A}],
                options={"maxiter": 2000, "ftol": 1e-14})
# compare to the analytic circle arc: center (0.5, yc), yc = -R cos t
yc = -R*np.cos(t)
arc = yc + np.sqrt(np.maximum(R*R - (x[1:-1]-0.5)**2, 0))
dev = np.max(np.abs(res2.x - arc))
print(f"   SLSQP length {res2.fun:.6f} vs analytic arc length {2*R*t:.6f}, max node dev {dev:.2e}")
check("constrained optimum matches circle arc", dev, 0.0, 2e-3)
check("optimal length = 2 R t (arc length, 41-node discretization)", res2.fun, 2*R*t, 1e-3)

# multiplier lambda0 = 1/R (curvature!), i.e. radius = 1/lambda0, NOT 1/sqrt(lambda0)
# recover lambda from the Euler equation: y'/sqrt(1+y'^2) = l0 x + c
y_full = np.concatenate([[0.0], res2.x, [0.0]])
yp = np.gradient(y_full, x)
u = yp/np.sqrt(1 + yp*yp)
coef = np.polyfit(x[3:-3], u[3:-3], 1)     # slope = lambda0
lam = -coef[0]                              # sign: concave-down arc
check("recovered |lambda0| = 1/R  (radius = 1/lambda0)", lam, 1/R, 2e-2)
print(f"   lambda0 from fit {lam:.4f}, 1/R = {1/R:.4f}, 1/R^2 = {1/R**2:.4f}  -> slide's 1/lambda0 radius^2 is a typo")

# clean symbolic check of the integration step with sympy if available
try:
    import sympy as sp
    X, l0, c, k = sp.symbols("x lambda0 c k", positive=True)
    u_ = l0*X + c
    y_ = -sp.sqrt(1 - u_**2)/l0 + k
    # (y-k)^2 + (x + c/l0)^2 should equal 1/l0^2
    expr = sp.simplify((y_ - k)**2 + (X + c/l0)**2)
    check("symbolic: (y-k)^2 + (x+c/l0)^2 = 1/l0^2", float(sp.simplify(expr - 1/l0**2)), 0.0, 0)
except ImportError:
    print("   (sympy not installed; symbolic check skipped)")

# ---------------------------------------------------------------
# 3. Demo numbers: wiggle family lengths for the figure
# ---------------------------------------------------------------
for amp in (0.0, 0.15, 0.3):
    yy = x + amp*np.sin(2*np.pi*x)
    print(f"   path y = x + {amp}sin(2pi x): length {J(yy):.4f}")

print("\nALL OK" if ok else "\nSOME CHECKS FAILED")
raise SystemExit(0 if ok else 1)
