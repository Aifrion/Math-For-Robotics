"""Lecture 3 (interpolation) verification."""
import numpy as np

print("=== slides' Lagrange example ===")
# points (0,1),(1,0) -> p(x) = -x+1
# add (-1,4): p(x) should recover (x-1)^2
xs = np.array([0, 1, -1]); ys = np.array([1, 0, 4])
c = np.polyfit(xs, ys, 2)
print("coeffs:", c.round(6), " -> (x-1)^2 = x^2 - 2x + 1 ✓" if np.allclose(c, [1, -2, 1]) else "MISMATCH")
L0 = lambda x: -(x-1)*(x+1); L2 = lambda x: 0.5*x*(x-1)
p = lambda x: 1*L0(x) + 4*L2(x)
print("p at (0,1,-1):", p(xs[0]), p(xs[1]), p(xs[2]))

print("\n=== Vandermonde conditioning (why not to solve V a = y) ===")
for n in [5, 10, 15, 20]:
    x = np.linspace(0, 1, n)
    V = np.vander(x, increasing=True)
    print(f"  n={n:2d}: cond(V) = {np.linalg.cond(V):.2e}")

print("\n=== spline C,D formulas (NR 3.3) vs scipy ===")
from scipy.interpolate import CubicSpline
xk = np.array([0., 1., 2.5, 3., 4.5])
yk = np.sin(xk)
cs = CubicSpline(xk, yk, bc_type='natural')
y2 = cs(xk, 2)  # second derivatives at knots
xq = 1.7
j = 1
h = xk[j+1]-xk[j]
A = (xk[j+1]-xq)/h; B = 1-A
C = (A**3-A)*h*h/6; D = (B**3-B)*h*h/6
val = A*yk[j]+B*yk[j+1]+C*y2[j]+D*y2[j+1]
print("  NR formula:", val, " scipy:", cs(xq), " match:", np.isclose(val, cs(xq)))

print("\n=== bilinear: weights sum to 1, exact for planes ===")
t, u = 0.3, 0.8
w = [(1-t)*(1-u), t*(1-u), t*u, (1-t)*u]
print("  weights:", np.round(w, 3), "sum:", sum(w))
f = lambda x, y: 2 + 3*x - 5*y
corners = [f(0,0), f(1,0), f(1,1), f(0,1)]
print("  plane: interp", np.dot(w, corners), " exact", f(t, u))

print("\n=== kriging / GP with Gaussian basis + regularization ===")
rng = np.random.default_rng(1)
X = np.array([0, .16, .32, .5, .63, .8, 1.])
Y = np.array([.1, .2, .8, .6, .3, .86, .7])
s = 0.12
Phi = np.exp(-(X[:, None]-X[None, :])**2/(2*s*s))
for lam in [1e-3, 0.3]:
    w = np.linalg.solve(lam*np.eye(len(X)) + Phi.T@Phi, Phi.T@Y)
    pred = Phi @ w
    print(f"  lambda={lam}: max |pred - y| at data = {np.abs(pred-Y).max():.4f}",
          "(interpolates)" if np.abs(pred-Y).max() < 0.02 else "(smooths, no longer exact)")
