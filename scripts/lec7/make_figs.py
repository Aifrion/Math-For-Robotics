"""Lecture 7 figures."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from svgplot import Fig, Panel, BLUE, RED, GREEN, GRAY, ORANGE, PURPLE, BLACK

OUT = "docs/figures/lec7/"
os.makedirs(OUT, exist_ok=True)

# ---------- 01 quadrature error vs N ----------
f = np.sin; a, b = 0, np.pi; I = 2.0
def trap(N):
    x = np.linspace(a, b, N+1); y = f(x); h = (b-a)/N
    return h*(y[0]/2 + y[1:-1].sum() + y[-1]/2)
def simp(N):
    x = np.linspace(a, b, N+1); y = f(x); h = (b-a)/N
    return h/3*(y[0] + 4*y[1:-1:2].sum() + 2*y[2:-2:2].sum() + y[-1])
def midp(N):
    h = (b-a)/N; return h*f(a + (np.arange(N)+0.5)*h).sum()
Ns = [2**k for k in range(1, 10)]
fig = Fig(760, 360)
p = Panel(fig, 70, 45, 620, 250, (0.2, 2.8), (-16, 0), "Error vs number of panels for ∫₀^π sin x dx = 2 (log-log)",
          xticks=[1, 2], yticks=[0, -4, -8, -12, -16], xlabel="log10 N", ylabel="log10 |error|", logy=True)
EL = lambda fn: [max(-16, np.log10(abs(fn(N)-I)+1e-18)) for N in Ns]
LN = [np.log10(N) for N in Ns]
p.line(LN, EL(trap), BLUE, 2.2); p.dots(LN, EL(trap), BLUE, 3)
p.line(LN, EL(midp), ORANGE, 2.2); p.dots(LN, EL(midp), ORANGE, 3)
p.line(LN, EL(simp), GREEN, 2.2); p.dots(LN, EL(simp), GREEN, 3)
# romberg diagonal
R = [[trap(2**k)] for k in range(1, 10)]
for k in range(1, 9):
    for j in range(1, k+1):
        R[k].append(R[k][j-1] + (R[k][j-1]-R[k-1][j-1])/(4**j-1))
rl = [max(-16, np.log10(abs(R[k][k]-I)+1e-18)) for k in range(9)]
p.line(LN, rl, PURPLE, 2.2); p.dots(LN, rl, PURPLE, 3)
p.legend([("trapezoid — slope −2", BLUE, None), ("midpoint — slope −2, half the error", ORANGE, None),
          ("Simpson — slope −4", GREEN, None), ("Romberg diagonal — superpolynomial", PURPLE, None)], x=96, y=200)
fig.text(380, 345, "Doubling N buys 2 digits (Simpson's ×16) vs 0.6 digits (trapezoid's ×4). Romberg reuses the SAME trapezoid sums and dives to machine precision.", 11, "middle", GRAY)
print("01", fig.save(OUT + "01_quadrature_orders.svg"))

# ---------- 02 RK4 anatomy ----------
fig = Fig(760, 348)
p = Panel(fig, 60, 45, 640, 230, (0, 1.12), (0.9, 3.1), "One RK4 step on y′ = y: four slope samples, one weighted average",
          xticks=[0, 0.5, 1], yticks=[1, 2, 3], xlabel="x (one step, h = 1)")
xs = np.linspace(0, 1.12, 80)
p.line(xs, np.exp(xs), GRAY, 2)
h = 1.0; y0 = 1.0
k1 = h*y0
k2 = h*(y0 + k1/2)
k3 = h*(y0 + k2/2)
k4 = h*(y0 + k3)
ynew = y0 + k1/6 + k2/3 + k3/3 + k4/6
def slope_seg(x0, y0_, s, col, lab, dy=-8):
    d = 0.16
    p.line([x0-d, x0+d], [y0_-s*d, y0_+s*d], col, 2.6)
    p.dots([x0], [y0_], col, 4)
    p.label(x0, y0_, lab, col, 12, dx=8, dy=dy, weight="bold")
slope_seg(0, 1.0, k1/h, BLUE, "k₁: slope at start", dy=22)
slope_seg(0.5, y0+k1/2, k2/h, ORANGE, "k₂: slope at midpoint, using k₁", dy=18)
slope_seg(0.5, y0+k2/2, k3/h, PURPLE, "k₃: midpoint again, using k₂", dy=-22)
d = 0.16
p.line([1.0-d, 1.0+d], [y0+k3-(k4/h)*d, y0+k3+(k4/h)*d], RED, 2.6)
p.dots([1.0], [y0+k3], RED, 4)
p.label(1.0, y0+k3, "k₄: slope at the end, using k₃", RED, 12, anchor="end", dx=-12, dy=-6, weight="bold")
p.dots([1.0], [ynew], GREEN, 6)
p.label(1.0, ynew, f"y₊ = y + (k₁+2k₂+2k₃+k₄)/6 = {ynew:.4f}  (true e = {np.e:.4f})", GREEN, 12, anchor="end", dx=-12, dy=30, weight="bold")
fig.text(380, 334, "Four cheap slope probes per step, weighted 1:2:2:1, cancel error terms up to h⁴ — per-step error O(h⁵). Even with h = 1, RK4 lands within 0.2%.", 11.5, "middle", GRAY)
print("02", fig.save(OUT + "02_rk4_anatomy.svg"))

# ---------- 03 stiffness: two time scales ----------
fig = Fig(860, 345)
p = Panel(fig, 55, 45, 340, 220, (0, 0.012), (0, 2.1), "First 0.012 s: the fast part dies",
          xticks=[0, 0.005, 0.01], yticks=[0, 1, 2], xlabel="x")
xs = np.linspace(0, 0.012, 300)
u = 2*np.exp(-xs) - np.exp(-1000*xs)
p.line(xs, u, BLUE, 2.4)
p.line(xs, 2*np.exp(-xs), GRAY, 1.4, "4 3")
p.label(0.0062, 1.62, "u(x) = 2e⁻ˣ − e⁻¹⁰⁰⁰ˣ", BLUE, 12, anchor="middle", dy=26, weight="bold")
p.label(0.0075, 2.02, "2e⁻ˣ (slow part)", GRAY, 11, dy=16)
q = Panel(fig, 475, 45, 340, 220, (0, 5), (0, 2.1), "Next 5 s: only the slow part remains",
          xticks=[0, 1, 2, 3, 4, 5], yticks=[0, 1, 2], xlabel="x")
xs2 = np.linspace(0, 5, 300)
q.line(xs2, 2*np.exp(-xs2) - np.exp(-1000*xs2), BLUE, 2.4)
fig.text(430, 310, "The slides' stiff system (eigenvalues −1 and −1000). The e⁻¹⁰⁰⁰ˣ term is invisible after x ≈ 0.01, yet explicit Euler must keep", 11.5, "middle", GRAY)
fig.text(430, 328, "h < 2/1000 = 0.002 forever: at h = 0.0021 the solution explodes to −7×10⁹ (verified). The dead component still owns your step size.", 11.5, "middle", GRAY)
print("03", fig.save(OUT + "03_stiffness.svg"))
