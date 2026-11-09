"""Lecture 11 figures (regression & classification)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from svgplot import Fig, Panel, BLUE, RED, GREEN, GRAY, ORANGE, PURPLE, BLACK

OUT = "docs/figures/lec11/"
os.makedirs(OUT, exist_ok=True)
AMBER = "#BA7517"
rng = np.random.default_rng(11)

# ---------- 01: overfitting (same seed as verify.py) ----------
N = 10
xs = np.linspace(0, 1, N)
ts = np.sin(2*np.pi*xs) + 0.25*rng.standard_normal(N)
xt = np.linspace(0, 1, 160)
truth = np.sin(2*np.pi*xt)
def design(x, M): return np.vander(x, M+1, increasing=True)
def fit(M, lam=0.0):
    P = design(xs, M)
    return np.linalg.solve(lam*np.eye(M+1) + P.T@P, P.T@ts)
fig = Fig(860, 330)
cases = [(1, "M = 1: too stiff", "test rms 0.45", ORANGE),
         (3, "M = 3: right-sized", "test rms 0.15 ✓", GREEN),
         (9, "M = 9: memorizes noise", "test rms 0.45", RED)]
for i, (M, title, lab, col) in enumerate(cases):
    p = Panel(fig, 55 + i*272, 45, 240, 210, (-0.03, 1.03), (-1.9, 1.9), title,
              xticks=[0, 1], yticks=[-1, 0, 1])
    w = fit(M)
    yy = design(xt, M)@w
    p.line(xt, truth, GRAY, 1.4, "5 4")
    p.line(xt, np.clip(yy, -1.85, 1.85), col, 2.3)
    p.dots(xs, ts, BLUE, 4)
    p.label(0.5, -1.72, lab, col, 11.5, "middle", weight="bold")
fig.text(430, 305, "10 noisy samples of sin(2πx) (dashed). Same least squares, three capacities: underfit, fit, overfit. Ridge (λ = 10⁻⁴) tames M = 9 back to test rms 0.14.", 11.5, "middle", GRAY)
print("01", fig.save(OUT + "01_overfitting.svg"))

# ---------- 02: bolts vs needles minimum-distance boundary ----------
mn = np.array([0.86, 2.34]); mb = np.array([5.74, 5.85])
needles = np.array([[0.6,2.0],[0.7,1.9],[0.75,2.1],[0.86,2.05],[0.9,2.4],[0.95,2.5],[1.0,2.5],[1.1,2.6],[1.5,3.0],[0.65,1.85]])
bolts = np.array([[4.0,6.5],[4.1,6.0],[4.5,5.8],[4.5,5.5],[4.9,5.5],[5.0,5.0],[7.0,6.0],[7.8,6.2],[7.8,6.0],[7.9,6.1]])
fig = Fig(860, 380)
p = Panel(fig, 60, 45, 500, 290, (0, 8.4), (0, 8), "Bolts vs needles: the minimum-distance rule is a LINE",
          xticks=[0, 2, 4, 6, 8], yticks=[0, 2, 4, 6, 8], xlabel="length (cm)", ylabel="head diameter (mm)")
# boundary -4.88 x1 - 3.51 x2 + 30.48 = 0  ->  x2 = (30.48 - 4.88 x1)/3.51
bx = np.linspace(0.4, 6.3, 50)
p.line(bx, (30.4774 - 4.88*bx)/3.51, RED, 2.2, "6 4")
for q in needles:
    fig.parts.append(f'<path d="M{p.X(q[0])-4:.1f} {p.Y(q[1]):.1f}h8M{p.X(q[0]):.1f} {p.Y(q[1])-4:.1f}v8" stroke="{BLUE}" stroke-width="1.6"/>')
for q in bolts:
    fig.parts.append(f'<circle cx="{p.X(q[0]):.1f}" cy="{p.Y(q[1]):.1f}" r="4" fill="none" stroke="{GREEN}" stroke-width="1.6"/>')
p.dots([mn[0]], [mn[1]], BLUE, 6)
p.dots([mb[0]], [mb[1]], GREEN, 6)
p.label(mn[0]+0.3, mn[1]-0.75, "m_needle = (0.86, 2.34)", BLUE, 11, weight="bold")
p.label(mb[0]+0.35, mb[1]-0.45, "m_bolt = (5.74, 5.85)", GREEN, 11, weight="bold")
p.label(5.6, 2.6, "d_needle = d_bolt", RED, 11.5, weight="bold")
X0 = 596
fig.text(X0, 90, "expand ‖x − m‖² and the", 12, "start", BLACK)
fig.text(X0, 108, "squared terms cancel:", 12, "start", BLACK)
fig.text(X0, 140, "d_needle = 0.86x₁+2.34x₂−3.11", 11.5, "start", BLUE)
fig.text(X0, 162, "d_bolt = 5.74x₁+5.85x₂−33.59", 11.5, "start", GREEN)
fig.text(X0, 194, "boundary (difference):", 12, "start", BLACK)
fig.text(X0, 216, "−4.88x₁ − 3.51x₂ + 30.48 = 0", 11.5, "start", RED, weight="bold")
fig.text(X0, 248, "⊥ bisector of the two means —", 11, "start", GRAY)
fig.text(X0, 264, "nearest-mean IS a linear", 11, "start", GRAY)
fig.text(X0, 280, "discriminant.", 11, "start", GRAY)
fig.text(430, 374, "The slides' own numbers, verified. Nearest-mean classification draws the perpendicular bisector between the means.", 11.5, "middle", GRAY)
print("02", fig.save(OUT + "02_bolts_needles.svg"))

# ---------- 03: Fisher vs naive projection ----------
C = np.array([[1.2, 1.05], [1.05, 1.0]])
f1 = rng.multivariate_normal([0, 0], C, 300)
f2 = rng.multivariate_normal([1.6, 0.4], C, 300)
m1_, m2_ = f1.mean(0), f2.mean(0)
Sw = (f1 - m1_).T@(f1 - m1_) + (f2 - m2_).T@(f2 - m2_)
w_f = np.linalg.solve(Sw, m2_ - m1_); w_f /= np.linalg.norm(w_f)
w_n = (m2_ - m1_)/np.linalg.norm(m2_ - m1_)
def class_err(w):
    p1, p2 = f1@w, f2@w
    thr = (p1.mean() + p2.mean())/2
    s = 1 if p2.mean() > p1.mean() else -1
    return (np.sum(s*(p1 - thr) > 0) + np.sum(s*(p2 - thr) < 0))/600
print(f"   fig3 errors: naive {100*class_err(w_n):.1f}%, fisher {100*class_err(w_f):.1f}%")
fig = Fig(860, 350)
for i, (w, title, col, errtxt) in enumerate([
        (w_n, "project onto m₂ − m₁ (naive)", ORANGE, f"{100*class_err(w_n):.1f}% wrong"),
        (w_f, "project onto S_w⁻¹(m₂ − m₁) (Fisher)", GREEN, f"{100*class_err(w_f):.1f}% wrong")]):
    allp = np.concatenate([f1@w, f2@w])
    lo, hi = np.quantile(allp, 0.002), np.quantile(allp, 0.998)
    pad = 0.06*(hi-lo)
    p = Panel(fig, 60 + i*420, 45, 350, 230, (lo-pad, hi+pad), (0, 0.9), title, xticks=None, yticks=None, grid=False)
    for (data, cc) in ((f1, BLUE), (f2, RED)):
        proj = data@w
        h, edges = np.histogram(proj, bins=34, range=(lo, hi))
        hmax = h.max()
        for b in range(34):
            if h[b]:
                p.bar(edges[b], 0.82*h[b]/hmax, (edges[1]-edges[0])*0.88, cc)
    p.label(lo, 0.86, errtxt, col, 12.5, weight="bold")
fig.text(430, 330, "Same correlated clusters, two 1-D projections: Fisher divides the mean difference by the within-class spread (S_w⁻¹) and the histograms separate.", 11.5, "middle", GRAY)
print("03", fig.save(OUT + "03_fisher.svg"))
