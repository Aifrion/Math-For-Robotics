"""HW3 figures. Every number comes from the same code paths as verify.py."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from svgplot import Fig, Panel, BLUE, RED, GREEN, GRAY, ORANGE, PURPLE, BLACK

OUT = "docs/figures/hw3/"
os.makedirs(OUT, exist_ok=True)
AMBER = "#BA7517"
rng = np.random.default_rng(3)

# ================= Q1 solvers (same as verify.py) =================
f = lambda x, y: 1.0/(x*x*(1.0-y))
y_exact = lambda x: 1.0 - np.sqrt(2.0 + 2.0/x)
H = 0.05

def euler():
    xs, ys = [1.0], [-1.0]
    for _ in range(20):
        x, y = xs[-1], ys[-1]
        ys.append(y - H*f(x, y)); xs.append(round(x - H, 10))
    return np.array(xs), np.array(ys)

def rk4():
    xs, ys = [1.0], [-1.0]
    for _ in range(19):           # stops at x=0.05: the step to 0 needs f(0,.)
        x, y = xs[-1], ys[-1]
        xn = round(x - H, 10)
        k1 = f(x, y); k2 = f(x - H/2, y - H/2*k1)
        k3 = f(x - H/2, y - H/2*k2); k4 = f(xn, y - H*k3)
        ys.append(y - H/6*(k1 + 2*k2 + 2*k3 + k4)); xs.append(xn)
    return np.array(xs), np.array(ys)

def euler_h(h, x_end):
    x, y = 1.0, -1.0
    for _ in range(int(round((1.0 - x_end)/h))):
        y = y - h*f(x, y); x = round(x - h, 12)
    return y

xe, ye = euler()
xr, yr = rk4()

# ---------- 01: the solution and the three methods ----------
fig = Fig(860, 380)
p = Panel(fig, 60, 48, 520, 280, (0, 1.02), (-9, 0.3),
          "y(x) = 1 − √(2 + 2/x): integrate from x = 1 backward toward the wall",
          xticks=[0, 0.25, 0.5, 0.75, 1], yticks=[0, -2, -4, -6, -8], ylabel="y")
xs = np.concatenate([np.linspace(0.028, 0.2, 200), np.linspace(0.2, 1.0, 160)])
p.line(xs, y_exact(xs), BLACK, 2.6)
p.line(xe, ye, ORANGE, 2, "5 4")
p.dots(xe[::2], ye[::2], ORANGE, 3)
p.line(xr, yr, GREEN, 2)
p.dots(xr[::2], yr[::2], GREEN, 3, True)
p.dots([1], [-1], AMBER, 6)
p.label(0.97, -0.75, "start: y(1) = −1", AMBER, 11, anchor="end", weight="bold")
p.label(0.09, -8.6, "y → −∞", RED, 11.5, weight="bold")
p.vseg(0.0, -9, 0.3, RED, 1.6, "5 4")
p.legend([("exact", BLACK, None), ("Euler h=0.05", ORANGE, "5 4"), ("RK4 h=0.05", GREEN, None)],
         x=p.x + 230, y=p.y + 120)
# right column: the three verdicts at x=0.05
X0 = 612
fig.text(X0, 70, "at x = 0.05:", 13, "start", BLACK, weight="bold")
fig.text(X0, 96, f"exact      {y_exact(0.05):.4f}", 12.5, "start", BLACK)
fig.text(X0, 120, f"Euler      {ye[-2]:.4f}  (err 1.01)", 12.5, "start", ORANGE)
fig.text(X0, 144, f"RK4        {yr[-1]:.4f}  (err 0.02)", 12.5, "start", GREEN)
fig.text(X0, 186, "at x = 0:", 13, "start", BLACK, weight="bold")
fig.text(X0, 212, "exact      −∞", 12.5, "start", BLACK)
fig.text(X0, 236, f"Euler      {ye[-1]:.2f} (!)", 12.5, "start", ORANGE)
fig.text(X0, 260, "RK4        can't step:", 12.5, "start", GREEN)
fig.text(X0, 278, "k₄ needs f(0, ·)", 12, "start", GRAY)
fig.text(430, 356, "The ODE blows up at x = 0. Euler never notices (it only ever evaluates f at the point it is leaving); RK4's last step is impossible.", 11.5, "middle", GRAY)
print("01", fig.save(OUT + "01_ode_solutions.svg"))

# ---------- 02: error vs x (log) for the three methods ----------
fig = Fig(760, 330)
p = Panel(fig, 70, 45, 620, 220, (0, 1.0), (-8, 1),
          "log₁₀ |error| marching from x = 1 toward the singularity (h = 0.05)",
          xticks=[0.05, 0.25, 0.5, 0.75, 1], yticks=[0, -2, -4, -6, -8], xlabel="x", logy=True)
grid = xe[1:-1]  # 0.95 .. 0.05
ee = np.abs(ye[1:-1] - y_exact(grid))
er = np.abs(yr[1:] - y_exact(xr[1:]))
rich = []
for x in grid:
    rich.append(abs(2*euler_h(H/2, x) - euler_h(H, x) - y_exact(x)))
p.line(grid, np.log10(ee + 1e-16), ORANGE, 2.2)
p.line(xr[1:], np.log10(er + 1e-16), GREEN, 2.2)
p.line(grid, np.log10(np.array(rich) + 1e-16), PURPLE, 2.2, "5 4")
p.legend([("Euler", ORANGE, None), ("Richardson on Euler", PURPLE, "5 4"), ("RK4", GREEN, None)],
         x=p.x + 220, y=p.y + 16)
p.label(0.5, np.log10(2.55e-2), "2.6e−2", ORANGE, 10.5, dy=-8)
p.label(0.5, np.log10(2.65e-4), "2.7e−4", PURPLE, 10.5, dy=-8)
p.label(0.5, np.log10(7.47e-7), "7.5e−7", GREEN, 10.5, dy=14)
fig.text(380, 310, "Each extra order of accuracy buys ~2-3 digits at the same cost — until the singularity at x = 0 eats everyone's accuracy.", 11.5, "middle", GRAY)
print("02", fig.save(OUT + "02_ode_errors.svg"))

# ================= Q2 data (same filtering as verify.py) =================
def load(path):
    P = np.loadtxt(path)
    return P[np.linalg.norm(P, axis=1) > 1e-6]

def fit_plane(P):
    c = P.mean(axis=0)
    _, S, Vt = np.linalg.svd(P - c, full_matrices=False)
    n = Vt[2]
    return n, -n @ c

def ransac_plane(P, thresh, trials=300, rr=None):
    rr = np.random.default_rng(0) if rr is None else rr
    best = (0, None, None)
    for _ in range(trials):
        A = P[rr.choice(len(P), 3, replace=False)]
        n = np.cross(A[1]-A[0], A[2]-A[0]); nn = np.linalg.norm(n)
        if nn < 1e-9: continue
        n /= nn; d = -n @ A[0]
        cnt = (np.abs(P @ n + d) < thresh).sum()
        if cnt > best[0]: best = (cnt, n, d)
    _, n, d = best
    keep = np.abs(P @ n + d) < thresh
    for _ in range(8):
        n, d = fit_plane(P[keep])
        keep2 = np.abs(P @ n + d) < thresh
        if (keep2 == keep).all(): break
        keep = keep2
    return n, d, keep

T = load("hw3/TableWithObjects2.asc")
nT, dT, keepT = ransac_plane(T, 0.01)
if nT[1] < 0: nT, dT = -nT, -dT

# ---------- 03: cluttered table, side view, inliers vs clutter ----------
fig = Fig(860, 360)
p = Panel(fig, 60, 45, 560, 270, (-0.05, 0.75), (1.55, 0.75),
          "Cluttered table, side slice (horizontal: y, down · vertical: depth z): RANSAC keeps the plane",
          xticks=[0, 0.2, 0.4, 0.6], yticks=[0.8, 1.0, 1.2, 1.4], ylabel="z")
def dotcloud(p, XY, color, opacity, r=1.4):
    """Compact scatter: one stroked path, ~12 bytes per point."""
    seen, d = set(), []
    for a, b in XY:
        q = (round(p.X(a)), round(p.Y(b)))
        if q in seen: continue
        seen.add(q)
        d.append(f"M{q[0]} {q[1]}h.1")
    p.f.parts.append(f'<path d="{"".join(d)}" stroke="{color}" stroke-opacity="{opacity}" '
                     f'stroke-width="{2*r}" stroke-linecap="round" fill="none"/>')

sub = rng.permutation(len(T))[:2600]
inbox = lambda i: -0.05 < T[i,1] < 0.75 and 0.75 < T[i,2] < 1.55
dotcloud(p, [(T[i,1], T[i,2]) for i in sub if inbox(i) and not keepT[i]], GRAY, 0.35)
dotcloud(p, [(T[i,1], T[i,2]) for i in sub if inbox(i) and keepT[i]], GREEN, 0.8)
# the fitted plane as a line in this (y,z) slice at x = median x of inliers,
# drawn over the inliers' actual y-extent
xmed = np.median(T[keepT, 0])
ylo, yhi = np.quantile(T[keepT, 1], [0.01, 0.99])
yy = np.linspace(ylo - 0.03, yhi + 0.06, 50)
zz = (-dT - nT[0]*xmed - nT[1]*yy)/nT[2]
p.line(yy, zz, RED, 2.2, "6 4")
p.label(yhi + 0.02, (-dT - nT[0]*xmed - nT[1]*(yhi + 0.02))/nT[2],
        "fitted table plane", RED, 11, dx=10, dy=4, weight="bold")
p.label(0.33, 1.13, "objects on the table", GRAY, 11)
X0 = 648
fig.text(X0, 80, "cluttered-view fit:", 12.5, "start", BLACK, weight="bold")
fig.text(X0, 104, f"n = ({nT[0]:.3f}, {nT[1]:.3f}, {nT[2]:.3f})", 12, "start", RED)
fig.text(X0, 126, f"{keepT.sum()} / {len(T)} inliers", 12, "start", BLACK)
fig.text(X0, 148, "rms 4.2 mm", 12, "start", BLACK)
fig.text(X0, 186, "vs empty-table fit:", 12.5, "start", BLACK, weight="bold")
fig.text(X0, 210, "tilt 0.28°", 12, "start", GREEN)
fig.text(X0, 232, "offset 5.8 mm", 12, "start", GREEN)
fig.text(430, 342, "Gray = points more than 1 cm off the plane (objects, floor, background). The plane survives 76% of the view being 'not table'.", 11.5, "middle", GRAY)
print("03", fig.save(OUT + "03_table_fit.svg"))

# ---------- 04: CSE building, top view, sequential planes ----------
C = load("hw3/CSE.asc")
P = C.copy(); plane_pts = []; cols = [GRAY, BLUE, RED, PURPLE]
labels = []
for k in range(4):
    n, d, keep = ransac_plane(P, 0.15)
    plane_pts.append(P[keep]); labels.append((n, d, keep.sum()))
    P = P[~keep]
fig = Fig(860, 420)
p = Panel(fig, 60, 45, 560, 330, (-5, 40), (-25, 8),
          "CSE building, top view (x–y, meters): peel off planes one at a time",
          xticks=[0, 10, 20, 30, 40], yticks=[-20, -10, 0], ylabel="y (m)")
names = ["ground", "wall A", "wall B", "wall A′"]
for k in [1, 2, 3, 0]:
    Q = plane_pts[k]
    sub = Q[rng.permutation(len(Q))[:2400 if k else 1600]]
    op = 0.25 if k == 0 else 0.7
    pts = [(q[0], q[1]) for q in sub if -5 < q[0] < 40 and -25 < q[1] < 8]
    dotcloud(p, pts, cols[k], op, 1.3)
X0 = 648
fig.text(X0, 70, "sequential RANSAC:", 12.5, "start", BLACK, weight="bold")
for i, (nm, co) in enumerate(zip(names, cols)):
    n, d, cnt = labels[i]
    fig.text(X0, 100 + i*46, f"{i+1}. {nm}", 12.5, "start", co, weight="bold")
    fig.text(X0, 118 + i*46, f"{cnt:,} pts, n=({n[0]:+.2f},{n[1]:+.2f},{n[2]:+.2f})", 10.5, "start", GRAY)
aw = np.degrees(np.arccos(np.clip(abs(labels[0][0] @ labels[1][0]), -1, 1)))
sep = abs(abs(labels[1][1]) - abs(labels[3][1]))
fig.text(X0, 298, f"wall A ⊥ ground: {aw:.1f}°", 11.5, "start", BLACK)
fig.text(X0, 318, f"A′ ∥ A, {sep*100:.0f} cm behind", 11.5, "start", BLACK)
fig.text(X0, 334, "(recessed facade layer)", 11.5, "start", GRAY)
fig.text(430, 404, "Fit the dominant plane, remove its inliers, repeat. The ground takes 56% of the cloud; the walls emerge one by one from what's left.", 11.5, "middle", GRAY)
print("04", fig.save(OUT + "04_cse_planes.svg"))
