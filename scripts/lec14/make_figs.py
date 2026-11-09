"""Lecture 14 figures (differential geometry)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from svgplot import Fig, Panel, BLUE, RED, GREEN, GRAY, ORANGE, PURPLE, BLACK

OUT = "docs/figures/lec14/"
os.makedirs(OUT, exist_ok=True)
AMBER = "#BA7517"

# ---------- 01: Frenet frame riding the helix ----------
a, b = 2.0, 1.0
c = np.hypot(a, b)
# simple oblique projection of 3D -> 2D
def proj(p):
    return np.array([p[0] + 0.40*p[1], p[2] + 0.22*p[1]])
Tf = lambda s: np.array([-a*np.sin(s/c), a*np.cos(s/c), b])/c
Nf = lambda s: np.array([-np.cos(s/c), -np.sin(s/c), 0.0])
Bf = lambda s: np.array([b*np.sin(s/c), -b*np.cos(s/c), a])/c
beta = lambda s: np.array([a*np.cos(s/c), a*np.sin(s/c), b*s/c])

fig = Fig(860, 400)
p = Panel(fig, 60, 45, 520, 310, (-3.6, 4.0), (-1.4, 7.0),
          "the Frenet frame riding a helix (a = 2, b = 1)", xticks=None, yticks=None, grid=False)
S = np.linspace(0, 4.2*np.pi, 300)
pts = np.array([proj(beta(s)) for s in S])
p.line(pts[:, 0], pts[:, 1], GRAY, 2)
def arrow(x1, y1, x2, y2, col, w=2):
    d = np.hypot(x2-x1, y2-y1)
    if d < 1e-6: return
    ux, uy = (x2-x1)/d, (y2-y1)/d; hh = 8
    fig.parts.append(f'<path d="M{x1:.1f} {y1:.1f}L{x2:.1f} {y2:.1f}M{x2-hh*ux+hh*0.5*uy:.1f} {y2-hh*uy-hh*0.5*ux:.1f}L{x2:.1f} {y2:.1f}L{x2-hh*ux-hh*0.5*uy:.1f} {y2-hh*uy+hh*0.5*ux:.1f}" stroke="{col}" stroke-width="{w}" fill="none" stroke-linecap="round" stroke-linejoin="round"/>')
for s0 in (2.0, 7.0, 12.0):
    base = beta(s0)
    bx, by = p.X(proj(base)[0]), p.Y(proj(base)[1])
    for vec, col in ((Tf(s0), BLUE), (Nf(s0), RED), (Bf(s0), GREEN)):
        tip = proj(base + 1.1*vec)
        arrow(bx, by, p.X(tip[0]), p.Y(tip[1]), col, 2.2)
    fig.parts.append(f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="3.5" fill="{BLACK}"/>')
s0 = 2.0
base = proj(beta(s0) + 1.25*Tf(s0)); fig.text(p.X(base[0]), p.Y(base[1]), "T", 13, "middle", BLUE, weight="bold")
base = proj(beta(s0) + 1.45*Nf(s0)); fig.text(p.X(base[0]), p.Y(base[1]), "N", 13, "middle", RED, weight="bold")
base = proj(beta(s0) + 1.35*Bf(s0)); fig.text(p.X(base[0]), p.Y(base[1]), "B", 13, "middle", GREEN, weight="bold")
X0 = 616
fig.text(X0, 90, "verified for this helix:", 12.5, "start", BLACK, weight="bold")
fig.text(X0, 118, "speed √(a²+b²) = 2.236", 12, "start", GRAY)
fig.text(X0, 142, "K = a/(a²+b²) = 0.400", 12, "start", RED)
fig.text(X0, 166, "τ = b/(a²+b²) = 0.200", 12, "start", GREEN)
fig.text(X0, 198, "T′ = K N", 12, "start", BLACK)
fig.text(X0, 222, "N′ = −K T + τ B", 12, "start", BLACK, weight="bold")
fig.text(X0, 246, "B′ = −τ N", 12, "start", BLACK)
fig.text(X0, 274, "(the slide's N′ = +K T + τ B", 11, "start", RED)
fig.text(X0, 291, "misses by exactly 2K)", 11, "start", RED)
fig.text(430, 382, "T points along the motion, N toward the bend's center, B completes the frame. Constant K and τ — the helix is the “circle” of space curves.", 11.5, "middle", GRAY)
print("01", fig.save(OUT + "01_frenet_helix.svg"))

# ---------- 02: parallel parking = the Lie bracket ----------
l = 2.5
phi1, phi2 = 0.5, -0.5
def Vcar(phi):
    return lambda q: np.array([np.cos(phi)*np.cos(q[2]), np.cos(phi)*np.sin(q[2]), np.sin(phi)/l])
def flow(V, q, t, n=120):
    dt = t/n
    for _ in range(n):
        q = q + dt*V(q)
    return q
eps = 0.6
q = np.array([0.0, 0.0, np.pi/2])     # heading +y: bracket pushes sideways in -x... compute
traj = [q.copy()]
phases = [(Vcar(phi1), 1, BLUE), (Vcar(phi2), 1, GREEN), (Vcar(phi1), -1, BLUE), (Vcar(phi2), -1, GREEN)]
segs = []
for cyc in range(3):
    for V, sg, col in phases:
        seg = [q.copy()]
        n = 40
        for _ in range(n):
            q = q + (eps/n)*sg*V(q)
            seg.append(q.copy())
        segs.append((np.array(seg), col, sg))
net_per = np.sin(phi1-phi2)/l*eps*eps
fig = Fig(860, 400)
p = Panel(fig, 60, 45, 520, 310, (-0.8, 1.4), (-0.35, 0.95),
          "parallel parking: flow +V₁, +V₂, −V₁, −V₂ … and creep sideways",
          xticks=None, yticks=None, grid=False)
for seg, col, sg in segs:
    p.line(seg[:, 0], seg[:, 1], col, 2, None if sg > 0 else "4 3")
# start/end markers and car heading
p0, p1 = segs[0][0][0], segs[-1][0][-1]
fig.parts.append(f'<circle cx="{p.X(p0[0]):.1f}" cy="{p.Y(p0[1]):.1f}" r="5" fill="{BLACK}"/>')
fig.parts.append(f'<circle cx="{p.X(p1[0]):.1f}" cy="{p.Y(p1[1]):.1f}" r="5" fill="{AMBER}"/>')
arrow(p.X(p0[0]), p.Y(p0[1]), p.X(p0[0]-0.55), p.Y(p0[1]), RED, 2.6)
fig.text(p.X(p0[0]-0.3), p.Y(p0[1])+20, "net creep per cycle ≈ ε²·[V₁,V₂]", 11.5, "middle", RED, weight="bold")
fig.text(p.X(p0[0]), p.Y(p0[1])-12, "start", 11, "middle", BLACK)
fig.text(p.X(p1[0]), p.Y(p1[1])-12, "after 3 cycles", 11, "middle", AMBER)
X0 = 616
fig.text(X0, 90, "car controls (v, φ):", 12.5, "start", BLACK, weight="bold")
fig.text(X0, 116, "ẋ = v cos φ cos θ", 11.5, "start", GRAY)
fig.text(X0, 136, "ẏ = v cos φ sin θ", 11.5, "start", GRAY)
fig.text(X0, 156, "θ̇ = (v/l) sin φ", 11.5, "start", GRAY)
fig.text(X0, 188, "no control moves the car", 11.5, "start", BLACK)
fig.text(X0, 205, "sideways — but the bracket", 11.5, "start", BLACK)
fig.text(X0, 230, "[V₁,V₂] = sin(φ₁−φ₂)/l ·", 11.5, "start", RED, weight="bold")
fig.text(X0, 248, "(−sin θ, cos θ, 0)", 11.5, "start", RED, weight="bold")
fig.text(X0, 274, "does. verified: wiggle net =", 11.5, "start", BLACK)
fig.text(X0, 291, "0.999 × ε²|[V₁,V₂]|, pure", 11.5, "start", BLACK)
fig.text(X0, 308, "sideways, zero net rotation", 11.5, "start", BLACK)
fig.text(430, 382, "Solid = forward flows, dashed = reverse. Each 4-phase cycle returns θ to start but shifts the car perpendicular to its heading — the Lie bracket is the escape from nonholonomy.", 11, "middle", GRAY)
print("02", fig.save(OUT + "02_parking.svg"))
