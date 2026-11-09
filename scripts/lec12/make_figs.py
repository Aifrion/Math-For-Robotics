"""Lecture 12 figures (configuration space)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from svgplot import Fig, Panel, BLUE, RED, GREEN, GRAY, ORANGE, PURPLE, BLACK

OUT = "docs/figures/lec12/"
os.makedirs(OUT, exist_ok=True)
AMBER = "#BA7517"

# ---------- 01: inflation + why rotation breaks it ----------
fig = Fig(860, 360)
# panel a: workspace
p = Panel(fig, 50, 45, 240, 240, (0, 3), (0, 3), "workspace: disk robot, real obstacle",
          xticks=None, yticks=None, grid=False)
p.f.parts.append(f'<rect x="{p.X(1):.0f}" y="{p.Y(1.8):.0f}" width="{p.X(2)-p.X(1):.0f}" height="{p.Y(1)-p.Y(1.8):.0f}" fill="{BLACK}" opacity="0.8"/>')
p.f.parts.append(f'<circle cx="{p.X(2.45):.0f}" cy="{p.Y(2.3):.0f}" r="{(p.X(2.95)-p.X(2.45)):.0f}" fill="{AMBER}" opacity="0.9"/>')
p.f.parts.append(f'<circle cx="{p.X(2.45):.0f}" cy="{p.Y(2.3):.0f}" r="3" fill="{BLACK}"/>')
p.label(2.45, 2.72, "r = 0.5", BLACK, 10.5, "middle")
# panel b: c-space, rounded inflation
q = Panel(fig, 320, 45, 240, 240, (0, 3), (0, 3), "C-space: point robot, inflated obstacle",
          xticks=None, yticks=None, grid=False)
R_ = 0.5
x0, x1, y0, y1 = 1, 2, 1, 1.8
d = f"M{q.X(x0):.0f} {q.Y(y1+R_):.0f}H{q.X(x1):.0f}"
d += f"A{q.X(R_)-q.X(0):.0f} {q.X(R_)-q.X(0):.0f} 0 0 1 {q.X(x1+R_):.0f} {q.Y(y1):.0f}"
d += f"V{q.Y(y0):.0f}A{q.X(R_)-q.X(0):.0f} {q.X(R_)-q.X(0):.0f} 0 0 1 {q.X(x1):.0f} {q.Y(y0-R_):.0f}"
d += f"H{q.X(x0):.0f}A{q.X(R_)-q.X(0):.0f} {q.X(R_)-q.X(0):.0f} 0 0 1 {q.X(x0-R_):.0f} {q.Y(y0):.0f}"
d += f"V{q.Y(y1):.0f}A{q.X(R_)-q.X(0):.0f} {q.X(R_)-q.X(0):.0f} 0 0 1 {q.X(x0):.0f} {q.Y(y1+R_):.0f}Z"
q.f.parts.append(f'<path d="{d}" fill="{GREEN}" opacity="0.25" stroke="{GREEN}" stroke-width="1.5"/>')
q.f.parts.append(f'<rect x="{q.X(x0):.0f}" y="{q.Y(y1):.0f}" width="{q.X(x1)-q.X(x0):.0f}" height="{q.Y(y0)-q.Y(y1):.0f}" fill="{BLACK}" opacity="0.8"/>')
# sharp corner overcount
for cx, cy in ((x0, y0), (x0, y1), (x1, y0), (x1, y1)):
    sx = -1 if cx == x0 else 1; sy = -1 if cy == y0 else 1
    q.f.parts.append(f'<path d="M{q.X(cx):.0f} {q.Y(cy+sy*R_):.0f}L{q.X(cx+sx*R_):.0f} {q.Y(cy+sy*R_):.0f}L{q.X(cx+sx*R_):.0f} {q.Y(cy):.0f}" fill="none" stroke="{RED}" stroke-width="1.2" stroke-dasharray="3 3"/>')
q.f.parts.append(f'<circle cx="{q.X(2.45):.0f}" cy="{q.Y(2.3):.0f}" r="4" fill="{AMBER}"/>')
q.label(1.5, 0.28, "corners are ROUND:", RED, 10.5, "middle")
q.label(1.5, 0.08, "square corners over-forbid (4−π)r² = 0.21", RED, 10, "middle")
# panel c: rotation curve
r2 = Panel(fig, 610, 45, 220, 240, (0, 90), (0, 2.6), "a 2.4 × 0.6 robot vs a 1.0 gap",
           xticks=[0, 45, 90], yticks=[0, 1, 2], xlabel="θ (deg)")
ts = np.linspace(0, np.pi/2, 80)
r2.line(np.degrees(ts), 2.4*np.abs(np.cos(ts)) + 0.6*np.abs(np.sin(ts)), BLUE, 2.2)
r2.hline(1.0, RED, 1.5)
r2.label(4, 1.12, "gap", RED, 10.5)
r2.band(np.degrees(ts[ts > np.radians(80.2)]), 0*ts[ts > np.radians(80.2)], 2.6 + 0*ts[ts > np.radians(80.2)], GREEN, 0.15)
r2.label(58, 0.32, "fits only for", GREEN, 10.5)
r2.label(58, 0.12, "θ > 80.2°", GREEN, 10.5, weight="bold")
fig.text(430, 340, "Point-robot trick: inflate the obstacle (rounded corners!). But inflation depends on θ — (x, y) alone is not the configuration.", 11.5, "middle", GRAY)
print("01", fig.save(OUT + "01_inflation.svg"))

# ---------- 02: arm C-space obstacle on the torus ----------
L1 = L2 = 1.0
OB = np.array([0.9, 0.75]); OR = 0.30
def seg_hits_disk(p_, q_, c, r):
    d_ = q_ - p_; t = np.clip(np.dot(c - p_, d_)/max(np.dot(d_, d_), 1e-12), 0, 1)
    return np.linalg.norm(p_ + t*d_ - c) < r
def arm_collides(a, b):
    j0 = np.zeros(2); j1 = np.array([np.cos(a), np.sin(a)])
    j2 = j1 + [np.cos(a+b), np.sin(a+b)]
    return seg_hits_disk(j0, j1, OB, OR) or seg_hits_disk(j1, j2, OB, OR)

fig = Fig(860, 420)
p = Panel(fig, 55, 45, 330, 330, (-2.2, 2.2), (-2.2, 2.2), "workspace: 2-link arm + obstacle",
          xticks=None, yticks=None, grid=False)
p.f.parts.append(f'<circle cx="{p.X(OB[0]):.0f}" cy="{p.Y(OB[1]):.0f}" r="{p.X(OR)-p.X(0):.0f}" fill="{GRAY}" opacity="0.75"/>')
th = np.linspace(0, 2*np.pi, 60)
p.line(2*np.cos(th), 2*np.sin(th), GRAY, 0.7, "3 4")
p.f.parts.append(f'<rect x="{p.X(0)-10:.0f}" y="{p.Y(0)-5:.0f}" width="20" height="10" rx="2" fill="none" stroke="{GRAY}" stroke-width="1.5"/>')
p.label(0, -0.32, "base", GRAY, 10, "middle")
def draw_arm(a, b, col, lab, dash=None):
    j1 = np.array([np.cos(a), np.sin(a)]); j2 = j1 + [np.cos(a+b), np.sin(a+b)]
    extra = f' stroke-dasharray="{dash}"' if dash else ''
    p.f.parts.append(f'<path d="M{p.X(0):.0f} {p.Y(0):.0f}L{p.X(j1[0]):.0f} {p.Y(j1[1]):.0f}L{p.X(j2[0]):.0f} {p.Y(j2[1]):.0f}" stroke="{col}" stroke-width="4" fill="none" stroke-linecap="round" stroke-linejoin="round"{extra}/>')
    p.f.parts.append(f'<circle cx="{p.X(j1[0]):.0f}" cy="{p.Y(j1[1]):.0f}" r="4" fill="none" stroke="{col}" stroke-width="1.5"/>')
    p.f.parts.append(f'<circle cx="{p.X(j2[0]):.0f}" cy="{p.Y(j2[1]):.0f}" r="5" fill="{col}"/>')
    p.label(j2[0], j2[1]+0.22, lab, col, 12, "middle", weight="bold")
qA = (np.radians(120), np.radians(-20)); qB = (np.radians(-40), np.radians(30))
draw_arm(*qA, BLUE, "A")
draw_arm(*qB, GREEN, "B")
# right: c-space torus grid
q = Panel(fig, 470, 45, 330, 330, (-180, 180), (-180, 180), "C-space (α, β): the obstacle as seen by the JOINTS",
          xticks=[-180, 0, 180], yticks=[-180, 0, 180], xlabel="α (deg)", ylabel="β (deg)")
N = 110
A = np.linspace(-np.pi, np.pi, N, endpoint=False)
pts = []
for i, a in enumerate(A):
    for j, b in enumerate(A):
        if arm_collides(a, b):
            pts.append((np.degrees(a), np.degrees(b)))
seen = set(); dd = []
for a, b in pts:
    key = (round(q.X(a)), round(q.Y(b)))
    if key in seen: continue
    seen.add(key)
    dd.append(f"M{key[0]} {key[1]}h.1")
q.f.parts.append(f'<path d="{"".join(dd)}" stroke="{GRAY}" stroke-opacity="0.6" stroke-width="3.2" stroke-linecap="round" fill="none"/>')
q.f.parts.append(f'<circle cx="{q.X(np.degrees(qA[0])):.0f}" cy="{q.Y(np.degrees(qA[1])):.0f}" r="6" fill="{BLUE}"/>')
q.f.parts.append(f'<circle cx="{q.X(np.degrees(qB[0])):.0f}" cy="{q.Y(np.degrees(qB[1])):.0f}" r="6" fill="{GREEN}"/>')
q.label(np.degrees(qA[0]), np.degrees(qA[1])+14, "q_A", BLUE, 12, "middle", weight="bold")
q.label(np.degrees(qB[0])+8, np.degrees(qB[1])+10, "q_B", GREEN, 12, weight="bold")
# a wrapped path: straight-ish going right from qA, wrapping at 180 to -180, to qB
path = [(np.degrees(qA[0]), np.degrees(qA[1])), (178, -5), (-178, 8), (np.degrees(qB[0]), np.degrees(qB[1]))]
dpa = ""
prev = None
for a, b in path:
    if prev is not None and abs(a - prev[0]) > 180:
        dpa += f"L{q.X(180 if prev[0]>0 else -180):.0f} {q.Y((b+prev[1])/2):.0f}"
        dpa += f"M{q.X(-180 if prev[0]>0 else 180):.0f} {q.Y((b+prev[1])/2):.0f}"
    dpa += ("M" if prev is None else "L") + f"{q.X(a):.0f} {q.Y(b):.0f}"
    prev = (a, b)
q.f.parts.append(f'<path d="{dpa}" stroke="{AMBER}" stroke-width="2.4" fill="none" stroke-dasharray="6 4"/>')
q.label(176, -45, "path wraps:", AMBER, 10.5, "end")
q.label(176, -63, "±180° are glued", AMBER, 10.5, "end")
fig.text(430, 414, "One gray disk becomes a 12.3% blob on the (α, β) torus. The A→B path survives only because the square's edges wrap.", 11, "middle", GRAY)
print("02", fig.save(OUT + "02_arm_cspace.svg"))
